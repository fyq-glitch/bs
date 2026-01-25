import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss, xywh2xyxy, dist2bbox, make_anchors
from ultralytics.nn.tasks import yaml_model_load, parse_model
from copy import deepcopy
import torch
from ultralytics.nn.modules import (
    OBB,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    Pose,
    RepConv,
    RepVGGDW,
    Segment,
    YOLOESegment,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
class BaseModel(torch.nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        extracted_features_indices = [16, 19, 22]
        extracted_features={}
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
            if extracted_features_indices and m.i in extracted_features_indices:
                extracted_features[m.i]=x
        return x,extracted_features

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DistillationLoss(nn.Module):
    def __init__(self, model, tal_topk=10, teacher_model=None,
                 kdcls_weight=0.0, kddfl_weight=0.0, kdf_weight=0.0, student_neck_channels_list=None,
                 teacher_neck_channels_list=None, distill_feature_indices=None, temperature=20.0):
        super().__init__()
        device = next(model.parameters()).device
        h = model.args  # Hyperparameters from student model
        m = model.model[-1]  # Detect() head module of student model

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # Strides of the student model's detection heads
        self.nc = m.nc  # Number of classes
        self.reg_max = m.reg_max  # For DFL
        self.no = m.nc + self.reg_max * 4  # Number of outputs per anchor
        self.device = device
        self.use_dfl = m.reg_max > 1

        # TaskAlignedAssigner for label assignment
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        # BboxLoss for bounding box regression loss
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        # Projection for DFL
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

        # Distillation parameters
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model.eval()  # Ensure teacher is in eval mode
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # Freeze teacher parameters

        self.kdcls_weight = kdcls_weight  # Weight for classification distillation
        self.kddfl_weight = kddfl_weight  # Weight for DFL/regression distillation
        self.kdf_weight = kdf_weight  # Weight for feature distillation
        self.adaptation_layers = nn.ModuleDict()
        self.teacher_distill_layer_indices = [16, 19, 22]
        self.distill_feature_indices = distill_feature_indices if distill_feature_indices is not None else []
        if self.kdf_weight > 0 and self.teacher_model and student_neck_channels_list and teacher_neck_channels_list and distill_feature_indices:
            if len(student_neck_channels_list) == len(teacher_neck_channels_list) >= len(self.distill_feature_indices):
                for i, feature_idx in enumerate(self.distill_feature_indices):
                    s_ch = student_neck_channels_list[feature_idx]
                    t_ch = teacher_neck_channels_list[feature_idx]
                    if s_ch != t_ch:
                        self.adaptation_layers[f'adapt_layer_{feature_idx}'] = nn.Conv2d(s_ch, t_ch, kernel_size=1,
                                                                                         bias=False)
            else:
                print("Channel list length mismatch between student and teacher")
        self.to(self.device)
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss", "kdcls_loss", "kddfl_loss", "kdf_loss"]
        # Loss functions for distillation
        self.kdcls_loss_fn = nn.KLDivLoss(reduction='batchmean')  # For soft label classification
        self.kddfl_loss_fn = nn.MSELoss(reduction='mean')  # Changed from 'none' to 'mean' for direct averaging
        self.kdf_loss_fn = nn.MSELoss(reduction='mean')  # For feature map distillation

        self.temperature = max(temperature, 1e-9)  # Ensure temperature is positive to avoid division by zero

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets: convert to tensor, scale coordinates."""
        nl, ne = targets.shape  # Number of labels, number of elements per label
        if nl == 0:  # No ground truth targets in the batch
            return torch.zeros(batch_size, 0, ne - 1 if ne > 0 else 0, device=self.device)

        i = targets[:, 0]  # Image index in batch
        _, counts = i.unique(return_counts=True)  # Number of targets per image
        counts = counts.to(dtype=torch.int32)
        # Initialize output tensor for preprocessed targets
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):  # Iterate through images in the batch
            matches = i == j
            n = matches.sum()
            if n:  # If there are targets for the current image
                out[j, :n] = targets[matches, 1:]  # Fill with class and bbox info
        # Scale bounding boxes (xywh to xyxy format, then scale)
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode bounding boxes from anchor points and predicted distribution (for DFL)."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels (reg_max * 4)
            # Reshape, apply softmax, and matmul with projection for DFL
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # Convert distance predictions to bounding box coordinates
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _distill_feature_loss(self, student_features_list, teacher_features_list):
        feature_loss = torch.tensor(0.0).to(self.device)
        num_features_distilled = 0
        if not student_features_list or not teacher_features_list:
            return feature_loss

        # Make sure feature_indices refers to valid indices in the feature lists
        for i, feature_idx in enumerate(self.distill_feature_indices):
            if feature_idx not in student_features_list or feature_idx not in teacher_features_list:
                print(f"Feature index {feature_idx} not found in feature dictionaries, skipping")
                continue

            s_feat = student_features_list[feature_idx].to(self.device)
            t_feat = teacher_features_list[feature_idx].to(self.device)
            s_feat_adapted = s_feat
            adapt_layer_name = f'adapt_layer_{feature_idx}'

            if adapt_layer_name in self.adaptation_layers:
                s_feat_adapted = self.adaptation_layers[adapt_layer_name](s_feat_adapted)

            if s_feat_adapted.shape[1] != t_feat.shape[1]:
                print(f"Warning: Channel mismatch for feature pair (index {feature_idx}) after adaptation. "
                      f"S_ch: {s_feat_adapted.shape[1]}, T_ch: {t_feat.shape[1]}. Skipping.")
                continue

            if s_feat_adapted.shape[2:] != t_feat.shape[2:]:
                print(f"Warning: Spatial dim mismatch for feature pair (index {feature_idx}). "
                      f"S_shape: {s_feat_adapted.shape[2:]}, T_shape: {t_feat.shape[2:]}. Applying F.interpolate.")
                s_feat_adapted = F.interpolate(s_feat_adapted, size=t_feat.shape[2:], mode='bilinear',
                                               align_corners=False)

            feature_loss += F.mse_loss(s_feat_adapted, t_feat, reduction='mean')
            num_features_distilled += 1

        if num_features_distilled > 0:
            return feature_loss / num_features_distilled
        else:
            return feature_loss

    def __call__(self, preds, batch):
        # Initialize loss tensor
        loss = torch.zeros(6, device=self.device)
        student_preds_raw, student_feats = preds[0], preds[0]
        student_neck_features = preds[1]
        student_pred_distri, student_pred_scores = torch.cat(
            [xi.view(student_feats[0].shape[0], self.no, -1) for xi in student_feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        student_pred_scores = student_pred_scores.permute(0, 2, 1).contiguous()  # (b, h*w, nc)
        student_pred_distri = student_pred_distri.permute(0, 2, 1).contiguous()  # (b, h*w, reg_max * 4)

        dtype = student_pred_scores.dtype
        batch_size = student_pred_scores.shape[0]
        imgsz = torch.tensor(student_feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[
            0]
        anchor_points, stride_tensor = make_anchors(student_feats, self.stride, 0.5)
        with torch.no_grad():
            output = self.teacher_model.forward_for_distill(batch["img"])
            teacher_preds_raw, teacher_feats = output[0], output[0]
            print(type(teacher_feats))
            print(type(student_feats))
            print(type(teacher_feats[0]))
            print(type(teacher_feats[0][0]))
            print(type(student_feats[0]))
            print(type(student_feats[0][0]))

            teacher_neck_features = preds[1]
            teacher_pred_distri, teacher_pred_scores = torch.cat(
                [xi.view(teacher_feats[0].shape[0], self.no, -1) for xi in teacher_feats], 2
            ).split((self.reg_max * 4, self.nc), 1)
            teacher_pred_scores = teacher_pred_scores.permute(0, 2, 1).contiguous()  # (b, h*w, nc)
            teacher_pred_distri = teacher_pred_distri.permute(0, 2, 1).contiguous()  # (b, h*w, reg_max * 4)

        # Process targets for loss calculation
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode student bounding box predictions
        student_pred_bboxes_for_assign = self.bbox_decode(anchor_points, student_pred_distri)

        # Assign targets using task-aligned assigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            student_pred_scores.detach().sigmoid(),
            (student_pred_bboxes_for_assign.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Calculate standard detection losses
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(student_pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                student_pred_distri,
                student_pred_bboxes_for_assign,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )

        # Apply loss weights
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # Calculate distillation losses for classification and regression
        if fg_mask.sum():
            # Check if shapes match before proceeding
            if (student_pred_scores.shape == teacher_pred_scores.shape and
                    student_pred_distri.shape == teacher_pred_distri.shape):

                student_scores_fg = student_pred_scores[fg_mask].reshape(-1, self.nc)
                teacher_scores_fg = teacher_pred_scores[fg_mask].reshape(-1, self.nc)
                student_distri_fg = student_pred_distri[fg_mask].reshape(-1, self.reg_max * 4)
                teacher_distri_fg = teacher_pred_distri[fg_mask].reshape(-1, self.reg_max * 4)

                # Apply temperature scaling for KD classification loss
                teacher_scores_soft = teacher_scores_fg / self.temperature
                student_scores_soft = student_scores_fg / self.temperature

                # Calculate KD classification loss
                kdcls_loss = self.kdcls_loss_fn(
                    F.log_softmax(student_scores_soft, dim=-1),
                    F.softmax(teacher_scores_soft, dim=-1)
                ) * (self.temperature ** 2)

                # Calculate KD distribution (regression) loss
                kddfl_loss = self.kddfl_loss_fn(student_distri_fg, teacher_distri_fg)

                # Apply distillation weights
                loss[3] = kdcls_loss * self.kdcls_weight
                loss[4] = kddfl_loss * self.kddfl_weight
            else:
                print(f"Shape mismatch! Student: {student_pred_scores.shape}, Teacher: {teacher_pred_scores.shape}")
                loss[3] = torch.tensor(0.0, device=self.device)
                loss[4] = torch.tensor(0.0, device=self.device)

        # Calculate feature distillation loss
        loss[5] = torch.tensor(0.0, device=self.device)
        if self.kdf_weight > 0 and self.teacher_model and student_neck_features and teacher_neck_features:
            kdf_loss_val = self._distill_feature_loss(student_neck_features, teacher_neck_features)
            loss[5] = kdf_loss_val * self.kdf_weight

        # Compute total loss
        total_loss = loss.sum()
        return total_loss * batch_size, loss.detach()