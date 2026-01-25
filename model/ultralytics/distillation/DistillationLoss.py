import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import TaskAlignedAssigner,BboxLoss,xywh2xyxy,dist2bbox,make_anchors


class DistillationLoss(nn.Module):
    def __init__(self, model, tal_topk=10, teacher_model=None,
                 kdcls_weight=0.0, kddfl_weight=0.0, kdf_weight=0.0,  # Added kdf_weight
                 temperature=20.0):  # Default temperature, ensure it's > 0
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
        # Assuming model.reg_max is the direct argument for BboxLoss as per user's original code
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

        # Loss functions for distillation
        self.kdcls_loss_fn = nn.KLDivLoss(reduction='batchmean')  # For soft label classification
        self.kddfl_loss_fn = nn.MSELoss(reduction='mean')  # Changed from 'none' to 'mean' for direct averaging
        self.kdf_loss_fn = nn.MSELoss(reduction='mean')  # For feature map distillation

        self.temperature = max(temperature, 1e-9)  # Ensure temperature is positive to avoid division by zero

        # Names of the loss components for logging
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss",
                           "kdcls_loss", "kddfl_loss", "kdf_loss")

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

    def _distill_feature_loss(self, student_feats_list, teacher_feats_list):
        """Helper function to compute feature distillation loss."""
        loss_kdf = torch.tensor(0.0, device=self.device)
        if not self.kdf_weight > 0 or not student_feats_list or not teacher_feats_list:
            return loss_kdf

        num_levels_to_distill = min(len(student_feats_list), len(teacher_feats_list))
        if num_levels_to_distill == 0:
            return loss_kdf

        for i in range(num_levels_to_distill):
            s_feat = student_feats_list[i]
            t_feat = teacher_feats_list[i].detach()  # Detach teacher features

            # Align spatial dimensions if they differ (student to teacher)
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)

            # Channel alignment (if needed) should be handled by adapter layers in the student model itself.
            # Here, we assume channels are compatible or already adapted.
            if s_feat.shape[1] != t_feat.shape[1]:
                # This is a basic adapter, ideally use a learned one in the model
                # For simplicity in loss, we might skip or use a simple conv here if absolutely necessary,
                # but it's better handled in the model architecture.
                # For now, we'll assume channels match or an adapter is in the student model.
                # If they don't match and no adapter, this will error or give bad results.
                # A common quick fix if channels differ and no adapter:
                # if s_feat.shape[1] > t_feat.shape[1]:
                #    s_feat = s_feat[:, :t_feat.shape[1], :, :]
                # elif t_feat.shape[1] > s_feat.shape[1]:
                #    t_feat = t_feat[:, :s_feat.shape[1], :, :]
                # This is a crude hack, proper adaptors are preferred.
                # For now, let's assume channels are aligned by the model design.
                pass

            loss_kdf += self.kdf_loss_fn(s_feat, t_feat)

        return loss_kdf / num_levels_to_distill if num_levels_to_distill > 0 else loss_kdf

    def __call__(self, preds_tuple, batch):
        # preds_tuple from DistillationModel.forward is (student_head_outputs_list, student_neck_features_list)
        student_head_outputs_list, student_neck_features_list = preds_tuple

        loss = torch.zeros(6, device=self.device)  # Now 6 loss components

        # Process student's head outputs (list of tensors from P3, P4, P5 heads)
        # This part is similar to how YOLOv8 standard loss processes head outputs.
        # student_head_outputs_list is used where 'student_feats' was used in the original code for head processing.
        student_pred_distri, student_pred_scores = torch.cat(
            [xi.view(student_head_outputs_list[0].shape[0], self.no, -1) for xi in student_head_outputs_list], 2
        ).split((self.reg_max * 4, self.nc), 1)

        student_pred_scores = student_pred_scores.permute(0, 2,
                                                          1).contiguous()  # (batch, num_anchors_total, num_classes)
        student_pred_distri = student_pred_distri.permute(0, 2, 1).contiguous()  # (batch, num_anchors_total, reg_max*4)

        dtype = student_pred_scores.dtype
        batch_size = student_pred_scores.shape[0]

        # Image size for scaling, derived from student's P3 head output shape
        # self.stride[0] corresponds to the stride of the P3 level
        img_spatial_shape = student_head_outputs_list[0].shape[2:]  # (H, W) of P3 feature map
        imgsz = torch.tensor(img_spatial_shape, device=self.device, dtype=dtype) * self.stride[0]

        # Generate anchor points and stride tensor based on student's head output feature maps
        anchor_points, stride_tensor = make_anchors(student_head_outputs_list, self.stride, 0.5)

        # Teacher model forward pass (if teacher exists)
        teacher_pred_scores_for_kd = None
        teacher_pred_distri_for_kd = None
        teacher_neck_features_list = None

        if self.teacher_model and (self.kdcls_weight > 0 or self.kddfl_weight > 0 or self.kdf_weight > 0):
            with torch.no_grad():  # Ensure no gradients for teacher
                # Assuming teacher_model's forward also returns (head_outputs_list, neck_features_list)
                teacher_head_outputs_list, teacher_neck_features_list_raw = self.teacher_model(batch["img"])

                if teacher_head_outputs_list:  # Process head outputs for KD
                    _teacher_pred_distri, _teacher_pred_scores = torch.cat(
                        [xi.view(teacher_head_outputs_list[0].shape[0], self.no, -1) for xi in
                         teacher_head_outputs_list], 2
                    ).split((self.reg_max * 4, self.nc), 1)
                    teacher_pred_scores_for_kd = _teacher_pred_scores.permute(0, 2, 1).contiguous().detach()
                    teacher_pred_distri_for_kd = _teacher_pred_distri.permute(0, 2, 1).contiguous().detach()

                if teacher_neck_features_list_raw:  # Store neck features for KD
                    teacher_neck_features_list = [feat.detach() for feat in teacher_neck_features_list_raw]

        # Preprocess ground truth targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # Scale bboxes
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # (bs, max_num_gt, 1), (bs, max_num_gt, 4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # Mask for valid ground truths

        # Label assignment using student's predictions
        student_pred_bboxes_for_assign = self.bbox_decode(anchor_points, student_pred_distri)  # Decoded bboxes

        # assigner returns: target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            student_pred_scores.detach().sigmoid(),  # Use detached, sigmoid-activated scores for assignment
            (student_pred_bboxes_for_assign.detach() * stride_tensor).type(gt_bboxes.dtype),
            # Student bboxes for assignment
            anchor_points * stride_tensor,  # Anchor points scaled to image dimensions
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1e-6)  # Sum of target scores for normalization, avoid div by zero

        # --- Standard Detection Losses (Student vs. Assigned GT) ---
        # 1. Classification Loss (BCE)
        loss[1] = self.bce(student_pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # 2. Bbox Regression Loss & DFL Loss
        if fg_mask.sum() > 0:  # Only if there are positive assignments
            target_bboxes /= stride_tensor  # Scale target bboxes to feature map level
            loss[0], loss[2] = self.bbox_loss(
                student_pred_distri, student_pred_bboxes_for_assign, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:  # No positive assignments, set bbox and dfl loss to 0
            loss[0] = torch.tensor(0.0, device=self.device)
            loss[2] = torch.tensor(0.0, device=self.device)

        # Apply YOLO hyperparameter gains to standard losses
        loss[0] *= self.hyp.box  # Box loss gain
        loss[1] *= self.hyp.cls  # Classification loss gain
        loss[2] *= self.hyp.dfl  # DFL loss gain

        # --- Knowledge Distillation Losses ---
        # Only compute KD losses if there are positive assignments (fg_mask) and teacher outputs are available
        if fg_mask.sum() > 0 and self.teacher_model:
            # 3. KD Classification Loss (KLDiv)
            if self.kdcls_weight > 0 and teacher_pred_scores_for_kd is not None:
                student_scores_fg = student_pred_scores[fg_mask].reshape(-1, self.nc)
                teacher_scores_fg = teacher_pred_scores_for_kd[fg_mask].reshape(-1, self.nc)

                student_log_softmax = F.log_softmax(student_scores_fg / self.temperature, dim=-1)
                teacher_softmax = F.softmax(teacher_scores_fg / self.temperature, dim=-1)

                kdcls_loss_val = self.kdcls_loss_fn(student_log_softmax, teacher_softmax) * (self.temperature ** 2)
                loss[3] = kdcls_loss_val * self.kdcls_weight

            # 4. KD DFL/Regression Loss (MSE)
            if self.kddfl_weight > 0 and teacher_pred_distri_for_kd is not None:
                student_distri_fg = student_pred_distri[fg_mask].reshape(-1, self.reg_max * 4)
                teacher_distri_fg = teacher_pred_distri_for_kd[fg_mask].reshape(-1, self.reg_max * 4)

                # kddfl_loss_fn is MSELoss with reduction='mean'
                kddfl_loss_val = self.kddfl_loss_fn(student_distri_fg, teacher_distri_fg)
                loss[4] = kddfl_loss_val * self.kddfl_weight

        # 5. KD Feature Loss (MSE) - Calculated regardless of fg_mask for whole feature maps
        if self.kdf_weight > 0 and self.teacher_model and student_neck_features_list and teacher_neck_features_list:
            kdf_loss_val = self._distill_feature_loss(student_neck_features_list, teacher_neck_features_list)
            loss[5] = kdf_loss_val * self.kdf_weight
        else:  # Ensure loss[5] is zero if not computed
            loss[5] = torch.tensor(0.0, device=self.device)

        total_loss = loss.sum()
        # Return total loss scaled by batch size, and detached individual loss components for logging
        return total_loss * batch_size, loss.detach()

