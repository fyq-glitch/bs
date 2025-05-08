import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import TaskAlignedAssigner,BboxLoss,xywh2xyxy,dist2bbox,make_anchors
class DistillationLoss(nn.Module):
    def __init__(self, model, tal_topk=10,teacher_model=None,kdcls_weight=0.0, kddfl_weight=0.0,temperature=0.0):
        super().__init__()
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        #蒸馏参数
        self.teacher_model = teacher_model
        self.kdcls_weight = kdcls_weight
        self.kddfl_weight = kddfl_weight
        self.kdcls_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.kddfl_loss_fn = nn.MSELoss(reduction='none')
        self.temperature = temperature
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "kdcls_loss", "kddfl_loss"

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # Apply softmax and matmul for DFL decoding
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)
        student_preds_raw, student_feats = preds[0], preds[1] if isinstance(preds, tuple) else preds
        student_pred_distri, student_pred_scores = torch.cat(
            [xi.view(student_feats[0].shape[0], self.no, -1) for xi in student_feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        student_pred_scores = student_pred_scores.permute(0, 2, 1).contiguous() # (b, h*w, nc)
        student_pred_distri = student_pred_distri.permute(0, 2, 1).contiguous() # (b, h*w, reg_max * 4)
        dtype = student_pred_scores.dtype
        batch_size = student_pred_scores.shape[0]
        imgsz = torch.tensor(student_feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(student_feats, self.stride, 0.5)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_feats = self.teacher_model(batch["img"])
            teacher_pred_distri, teacher_pred_scores = torch.cat(
                [xi.view(teacher_feats[1][0].shape[0], self.no, -1) for xi in teacher_feats[1]], 2
            ).split((self.reg_max * 4, self.nc), 1)
            teacher_pred_scores=teacher_pred_scores.permute(0, 2, 1).contiguous()
            teacher_pred_distri = teacher_pred_distri.permute(0, 2, 1).contiguous()
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        student_pred_bboxes_for_assign = self.bbox_decode(anchor_points, student_pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            student_pred_scores.detach().sigmoid(),
            (student_pred_bboxes_for_assign.detach() * stride_tensor).type(gt_bboxes.dtype), # Use student bboxes for assignment
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
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
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if fg_mask.sum():
            student_scores_fg = student_pred_scores[fg_mask].reshape(-1, self.nc)
            teacher_scores_fg = teacher_pred_scores[fg_mask].reshape(-1, self.nc)
            student_distri_fg = student_pred_distri[fg_mask].reshape(-1, self.reg_max * 4)
            teacher_distri_fg = teacher_pred_distri[fg_mask].reshape(-1, self.reg_max * 4)
            teacher_scores_soft = teacher_scores_fg / self.temperature
            student_scores_soft = student_scores_fg / self.temperature
            kdcls_loss = self.kdcls_loss_fn(
                F.log_softmax(student_scores_soft, dim=-1),
                F.softmax(teacher_scores_soft, dim=-1)
            ) * (self.temperature ** 2)
            kddfl_loss_per_sample = self.kddfl_loss_fn(student_distri_fg, teacher_distri_fg)
            kddfl_loss = kddfl_loss_per_sample.sum(dim=-1).mean()
            loss[3] = kdcls_loss * self.kdcls_weight
            loss[4] = kddfl_loss * self.kddfl_weight
        total_loss = loss.sum()
        return total_loss * batch_size, loss.detach()

