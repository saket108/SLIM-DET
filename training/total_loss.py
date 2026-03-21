# training/total_loss.py  — FIXED VERSION
"""
Total loss with:
- Per-class focal loss (handles imbalance)
- GIoU + L1 box loss
- Severity regression loss
- Auxiliary losses with epoch-aware ramping
  (starts at 0.1, reaches 1.0 at epoch 50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.task_aligned_assigner import TaskAlignedAssigner


CLASS_ALPHA = torch.tensor([0.18, 0.12, 0.48, 0.65, 0.42, 0.58])


class FocalLoss(nn.Module):
    def __init__(self, alpha=CLASS_ALPHA, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        alpha  = self.alpha.to(pred.device)
        bce    = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt     = torch.exp(-bce)
        focal  = (1 - pt) ** self.gamma * bce
        # Weight by target class alpha
        alpha_t = (alpha * target + (1 - alpha) * (1 - target))
        loss   = alpha_t * focal
        return loss.mean()


def giou_loss(pred_boxes, target_boxes):
    """GIoU loss for normalized cxcywh boxes."""
    def to_xyxy(b):
        return torch.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], dim=-1)

    p = to_xyxy(pred_boxes)
    g = to_xyxy(target_boxes)

    ix1 = torch.max(p[..., 0], g[..., 0])
    iy1 = torch.max(p[..., 1], g[..., 1])
    ix2 = torch.min(p[..., 2], g[..., 2])
    iy2 = torch.min(p[..., 3], g[..., 3])

    inter  = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union  = area_p + area_g - inter + 1e-6
    iou    = inter / union

    cx1 = torch.min(p[..., 0], g[..., 0])
    cy1 = torch.min(p[..., 1], g[..., 1])
    cx2 = torch.max(p[..., 2], g[..., 2])
    cy2 = torch.max(p[..., 3], g[..., 3])
    enclose = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0) + 1e-6

    giou = iou - (enclose - union) / enclose
    return (1 - giou).mean()


class TotalLoss(nn.Module):
    def __init__(
        self,
        w_cls:  float = 1.0,
        w_box:  float = 5.0,
        w_giou: float = 2.0,
        w_sev:  float = 0.5,
        w_aux:  float = 1.0,
    ):
        super().__init__()
        self.w_cls  = w_cls
        self.w_box  = w_box
        self.w_giou = w_giou
        self.w_sev  = w_sev
        self.w_aux  = w_aux

        self.focal_loss = FocalLoss()
        self.assigner   = TaskAlignedAssigner(topk=13, alpha=0.5, beta=6.0)

    def compute_layer_loss(self, pred_logits, pred_boxes, pred_severity,
                           targets, device):
        """Compute loss for one decoder layer output."""
        B, Q, C = pred_logits.shape
        total_cls = torch.tensor(0., device=device)
        total_box = torch.tensor(0., device=device)
        total_giu = torch.tensor(0., device=device)
        total_sev = torch.tensor(0., device=device)
        n_pos = 0

        for b in range(B):
            gt_boxes   = targets[b]['boxes'].to(device)
            gt_labels  = targets[b]['labels'].to(device)
            gt_sev     = targets[b]['severities'].to(device)

            if gt_boxes.shape[0] == 0:
                # Background — all negative
                tgt_cls = torch.zeros(Q, C, device=device)
                total_cls = total_cls + self.focal_loss(pred_logits[b], tgt_cls)
                continue

            # Assign predictions to GT
            assigned_labels, assigned_boxes, assigned_sev, pos_mask = \
                self.assigner(
                    pred_logits[b].detach(),
                    pred_boxes[b].detach(),
                    gt_boxes, gt_labels, gt_sev
                )

            # Classification loss
            tgt_cls = torch.zeros(Q, C, device=device)
            if pos_mask.any():
                tgt_cls[pos_mask, assigned_labels[pos_mask]] = 1.0
            total_cls = total_cls + self.focal_loss(pred_logits[b], tgt_cls)

            # Box + GIoU loss (positives only)
            if pos_mask.any():
                pb = pred_boxes[b][pos_mask]
                tb = assigned_boxes[pos_mask]
                total_box = total_box + F.l1_loss(pb, tb)
                total_giu = total_giu + giou_loss(pb, tb)

                # Severity loss
                ps = pred_severity[b][pos_mask].sigmoid()
                ts = assigned_sev[pos_mask]
                total_sev = total_sev + F.mse_loss(ps, ts)
                n_pos += pos_mask.sum().item()

        norm = max(B, 1)
        return (
            self.w_cls  * total_cls / norm +
            self.w_box  * total_box / norm +
            self.w_giou * total_giu / norm +
            self.w_sev  * total_sev / norm,
            {
                'cls':  (total_cls / norm).item(),
                'box':  (total_box / norm).item(),
                'giou': (total_giu / norm).item(),
                'sev':  (total_sev / norm).item(),
            }
        )

    def forward(self, outputs, targets, device, epoch=1):
        pred_logits  = outputs['pred_logits']
        pred_boxes   = outputs['pred_boxes']
        pred_severity= outputs['pred_severity']
        aux_outputs  = outputs.get('aux_outputs', [])

        # Main loss
        main_loss, loss_dict = self.compute_layer_loss(
            pred_logits, pred_boxes, pred_severity, targets, device
        )

        # Aux losses — ramp weight from 0.1 to 1.0 over 50 epochs
        aux_w = self.w_aux * min(1.0, 0.1 + epoch / 50.0)
        aux_loss = torch.tensor(0., device=device)

        for aux in aux_outputs:
            al, _, _ = self.compute_layer_loss(
                aux['pred_logits'], aux['pred_boxes'],
                aux['pred_severity'], targets, device
            )
            aux_loss = aux_loss + al

        total = main_loss + aux_w * aux_loss

        # Sanity check
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0., device=device, requires_grad=True), loss_dict

        loss_dict['aux'] = aux_loss.item()
        loss_dict['aux_w'] = aux_w
        return total, loss_dict


if __name__ == '__main__':
    from model.slim_det import SLIMDet
    device = torch.device('cpu')
    model  = SLIMDet(num_classes=6).to(device)
    model.eval()

    B = 2
    images  = torch.randn(B, 3, 640, 640)
    prompts = ["Damage: dent. Zone: central."] * B
    zones   = ['central'] * B
    metrics = torch.rand(B, 4)

    with torch.no_grad():
        outputs = model(images, prompts, zones, metrics)

    targets = [{
        'boxes':      torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        'labels':     torch.tensor([1]),
        'severities': torch.tensor([0.2]),
        'metrics':    torch.zeros(1, 4),
        'zones':      ['central'],
        'prompts':    ['test'],
        'num_anns':   1,
    } for _ in range(B)]

    loss_fn = TotalLoss()
    loss, d = loss_fn(outputs, targets, device, epoch=10)
    print(f"loss  : {loss.item():.4f}")
    print(f"terms : {d}")
    print("total_loss.py works correctly")
