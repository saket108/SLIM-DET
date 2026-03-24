"""Quality-target helpers for SLIM-Det detection heads."""

import torch
import torch.nn.functional as F


def matched_box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU for matched boxes in normalized cx, cy, w, h format."""
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.new_zeros((0,))

    pred_xyxy = torch.stack([
        pred_boxes[:, 0] - pred_boxes[:, 2] / 2,
        pred_boxes[:, 1] - pred_boxes[:, 3] / 2,
        pred_boxes[:, 0] + pred_boxes[:, 2] / 2,
        pred_boxes[:, 1] + pred_boxes[:, 3] / 2,
    ], dim=1)
    target_xyxy = torch.stack([
        target_boxes[:, 0] - target_boxes[:, 2] / 2,
        target_boxes[:, 1] - target_boxes[:, 3] / 2,
        target_boxes[:, 0] + target_boxes[:, 2] / 2,
        target_boxes[:, 1] + target_boxes[:, 3] / 2,
    ], dim=1)

    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) * (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)
    union = pred_area + target_area - inter
    return inter / union.clamp(min=1e-6)


def quality_bce_loss(
    pred_quality_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Supervise quality logits with the matched IoU target."""
    if pred_quality_logits.numel() == 0:
        return pred_quality_logits.sum() * 0.0

    with torch.no_grad():
        iou_targets = matched_box_iou(pred_boxes, target_boxes).clamp_(0.0, 1.0)

    return F.binary_cross_entropy_with_logits(
        pred_quality_logits,
        iou_targets,
        reduction='mean',
    )
