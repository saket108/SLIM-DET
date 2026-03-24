"""Detection metric helpers shared by dense baselines."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between normalized cxcywh boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=boxes1.dtype,
            device=boxes1.device,
        )

    def to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
            ],
            dim=1,
        )

    b1 = to_xyxy(boxes1)
    b2 = to_xyxy(boxes2)

    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP from the precision envelope."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_predictions(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[int, dict[str, float | int]]:
    """Return per-class AP, precision, recall, and F1."""
    class_metrics: dict[int, dict[str, float | int]] = {}

    for cls_id in range(num_classes):
        detections: list[tuple[float, int]] = []
        tp_count = 0
        fp_count = 0
        n_gt = 0

        for preds, targets in zip(all_preds, all_targets):
            gt_boxes = targets["boxes"]
            gt_labels = targets["labels"]

            gt_mask = gt_labels == cls_id
            gt_boxes_cls = gt_boxes[gt_mask]
            n_gt += gt_boxes_cls.shape[0]

            pred_mask = preds["labels"] == cls_id
            pred_boxes = preds["boxes"][pred_mask]
            pred_scores = preds["confidences"][pred_mask]

            if pred_scores.numel() > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

            matched = torch.zeros(gt_boxes_cls.shape[0], dtype=torch.bool)

            for idx, pred_box in enumerate(pred_boxes):
                score = float(pred_scores[idx])
                is_true_positive = 0

                if gt_boxes_cls.shape[0] > 0:
                    ious = box_iou_cxcywh(pred_box.unsqueeze(0), gt_boxes_cls)[0]
                    best_iou, best_j = ious.max(0)
                    if best_iou >= iou_threshold and not matched[best_j]:
                        matched[best_j] = True
                        is_true_positive = 1

                detections.append((score, is_true_positive))
                tp_count += is_true_positive
                fp_count += 1 - is_true_positive

        fn_count = max(n_gt - tp_count, 0)
        precision_value = tp_count / max(tp_count + fp_count, 1)
        recall_value = tp_count / max(n_gt, 1)
        f1_value = (
            2 * precision_value * recall_value / max(precision_value + recall_value, 1e-12)
            if (precision_value + recall_value) > 0
            else 0.0
        )

        if detections and n_gt > 0:
            score_arr = np.array([score for score, _ in detections], dtype=np.float32)
            tp_arr = np.array([tp for _, tp in detections], dtype=np.float32)
            order = score_arr.argsort()[::-1]
            tp_arr = tp_arr[order]

            cum_tp = np.cumsum(tp_arr)
            cum_fp = np.cumsum(1.0 - tp_arr)
            recall_curve = cum_tp / max(n_gt, 1)
            precision_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
            ap_value = compute_ap(recall_curve, precision_curve)
        else:
            ap_value = 0.0

        class_metrics[cls_id] = {
            "ap": float(ap_value),
            "precision": float(precision_value),
            "recall": float(recall_value),
            "f1": float(f1_value),
            "tp": int(tp_count),
            "fp": int(fp_count),
            "fn": int(fn_count),
            "num_gt": int(n_gt),
        }

    return class_metrics


def summarize_metrics(class_metrics: dict[int, dict[str, float | int]]) -> dict[str, float | int]:
    macro_precision = np.mean([metrics["precision"] for metrics in class_metrics.values()])
    macro_recall = np.mean([metrics["recall"] for metrics in class_metrics.values()])
    macro_f1 = np.mean([metrics["f1"] for metrics in class_metrics.values()])

    total_tp = sum(int(metrics["tp"]) for metrics in class_metrics.values())
    total_fp = sum(int(metrics["fp"]) for metrics in class_metrics.values())
    total_fn = sum(int(metrics["fn"]) for metrics in class_metrics.values())

    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = (
        2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
    }


def aggregate_map5095(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    num_classes: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, dict[str, float | int]], dict[str, float | int]]:
    """Compute AP50, AP50-95, and PR summary metrics."""
    ap50_metrics = evaluate_predictions(
        all_preds,
        all_targets,
        num_classes=num_classes,
        iou_threshold=0.5,
    )
    pr_metrics = evaluate_predictions(
        all_preds,
        all_targets,
        num_classes=num_classes,
        iou_threshold=0.5,
    )

    ap_all: dict[int, list[float]] = defaultdict(list)
    for iou_t in np.arange(0.5, 1.0, 0.05):
        threshold_metrics = evaluate_predictions(
            all_preds,
            all_targets,
            num_classes=num_classes,
            iou_threshold=float(iou_t),
        )
        for cls_id, metrics in threshold_metrics.items():
            ap_all[cls_id].append(float(metrics["ap"]))

    ap50 = {cls_id: float(metrics["ap"]) for cls_id, metrics in ap50_metrics.items()}
    ap5095 = {cls_id: float(np.mean(values)) for cls_id, values in ap_all.items()}
    summary = summarize_metrics(pr_metrics)
    return ap50, ap5095, pr_metrics, summary


def mean_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def print_results(
    ap50: dict[int, float],
    ap5095: dict[int, float],
    pr_metrics: dict[int, dict[str, float | int]],
    summary: dict[str, float | int],
    match_iou: float,
    class_names: list[str] | None = None,
) -> None:
    print("\n" + "=" * 90)
    print(
        f"{'Class':<20} {'AP50':>8} {'AP50-95':>10} "
        f"{'Prec':>8} {'Recall':>8}"
    )
    print("-" * 90)

    map50 = []
    map5095 = []
    num_classes = max(
        len(class_names or []),
        len(ap50),
        len(ap5095),
        len(pr_metrics),
    )

    for cls_id in range(num_classes):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        a50 = ap50.get(cls_id, 0.0)
        a95 = ap5095.get(cls_id, 0.0)
        pr = pr_metrics.get(cls_id, {})
        map50.append(a50)
        map5095.append(a95)
        print(
            f"  {name:<18} {a50:>8.3f} {a95:>10.3f} "
            f"{float(pr.get('precision', 0.0)):>8.3f} "
            f"{float(pr.get('recall', 0.0)):>8.3f}"
        )

    print("-" * 90)
    print(
        f"  {'mAP (macro)':<18} {np.mean(map50):>8.3f} {np.mean(map5095):>10.3f} "
        f"{float(summary['macro_precision']):>8.3f} "
        f"{float(summary['macro_recall']):>8.3f}"
    )
    print(
        f"  {'Micro @ IoU':<18} {'-':>8} {'-':>10} "
        f"{float(summary['micro_precision']):>8.3f} "
        f"{float(summary['micro_recall']):>8.3f}"
    )
    print("-" * 90)
    print(
        f"  Match IoU={match_iou:.2f} | "
        f"TP={int(summary['total_tp'])} FP={int(summary['total_fp'])} FN={int(summary['total_fn'])}"
    )
    print("=" * 90)
