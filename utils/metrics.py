# utils/metrics.py
"""
Metric utilities for SLIM-Det evaluation.
Tracks running mAP, per-class AP, and loss averages.
"""

import numpy as np
from collections import defaultdict

from data.prompt_builder import CLASS_ID_TO_NAME


class AverageMeter:
    """Tracks running average of a scalar value."""
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __str__(self):
        return f"{self.name}={self.avg:.4f}"


class LossTracker:
    """Tracks multiple loss terms across an epoch."""
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, loss_dict, n=1):
        for k, v in loss_dict.items():
            self.meters[k].update(float(v), n)

    def summary(self):
        return {k: m.avg for k, m in self.meters.items()}

    def __str__(self):
        parts = [str(m) for m in self.meters.values()]
        return ' | '.join(parts)


class MetricLogger:
    """
    Logs train/val metrics per epoch.
    Prints a formatted table at the end of training.
    """
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, epoch, **kwargs):
        self.history['epoch'].append(epoch)
        for k, v in kwargs.items():
            self.history[k].append(float(v))

    def best(self, metric='val_loss', mode='min'):
        vals = self.history.get(metric, [])
        if not vals:
            return None, None
        idx = np.argmin(vals) if mode == 'min' else np.argmax(vals)
        return self.history['epoch'][idx], vals[idx]

    def print_summary(self):
        print("\n── Training Summary ──")
        epochs = self.history['epoch']
        keys   = [k for k in self.history if k != 'epoch']
        header = f"{'Epoch':>6} " + " ".join(f"{k:>12}" for k in keys)
        print(header)
        print("-" * len(header))
        for i, ep in enumerate(epochs):
            row = f"{ep:>6} "
            row += " ".join(
                f"{self.history[k][i]:>12.4f}" for k in keys
            )
            print(row)


def format_results_table(ap50_dict, ap5095_dict):
    """
    Format per-class AP results as a paper-ready table string.

    Args:
        ap50_dict   : {class_id: ap50_float}
        ap5095_dict : {class_id: ap5095_float}

    Returns:
        formatted string
    """
    lines = []
    lines.append("=" * 62)
    lines.append(f"  {'Class':<18} {'AP50':>10} {'AP50-95':>10} {'Note':>10}")
    lines.append("-" * 62)

    map50_vals   = []
    map5095_vals = []

    notes = {
        0: '',
        1: '',
        2: 'medium',
        3: '← rare',
        4: 'medium',
        5: 'hard',
    }

    for cls_id in range(6):
        name  = CLASS_ID_TO_NAME.get(cls_id, str(cls_id))
        a50   = ap50_dict.get(cls_id, 0.0)
        a95   = ap5095_dict.get(cls_id, 0.0)
        note  = notes.get(cls_id, '')
        map50_vals.append(a50)
        map5095_vals.append(a95)
        lines.append(f"  {name:<18} {a50:>10.3f} {a95:>10.3f} {note:>10}")

    lines.append("-" * 62)
    lines.append(
        f"  {'mAP':<18} "
        f"{np.mean(map50_vals):>10.3f} "
        f"{np.mean(map5095_vals):>10.3f}"
    )
    lines.append("=" * 62)
    return "\n".join(lines)


def compute_confusion_matrix(all_preds, all_targets, num_classes, conf=0.25, iou=0.5):
    """
    Compute confusion matrix for error analysis.
    Useful for understanding which classes get confused with each other.
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    # +1 for background class

    for preds, targets in zip(all_preds, all_targets):
        pred_boxes   = preds['boxes']
        pred_scores  = preds['scores']
        gt_boxes     = targets['boxes']
        gt_labels    = targets['labels']

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        # Get predicted class for each detection
        pred_conf, pred_cls = pred_scores.max(dim=-1)
        keep = pred_conf >= conf
        pred_boxes  = pred_boxes[keep]
        pred_cls    = pred_cls[keep]

        if len(pred_boxes) == 0:
            continue

        # Match via IoU
        from evaluate import box_iou
        ious = box_iou(pred_boxes, gt_boxes)

        for j in range(len(gt_boxes)):
            gt_cls    = gt_labels[j].item()
            best_iou  = ious[:, j].max().item() if len(ious) > 0 else 0.0
            best_pred = ious[:, j].argmax().item() if len(ious) > 0 else -1

            if best_iou >= iou:
                detected_cls = pred_cls[best_pred].item()
                matrix[gt_cls][detected_cls] += 1
            else:
                matrix[gt_cls][num_classes] += 1   # missed — background

    return matrix


if __name__ == '__main__':
    # Test AverageMeter
    meter = AverageMeter('loss')
    for v in [1.0, 2.0, 3.0]:
        meter.update(v)
    print(f"AverageMeter avg: {meter.avg}")   # 2.0

    # Test LossTracker
    tracker = LossTracker()
    tracker.update({'cls': 0.5, 'box': 1.2, 'giou': 0.8})
    tracker.update({'cls': 0.4, 'box': 1.0, 'giou': 0.7})
    print(f"LossTracker: {tracker}")

    # Test MetricLogger
    logger = MetricLogger()
    for ep in range(1, 6):
        logger.log(ep, train_loss=1.0/ep, val_loss=1.2/ep)
    best_ep, best_val = logger.best('val_loss', mode='min')
    print(f"Best epoch: {best_ep}, val_loss: {best_val:.4f}")

    # Test format_results_table
    ap50   = {i: 0.5 + i * 0.05 for i in range(6)}
    ap5095 = {i: 0.3 + i * 0.03 for i in range(6)}
    print(format_results_table(ap50, ap5095))
    print("metrics.py works correctly")
