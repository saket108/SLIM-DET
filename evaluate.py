# evaluate.py
"""
SLIM-Det evaluation script.
Computes mAP50, mAP50-95, and per-class AP on the test set.

Usage:
    python evaluate.py --checkpoint runs/slim_det/slim_det_best.pt
    python evaluate.py --checkpoint runs/slim_det/slim_det_best.pt --split test
    python evaluate.py --checkpoint runs/slim_det/slim_det_best.pt --prompt_mode cat_only
"""

import os
import argparse
import torch
import numpy as np
from collections import defaultdict

from model.slim_det  import SLIMDet
from data.loader     import build_val_loader, AircraftDataset
from data.prompt_builder import CLASS_ID_TO_NAME

# ── Dataset paths ─────────────────────────────────────────────
DATASET_ROOT = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset'
VAL_JSON     = os.path.join(DATASET_ROOT, 'Aircraft_val.json')
TEST_JSON    = os.path.join(DATASET_ROOT, 'Aircraft_test.json')
VAL_IMAGES   = os.path.join(DATASET_ROOT, 'images', 'val')
TEST_IMAGES  = os.path.join(DATASET_ROOT, 'images', 'test')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  type=str, default=None)
    p.add_argument('--split',       type=str, default='val',
                   choices=['val', 'test'])
    p.add_argument('--prompt_mode', type=str, default='full')
    p.add_argument('--batch',       type=int, default=4)
    p.add_argument('--imgsz',       type=int, default=640)
    p.add_argument('--conf',        type=float, default=0.25)
    p.add_argument('--iou_thresh',  type=float, default=0.5)
    p.add_argument('--workers',     type=int, default=0)
    return p.parse_args()


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes: [N, 4] in cx,cy,w,h normalized format
    """
    def to_xyxy(b):
        return torch.stack([
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ], dim=1)

    b1 = to_xyxy(boxes1)
    b2 = to_xyxy(boxes2)

    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = b1[:, 2] * b1[:, 3]
    area2 = b2[:, 2] * b2[:, 3]
    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


def compute_ap(recall, precision):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = precision[recall >= t]
        ap += prec.max() if len(prec) > 0 else 0.0
    return ap / 11.0


def evaluate_predictions(all_preds, all_targets, num_classes, iou_threshold=0.5):
    """
    Compute per-class AP and mAP.

    all_preds  : list of dicts {boxes:[N,4], scores:[N,C], labels:[N]}
    all_targets: list of dicts {boxes:[M,4], labels:[M]}
    """
    class_aps = {}

    for cls_id in range(num_classes):
        tp_list    = []
        score_list = []
        n_gt       = 0

        for preds, targets in zip(all_preds, all_targets):
            gt_boxes  = targets['boxes']
            gt_labels = targets['labels']

            # Filter GT for this class
            gt_mask  = gt_labels == cls_id
            gt_boxes_cls = gt_boxes[gt_mask]
            n_gt    += gt_boxes_cls.shape[0]

            # Filter predictions for this class
            pred_boxes  = preds['boxes']
            pred_scores = preds['scores'][:, cls_id]
            pred_mask   = pred_scores >= 0.0   # keep all, sort by score
            pred_boxes  = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]

            # Sort by score descending
            order       = pred_scores.argsort(descending=True)
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order].cpu().numpy()

            matched = torch.zeros(gt_boxes_cls.shape[0], dtype=torch.bool)

            for i, pb in enumerate(pred_boxes):
                score_list.append(pred_scores[i])

                if gt_boxes_cls.shape[0] == 0:
                    tp_list.append(0)
                    continue

                ious    = box_iou(pb.unsqueeze(0), gt_boxes_cls)[0]
                best_iou, best_j = ious.max(0) if len(ious) > 0 else (torch.tensor(0.), torch.tensor(0))

                if best_iou >= iou_threshold and not matched[best_j]:
                    matched[best_j] = True
                    tp_list.append(1)
                else:
                    tp_list.append(0)

        if n_gt == 0:
            class_aps[cls_id] = 0.0
            continue

        tp_arr    = np.array(tp_list)
        score_arr = np.array(score_list)
        order     = score_arr.argsort()[::-1]
        tp_arr    = tp_arr[order]

        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)

        recall    = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)

        class_aps[cls_id] = compute_ap(recall, precision)

    return class_aps


def collate_targets_eval(targets, device):
    prompts, zones, metrics = [], [], []
    for t in targets:
        if t['num_anns'] > 0:
            prompts.append(t['prompts'][0])
            zones.append(t['zones'][0])
            metrics.append(t['metrics'][0])
        else:
            prompts.append("No damage detected.")
            zones.append('unknown')
            metrics.append(torch.zeros(4))
    metrics = torch.stack(metrics).to(device)
    return prompts, zones, metrics


@torch.no_grad()
def run_evaluation(model, loader, device, conf_thresh=0.25, iou_thresh=0.5):
    model.eval()
    all_preds   = []
    all_targets = []

    print("Running inference...")
    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        prompts, zones, metrics = collate_targets_eval(targets, device)

        outputs = model(images, prompts, zones, metrics)

        scores    = outputs['scores']      # [B, Q, C]
        boxes     = outputs['pred_boxes']  # [B, Q, 4]

        for i in range(images.shape[0]):
            # Filter by confidence
            max_scores = scores[i].max(dim=-1).values
            keep       = max_scores >= conf_thresh

            all_preds.append({
                'boxes':  boxes[i][keep].cpu(),
                'scores': scores[i][keep].cpu(),
            })

            all_targets.append({
                'boxes':  targets[i]['boxes'],
                'labels': targets[i]['labels'],
            })

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]")

    # Compute AP at IoU=0.5
    print("\nComputing mAP50...")
    ap50 = evaluate_predictions(
        all_preds, all_targets,
        num_classes=6, iou_threshold=0.5
    )

    # Compute mAP50-95
    print("Computing mAP50-95...")
    ap_all = defaultdict(list)
    for iou_t in np.arange(0.5, 1.0, 0.05):
        aps = evaluate_predictions(
            all_preds, all_targets,
            num_classes=6, iou_threshold=iou_t
        )
        for cls_id, ap in aps.items():
            ap_all[cls_id].append(ap)

    ap5095 = {cls_id: np.mean(vals) for cls_id, vals in ap_all.items()}

    return ap50, ap5095


def print_results(ap50, ap5095):
    print("\n" + "=" * 60)
    print(f"{'Class':<20} {'AP50':>10} {'AP50-95':>10}")
    print("-" * 60)

    map50   = []
    map5095 = []

    for cls_id in range(6):
        name = CLASS_ID_TO_NAME.get(cls_id, str(cls_id))
        a50  = ap50.get(cls_id, 0.0)
        a95  = ap5095.get(cls_id, 0.0)
        map50.append(a50)
        map5095.append(a95)
        print(f"  {name:<18} {a50:>10.3f} {a95:>10.3f}")

    print("-" * 60)
    print(f"  {'mAP (all)':<18} {np.mean(map50):>10.3f} {np.mean(map5095):>10.3f}")
    print("=" * 60)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Split  : {args.split}")
    print(f"Mode   : {args.prompt_mode}")

    # ── Build model ───────────────────────────────────────────
    print("\nBuilding SLIM-Det...")
    model = SLIMDet(
        num_classes = 6,
        hidden_dim  = 256,
        num_queries = 90,
        num_layers  = 4,
        text_model  = 'minilm',
        freeze_text = True,
        use_scf     = True,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating with random weights (sanity check only)")

    # ── Data loader ───────────────────────────────────────────
    json_path  = TEST_JSON  if args.split == 'test' else VAL_JSON
    images_dir = TEST_IMAGES if args.split == 'test' else VAL_IMAGES

    loader = build_val_loader(
        json_path   = json_path,
        images_dir  = images_dir,
        batch_size  = args.batch,
        image_size  = args.imgsz,
        prompt_mode = args.prompt_mode,
        num_workers = args.workers,
    )
    print(f"Eval batches : {len(loader)}")

    # ── Evaluate ──────────────────────────────────────────────
    ap50, ap5095 = run_evaluation(
        model, loader, device,
        conf_thresh = args.conf,
        iou_thresh  = args.iou_thresh,
    )

    print_results(ap50, ap5095)


if __name__ == '__main__':
    main()
