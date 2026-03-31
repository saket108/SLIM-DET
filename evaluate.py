"""
SLIM-Det evaluation script.

Computes mAP50, mAP50-95, precision, recall, and F1 on the
validation or test split.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import build_batch_context, build_val_loader
from model.slim_det import SLIMDet
from utils.runtime import (
    coalesce,
    load_yaml_config,
    normalize_class_names,
    require_existing_paths,
    resolve_detection_paths,
    resolve_dataset_paths,
)


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'configs',
    'slim_det.yaml',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SLIM-Det')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--data_format', type=str, default=None, choices=['json', 'detection'])
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--test_json', type=str, default=None)
    parser.add_argument('--val_images', type=str, default=None)
    parser.add_argument('--test_images', type=str, default=None)
    parser.add_argument('--val_labels', type=str, default=None)
    parser.add_argument('--test_labels', type=str, default=None)
    parser.add_argument('--prompt_mode', type=str, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_queries', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--iou_thresh', type=float, default=None)
    parser.add_argument('--nms_iou', type=float, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--text_model', type=str, default=None)
    parser.add_argument('--backbone_name', type=str, default=None)
    parser.add_argument('--no_scf', action='store_true')
    parser.add_argument('--image_only', dest='image_only', action='store_true')
    parser.add_argument('--multimodal', dest='image_only', action='store_false')
    parser.add_argument('--detection_only', action='store_true')
    parser.add_argument('--quality_head', dest='use_quality_head', action='store_true')
    parser.add_argument('--no_quality_head', dest='use_quality_head', action='store_false')
    parser.add_argument('--freeze_text', dest='freeze_text', action='store_true')
    parser.add_argument('--train_text', dest='freeze_text', action='store_false')
    parser.add_argument('--no_pretrained_backbone', action='store_true')
    parser.set_defaults(freeze_text=None, image_only=None, use_quality_head=None)
    return parser.parse_args()


def resolve_args(args):
    config = load_yaml_config(args.config)
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    prompt_cfg = config.get('prompt', {})
    train_cfg = config.get('train', {})
    eval_cfg = config.get('eval', {})
    data_format = coalesce(args.data_format, data_cfg.get('format'), 'json')
    detection_only = bool(args.detection_only or data_format == 'detection')
    class_names = normalize_class_names(data_cfg.get('class_names'))
    num_classes = coalesce(args.num_classes, data_cfg.get('num_classes'), len(class_names) if class_names else None, 6)

    json_path = None
    labels_dir = None
    dataset_root = args.dataset_root if data_format == 'detection' else coalesce(
        args.dataset_root,
        data_cfg.get('dataset_root'),
    )

    if data_format == 'detection':
        val_paths = resolve_detection_paths(
            config_path=args.data_config,
            dataset_root=dataset_root,
            split='val',
            images_dir=args.val_images,
            labels_dir=args.val_labels,
        )
        test_paths = resolve_detection_paths(
            config_path=args.data_config,
            dataset_root=val_paths['dataset_root'],
            split='test',
            images_dir=args.test_images,
            labels_dir=args.test_labels,
        )
        active_paths = test_paths if args.split == 'test' else val_paths
        dataset_root = active_paths['dataset_root']
        split_images = active_paths['images_dir']
        labels_dir = active_paths['labels_dir']
        class_names = coalesce(active_paths['class_names'], class_names)
        num_classes = coalesce(active_paths['num_classes'], num_classes, 6)
    else:
        _, val_json, val_images = resolve_dataset_paths(
            dataset_root=dataset_root,
            data_config=data_cfg,
            split='val',
            json_path=args.val_json,
            images_dir=args.val_images,
        )
        dataset_root, test_json, test_images = resolve_dataset_paths(
            dataset_root=dataset_root,
            data_config=data_cfg,
            split='test',
            json_path=args.test_json,
            images_dir=args.test_images,
        )
        json_path = test_json if args.split == 'test' else val_json
        split_images = test_images if args.split == 'test' else val_images

    resolved = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'data_config': args.data_config,
        'data_format': data_format,
        'detection_only': detection_only,
        'dataset_root': dataset_root,
        'json_path': json_path,
        'images_dir': split_images,
        'labels_dir': labels_dir,
        'prompt_mode': coalesce(args.prompt_mode, 'cat_only' if detection_only else prompt_cfg.get('mode'), 'cat_only' if detection_only else 'full'),
        'batch': coalesce(args.batch, eval_cfg.get('batch_size'), train_cfg.get('batch_size'), 4),
        'imgsz': coalesce(args.imgsz, train_cfg.get('image_size'), data_cfg.get('image_size'), 640),
        'hidden_dim': coalesce(args.hidden_dim, model_cfg.get('hidden_dim'), 256),
        'num_queries': coalesce(args.num_queries, model_cfg.get('detection_num_queries') if detection_only else model_cfg.get('num_queries'), 60 if detection_only else 90),
        'num_layers': coalesce(args.num_layers, model_cfg.get('detection_num_layers') if detection_only else model_cfg.get('num_layers'), 2 if detection_only else 4),
        'num_classes': num_classes,
        'class_names': class_names,
        'conf': coalesce(args.conf, eval_cfg.get('conf_thresh'), 0.25),
        'iou_thresh': coalesce(args.iou_thresh, eval_cfg.get('iou_thresh'), 0.5),
        'nms_iou': coalesce(args.nms_iou, eval_cfg.get('nms_iou'), 0.6),
        'workers': (
            args.workers
            if args.workers is not None
            else coalesce(train_cfg.get('num_workers'), 0)
        ),
        'max_batches': args.max_batches,
        'text_model': coalesce(args.text_model, model_cfg.get('text_model'), 'minilm'),
        'backbone_name': coalesce(
            args.backbone_name,
            model_cfg.get('detection_backbone_name') if detection_only else model_cfg.get('backbone_name'),
            'mobilenet' if detection_only else 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        ),
        'image_only': True if detection_only else coalesce(args.image_only, model_cfg.get('image_only'), False),
        'freeze_text': coalesce(args.freeze_text, model_cfg.get('freeze_text'), True),
        'use_quality_head': coalesce(
            args.use_quality_head,
            model_cfg.get('use_quality_head'),
            False,
        ),
        'use_scf': False if detection_only else (not args.no_scf if args.no_scf else coalesce(model_cfg.get('use_scf'), True)),
        'pretrained_backbone': (
            False if args.no_pretrained_backbone
            else coalesce(model_cfg.get('pretrained_backbone'), True)
        ),
    }

    if resolved['data_format'] == 'detection':
        require_existing_paths(
            data_config=resolved['data_config'],
            images_dir=resolved['images_dir'],
            labels_dir=resolved['labels_dir'],
        )
    else:
        require_existing_paths(json_path=resolved['json_path'], images_dir=resolved['images_dir'])
    if resolved['checkpoint'] is not None:
        resolved['checkpoint'] = resolve_checkpoint_path(
            resolved['checkpoint'],
            config=config,
        )
    return argparse.Namespace(**resolved)


def resolve_checkpoint_path(checkpoint_arg, config):
    """Resolve checkpoint aliases and produce a helpful error if missing."""
    checkpoint_cfg = config.get('checkpoint', {})
    save_dir = checkpoint_cfg.get('save_dir', 'runs/slim_det')

    aliases = {
        'best': os.path.join(save_dir, 'slim_det_best.pt'),
        'last': os.path.join(save_dir, 'slim_det_last.pt'),
        'latest': os.path.join(save_dir, 'slim_det_last.pt'),
    }
    candidate = aliases.get(checkpoint_arg, checkpoint_arg)

    if os.path.exists(candidate):
        return candidate

    checkpoint_files = []
    if os.path.isdir(save_dir):
        checkpoint_files = sorted(
            [
                os.path.join(save_dir, name)
                for name in os.listdir(save_dir)
                if name.endswith(('.pt', '.pth'))
            ]
        )

    if checkpoint_files:
        available = ", ".join(checkpoint_files)
        raise FileNotFoundError(
            f"Checkpoint not found: '{candidate}'. Available checkpoints: {available}"
        )

    raise FileNotFoundError(
        f"Checkpoint not found: '{candidate}'. No checkpoint files were found under "
        f"'{save_dir}'. Train the model first or pass the correct checkpoint path."
    )


def apply_checkpoint_model_config(args, checkpoint):
    """Prefer architecture settings stored in the checkpoint."""
    model_config = checkpoint.get('model_config')
    if not model_config:
        return args

    for key in (
        'hidden_dim',
        'num_queries',
        'num_layers',
        'text_model',
        'backbone_name',
        'prompt_mode',
        'freeze_text',
        'pretrained_backbone',
        'image_only',
        'use_quality_head',
        'use_scf',
        'num_classes',
        'class_names',
        'detection_only',
        'data_format',
    ):
        if key in model_config:
            setattr(args, key, model_config[key])

    return args


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of normalized cxcywh boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=boxes1.dtype,
            device=boxes1.device,
        )

    def to_xyxy(boxes):
        return torch.stack([
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ], dim=1)

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


def nms_cxcywh(boxes, scores, iou_threshold):
    """Simple NMS for normalized cxcywh boxes."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)


def prepare_image_predictions(score_matrix, boxes, conf_thresh=0.25, nms_iou=0.6):
    """Convert raw query outputs into one labeled detection per kept query."""
    confidences, labels = score_matrix.max(dim=-1)
    keep = confidences >= conf_thresh

    boxes = boxes[keep]
    score_matrix = score_matrix[keep]
    confidences = confidences[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return {
            'boxes': boxes.cpu(),
            'scores': score_matrix.cpu(),
            'labels': labels.cpu(),
            'confidences': confidences.cpu(),
        }

    if nms_iou is not None:
        kept_indices = []
        for cls_id in labels.unique(sorted=True):
            cls_mask = labels == cls_id
            cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            cls_keep = nms_cxcywh(
                boxes[cls_mask],
                confidences[cls_mask],
                iou_threshold=nms_iou,
            )
            kept_indices.append(cls_indices[cls_keep])

        if kept_indices:
            keep_idx = torch.cat(kept_indices)
            keep_idx = keep_idx[confidences[keep_idx].argsort(descending=True)]
            boxes = boxes[keep_idx]
            score_matrix = score_matrix[keep_idx]
            confidences = confidences[keep_idx]
            labels = labels[keep_idx]

    return {
        'boxes': boxes.cpu(),
        'scores': score_matrix.cpu(),
        'labels': labels.cpu(),
        'confidences': confidences.cpu(),
    }


def compute_ap(recall, precision):
    """Compute AP from the precision envelope."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_predictions(all_preds, all_targets, num_classes, iou_threshold=0.5):
    """Return per-class AP, precision, recall, and F1."""
    class_metrics = {}

    for cls_id in range(num_classes):
        detections = []
        tp_count = 0
        fp_count = 0
        n_gt = 0

        for preds, targets in zip(all_preds, all_targets):
            gt_boxes = targets['boxes']
            gt_labels = targets['labels']

            gt_mask = gt_labels == cls_id
            gt_boxes_cls = gt_boxes[gt_mask]
            n_gt += gt_boxes_cls.shape[0]

            pred_mask = preds['labels'] == cls_id
            pred_boxes = preds['boxes'][pred_mask]
            pred_scores = preds['confidences'][pred_mask]

            if pred_scores.numel() > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

            matched = torch.zeros(gt_boxes_cls.shape[0], dtype=torch.bool)

            for idx, pred_box in enumerate(pred_boxes):
                score = float(pred_scores[idx])
                is_true_positive = 0

                if gt_boxes_cls.shape[0] > 0:
                    ious = box_iou(pred_box.unsqueeze(0), gt_boxes_cls)[0]
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
            'ap': float(ap_value),
            'precision': float(precision_value),
            'recall': float(recall_value),
            'f1': float(f1_value),
            'tp': int(tp_count),
            'fp': int(fp_count),
            'fn': int(fn_count),
            'num_gt': int(n_gt),
        }

    return class_metrics


def summarize_metrics(class_metrics):
    macro_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
    macro_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
    macro_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])

    total_tp = sum(metrics['tp'] for metrics in class_metrics.values())
    total_fp = sum(metrics['fp'] for metrics in class_metrics.values())
    total_fn = sum(metrics['fn'] for metrics in class_metrics.values())

    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = (
        2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
    }


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    device,
    num_classes,
    conf_thresh=0.25,
    match_iou=0.5,
    nms_iou=0.6,
    max_batches=None,
    verbose=True,
    progress_label=None,
):
    model.eval()
    all_preds = []
    all_targets = []

    iterator = loader
    progress = None
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    if tqdm is not None:
        progress = tqdm(
            loader,
            total=total_batches,
            desc=progress_label or "Eval",
            dynamic_ncols=True,
            leave=False,
            disable=(not verbose),
        )
        iterator = progress
    elif verbose:
        print("Running inference...")

    if verbose and progress is None:
        print("Running inference...")

    for batch_idx, (images, targets, _) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        prompts, zones, metrics = build_batch_context(targets, device)
        outputs = model(images, prompts, zones, metrics)

        scores = outputs['scores']
        boxes = outputs['pred_boxes']

        for i in range(images.shape[0]):
            all_preds.append(
                prepare_image_predictions(
                    scores[i],
                    boxes[i],
                    conf_thresh=conf_thresh,
                    nms_iou=nms_iou,
                )
            )
            all_targets.append({
                'boxes': targets[i]['boxes'],
                'labels': targets[i]['labels'],
            })

        if progress is not None:
            progress.set_postfix({'images': len(all_targets)})
        elif verbose and (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}]")

    if progress is not None:
        progress.close()

    if verbose:
        print("\nComputing mAP50...")
    ap50_metrics = evaluate_predictions(
        all_preds,
        all_targets,
        num_classes=num_classes,
        iou_threshold=0.5,
    )

    if verbose:
        print(f"Computing precision/recall at IoU={match_iou:.2f}...")
    pr_metrics = evaluate_predictions(
        all_preds,
        all_targets,
        num_classes=num_classes,
        iou_threshold=match_iou,
    )

    if verbose:
        print("Computing mAP50-95...")
    ap_all = defaultdict(list)
    for iou_t in np.arange(0.5, 1.0, 0.05):
        threshold_metrics = evaluate_predictions(
            all_preds,
            all_targets,
            num_classes=num_classes,
            iou_threshold=float(iou_t),
        )
        for cls_id, metrics in threshold_metrics.items():
            ap_all[cls_id].append(metrics['ap'])

    ap50 = {cls_id: metrics['ap'] for cls_id, metrics in ap50_metrics.items()}
    ap5095 = {cls_id: float(np.mean(vals)) for cls_id, vals in ap_all.items()}
    summary = summarize_metrics(pr_metrics)
    return ap50, ap5095, pr_metrics, summary


def print_results(ap50, ap5095, pr_metrics, summary, match_iou, class_names=None):
    print("\n" + "=" * 90)
    print(
        f"{'Class':<20} {'AP50':>8} {'AP50-95':>10} "
        f"{'Prec':>8} {'Recall':>8} {'F1':>8}"
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
        if class_names and cls_id < len(class_names):
            name = class_names[cls_id]
        else:
            name = str(cls_id)
        a50 = ap50.get(cls_id, 0.0)
        a95 = ap5095.get(cls_id, 0.0)
        pr = pr_metrics.get(cls_id, {})
        map50.append(a50)
        map5095.append(a95)
        print(
            f"  {name:<18} {a50:>8.3f} {a95:>10.3f} "
            f"{pr.get('precision', 0.0):>8.3f} "
            f"{pr.get('recall', 0.0):>8.3f} "
            f"{pr.get('f1', 0.0):>8.3f}"
        )

    print("-" * 90)
    print(
        f"  {'mAP (macro)':<18} {np.mean(map50):>8.3f} {np.mean(map5095):>10.3f} "
        f"{summary['macro_precision']:>8.3f} "
        f"{summary['macro_recall']:>8.3f} "
        f"{summary['macro_f1']:>8.3f}"
    )
    print(
        f"  {'Micro @ IoU':<18} {'-':>8} {'-':>10} "
        f"{summary['micro_precision']:>8.3f} "
        f"{summary['micro_recall']:>8.3f} "
        f"{summary['micro_f1']:>8.3f}"
    )
    print("-" * 90)
    print(
        f"  Match IoU={match_iou:.2f} | "
        f"TP={summary['total_tp']} FP={summary['total_fp']} FN={summary['total_fn']}"
    )
    print("=" * 90)


def main():
    args = resolve_args(parse_args())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        args = apply_checkpoint_model_config(args, checkpoint)

    print(f"Device : {device}")
    print(f"Split  : {args.split}")
    print(f"Format : {args.data_format}")
    print(f"Mode   : {args.prompt_mode}")
    print(f"Image only : {args.image_only}")
    print(f"Quality head : {args.use_quality_head}")
    print(f"Data   : {args.images_dir}")
    print(f"Hidden dim : {args.hidden_dim}")
    print(f"Classes    : {args.num_classes}")
    print(f"Conf   : {args.conf}")
    print(f"Match IoU : {args.iou_thresh}")
    print(f"NMS IoU   : {args.nms_iou}")

    print("\nBuilding SLIM-Det...")
    model = SLIMDet(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_layers=args.num_layers,
        text_model=args.text_model,
        backbone_name=args.backbone_name,
        prompt_mode=args.prompt_mode,
        freeze_text=args.freeze_text,
        pretrained_backbone=args.pretrained_backbone,
        image_only=args.image_only,
        use_quality_head=args.use_quality_head,
        use_scf=args.use_scf,
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint specified - evaluating random weights only")

    loader = build_val_loader(
        json_path=args.json_path,
        images_dir=args.images_dir,
        batch_size=args.batch,
        image_size=args.imgsz,
        prompt_mode=args.prompt_mode,
        num_workers=args.workers,
        data_format=args.data_format,
        labels_dir=args.labels_dir,
        class_names=args.class_names,
    )
    print(f"Eval batches : {len(loader)}")

    ap50, ap5095, pr_metrics, summary = run_evaluation(
        model,
        loader,
        device,
        num_classes=args.num_classes,
        conf_thresh=args.conf,
        match_iou=args.iou_thresh,
        nms_iou=args.nms_iou,
        max_batches=args.max_batches,
    )
    print_results(
        ap50,
        ap5095,
        pr_metrics,
        summary,
        match_iou=args.iou_thresh,
        class_names=args.class_names,
    )


if __name__ == '__main__':
    main()
