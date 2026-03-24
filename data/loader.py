# data/loader.py
"""
PyTorch Dataset for Aircraft_dataset.
Reads from Aircraft_train.json / Aircraft_val.json / Aircraft_test.json.
Returns image + all structured annotation fields ready for SLIM-Det.
"""

import os
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.prompt_builder import (
    CLASS_NAME_TO_ID,
    build_batch_prompts,
    get_image_level_prompt,
)


# ── Default image transforms ──────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_PROMPT = "No damage detected in this inspection image."
DEFAULT_IMAGE_ZONE = "unknown"
DEFAULT_IMAGE_METRICS = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_ID_TO_NAME = {class_id: name for name, class_id in CLASS_NAME_TO_ID.items()}


def make_transforms(image_size: int = 640, is_train: bool = True):
    """Basic transform pipeline — augmentation handled separately."""
    import torchvision.transforms as T
    transforms = []
    transforms.append(T.Resize((image_size, image_size)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(transforms)


def _empty_target(image_id=''):
    return {
        'boxes': torch.zeros((0, 4), dtype=torch.float32),
        'labels': torch.zeros((0,), dtype=torch.long),
        'metrics': torch.zeros((0, 4), dtype=torch.float32),
        'severities': torch.zeros((0,), dtype=torch.float32),
        'sev_levels': torch.zeros((0,), dtype=torch.long),
        'zones': [],
        'prompts': [],
        'image_prompt': DEFAULT_IMAGE_PROMPT,
        'image_zone': DEFAULT_IMAGE_ZONE,
        'image_metrics': DEFAULT_IMAGE_METRICS.clone(),
        'image_id': image_id,
        'num_anns': 0,
    }


def _list_image_files(images_dir):
    image_paths = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return image_paths


def _label_path_for_image(image_path, images_dir, labels_dir):
    relative_path = os.path.relpath(image_path, images_dir)
    stem, _ = os.path.splitext(relative_path)
    return os.path.join(labels_dir, stem + '.txt')


def _parse_detection_label_file(label_path):
    annotations = []
    if not label_path or not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(float(parts[0]))
                cx, cy, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                continue

            annotations.append({
                'class_id': class_id,
                'box': [
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, width)),
                    max(0.0, min(1.0, height)),
                ],
            })

    return annotations


def _detection_target_from_annotations(annotations, image_id=''):
    if not annotations:
        return _empty_target(image_id)

    boxes = [ann['box'] for ann in annotations]
    labels = [ann['class_id'] for ann in annotations]
    num_anns = len(annotations)

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'metrics': torch.zeros((num_anns, 4), dtype=torch.float32),
        'severities': torch.zeros((num_anns,), dtype=torch.float32),
        'sev_levels': torch.zeros((num_anns,), dtype=torch.long),
        'zones': [],
        'prompts': [],
        'image_prompt': DEFAULT_IMAGE_PROMPT,
        'image_zone': DEFAULT_IMAGE_ZONE,
        'image_metrics': DEFAULT_IMAGE_METRICS.clone(),
        'image_id': image_id,
        'num_anns': num_anns,
    }


# ── Main Dataset ──────────────────────────────────────────────
class AircraftDataset(Dataset):
    """
    Loads Aircraft_dataset from JSON split files.

    Each __getitem__ returns:
        image       : FloatTensor [3, H, W]
        target      : dict with boxes, labels, metrics, prompts, zones
        image_id    : str

    Supports all 4 prompt modes for ablation studies.
    """

    def __init__(
        self,
        json_path:    str,
        images_dir:   str,
        image_size:   int  = 640,
        prompt_mode:  str  = 'full',
        is_train:     bool = True,
        max_anns:     int  = 100,
    ):
        """
        Args:
            json_path   : path to Aircraft_train/val/test.json
            images_dir  : path to images/train (or val/test) folder
            image_size  : resize target (default 640)
            prompt_mode : 'full' | 'no_desc' | 'minimal' | 'cat_only'
            is_train    : enables augmentation if True
            max_anns    : max annotations per image (pad/truncate)
        """
        self.images_dir  = images_dir
        self.image_size  = image_size
        self.prompt_mode = prompt_mode
        self.is_train    = is_train
        self.max_anns    = max_anns
        self.transform   = make_transforms(image_size, is_train)

        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = data['images']

        # Filter out images with missing files
        self.images = [
            img for img in self.images
            if os.path.exists(os.path.join(images_dir, img['file_name']))
        ]
        self.sample_class_ids = [
            [CLASS_NAME_TO_ID.get(ann['category_name'], 0) for ann in item.get('annotations', [])]
            for item in self.images
        ]

        print(f"Loaded {len(self.images)} images from {json_path}")
        self._print_class_stats()

    def _print_class_stats(self):
        """Print class distribution for verification."""
        counts = Counter()
        for item in self.images:
            for ann in item.get('annotations', []):
                counts[ann['category_name']] += 1
        print("  Class distribution:")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<16}: {cnt}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        item     = self.images[idx]
        img_path = os.path.join(self.images_dir, item['file_name'])
        image_id = item['image_id']
        anns     = item.get('annotations', [])

        # ── Load image ────────────────────────────────────────
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        image = self.transform(image)   # [3, H, W]

        # ── Background image (no annotations) ─────────────────
        if not anns:
            return image, self._empty_target(image_id), image_id

        # ── Parse annotations ──────────────────────────────────
        boxes       = []   # [N, 4] normalized cx cy w h
        labels      = []   # [N] int class ids
        metrics     = []   # [N, 4] area_ratio elongation edge_factor severity
        severities  = []   # [N] float raw severity score
        sev_levels  = []   # [N] int 0=low 1=medium 2=high 3=critical
        zones       = []   # [N] str zone names
        prompts     = []   # [N] str text prompts

        SEV_MAP = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

        for ann in anns[:self.max_anns]:
            # Bounding box
            bb = ann['bounding_box_normalized']
            boxes.append([
                bb['x_center'], bb['y_center'],
                bb['width'],    bb['height']
            ])

            # Class label
            labels.append(CLASS_NAME_TO_ID.get(ann['category_name'], 0))

            # Numeric damage metrics [4-D vector]
            m = ann.get('damage_metrics', {})
            metrics.append([
                m.get('area_ratio',         0.0),
                m.get('elongation',         1.0),
                m.get('edge_factor',        0.0),
                m.get('raw_severity_score', 0.0),
            ])

            # Severity
            risk = ann.get('risk_assessment', {})
            sev_str = risk.get('severity_level', 'low')
            severities.append(m.get('raw_severity_score', 0.0))
            sev_levels.append(SEV_MAP.get(sev_str, 0))

            # Zone
            zones.append(ann.get('zone_estimation', 'unknown'))

        # Build text prompts
        prompts = build_batch_prompts(anns[:self.max_anns], mode=self.prompt_mode)
        image_prompt = get_image_level_prompt(
            anns[:self.max_anns],
            mode=self.prompt_mode,
        )
        image_zone = _select_image_zone(zones, severities)
        image_metrics = _aggregate_image_metrics(metrics)

        # ── Convert to tensors ─────────────────────────────────
        target = {
            'boxes':      torch.tensor(boxes,      dtype=torch.float32),
            'labels':     torch.tensor(labels,     dtype=torch.long),
            'metrics':    torch.tensor(metrics,    dtype=torch.float32),
            'severities': torch.tensor(severities, dtype=torch.float32),
            'sev_levels': torch.tensor(sev_levels, dtype=torch.long),
            'zones':      zones,      # list of strings
            'prompts':    prompts,    # list of strings
            'image_prompt': image_prompt,
            'image_zone':   image_zone,
            'image_metrics': torch.tensor(image_metrics, dtype=torch.float32),
            'image_id':   image_id,
            'num_anns':   len(anns[:self.max_anns]),
        }

        return image, target, image_id

    def _empty_target(self, image_id=''):
        """Return empty target for background images."""
        return _empty_target(image_id)


class StandardDetectionDataset(Dataset):
    """Detection-only dataset using image folders and txt label files."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 640,
        class_names: list = None,
        is_train: bool = True,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.is_train = is_train
        self.transform = make_transforms(image_size, is_train)
        self.class_names = class_names or []

        image_paths = _list_image_files(images_dir)
        self.records = []
        self.sample_class_ids = []

        for image_path in image_paths:
            image_id = os.path.relpath(image_path, images_dir)
            label_path = _label_path_for_image(image_path, images_dir, labels_dir)
            annotations = _parse_detection_label_file(label_path)
            self.records.append({
                'image_path': image_path,
                'image_id': image_id,
                'annotations': annotations,
            })
            self.sample_class_ids.append([ann['class_id'] for ann in annotations])

        print(f"Loaded {len(self.records)} images from {images_dir}")
        self._print_class_stats()

    def _class_name(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _print_class_stats(self):
        counts = Counter()
        for annotations in self.sample_class_ids:
            for class_id in annotations:
                counts[self._class_name(class_id)] += 1
        print("  Class distribution:")
        for cls, cnt in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {cls:<16}: {cnt}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record['image_path']).convert('RGB')
        image = self.transform(image)
        target = _detection_target_from_annotations(
            record['annotations'],
            image_id=record['image_id'],
        )
        return image, target, record['image_id']


# ── Collate function ──────────────────────────────────────────
def collate_fn(batch):
    """
    Custom collate for variable-length annotations.
    Images are stacked, targets kept as list of dicts.
    """
    images     = []
    targets    = []
    image_ids  = []

    for image, target, image_id in batch:
        images.append(image)
        targets.append(target)
        image_ids.append(image_id)

    images = torch.stack(images, dim=0)   # [B, 3, H, W]
    return images, targets, image_ids


def _select_image_zone(zones, severities):
    if not zones:
        return DEFAULT_IMAGE_ZONE
    if severities:
        best_idx = max(range(len(severities)), key=severities.__getitem__)
        return zones[best_idx]
    return Counter(zones).most_common(1)[0][0]


def _aggregate_image_metrics(metrics):
    if not metrics:
        return DEFAULT_IMAGE_METRICS.tolist()
    metrics_arr = np.asarray(metrics, dtype=np.float32)
    summary = metrics_arr.mean(axis=0)
    summary[3] = metrics_arr[:, 3].max()
    return summary.tolist()


def summarize_target_context(target):
    """Return one prompt, one zone, and one metric vector for an image target."""
    if target.get('num_anns', 0) == 0:
        return (
            DEFAULT_IMAGE_PROMPT,
            DEFAULT_IMAGE_ZONE,
            DEFAULT_IMAGE_METRICS.clone(),
        )

    prompt = target.get('image_prompt') or DEFAULT_IMAGE_PROMPT
    zone = target.get('image_zone') or DEFAULT_IMAGE_ZONE

    metrics = target.get('image_metrics')
    if metrics is None or torch.as_tensor(metrics).numel() == 0:
        metrics = _aggregate_image_metrics(target.get('metrics', []))

    metrics_tensor = torch.as_tensor(metrics, dtype=torch.float32)
    return prompt, zone, metrics_tensor


def build_batch_context(targets, device):
    """Vectorize image-level conditioning inputs for a batch of targets."""
    prompts = []
    zones = []
    metrics = []

    for target in targets:
        prompt, zone, metric = summarize_target_context(target)
        prompts.append(prompt)
        zones.append(zone)
        metrics.append(metric)

    return prompts, zones, torch.stack(metrics).to(device)


# ── Class-balanced sampler ────────────────────────────────────
class ClassBalancedSampler(torch.utils.data.Sampler):
    """Oversample images containing rare classes using dataset-driven weights."""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int = None,
        rarity_power: float = 0.5,
        background_weight: float = None,
    ):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.rarity_power = rarity_power

        sample_class_ids = getattr(dataset, 'sample_class_ids', None)
        if sample_class_ids is None:
            raise ValueError("Dataset does not expose sample_class_ids for balanced sampling.")

        image_class_counts = Counter()
        for class_ids in sample_class_ids:
            image_class_counts.update(set(class_ids))

        if not image_class_counts:
            self.class_weights = {}
            self.background_weight = 1.0
            self.weights = torch.ones(len(sample_class_ids), dtype=torch.float32)
            return

        raw_weights = {
            class_id: 1.0 / float(max(count, 1)) ** rarity_power
            for class_id, count in image_class_counts.items()
        }
        max_weight = max(raw_weights.values())
        self.class_weights = {
            class_id: weight / max_weight
            for class_id, weight in raw_weights.items()
        }

        min_class_weight = min(self.class_weights.values())
        self.background_weight = (
            background_weight
            if background_weight is not None
            else max(min_class_weight * 0.5, 0.05)
        )

        sample_weights = []
        for class_ids in sample_class_ids:
            unique_ids = sorted(set(class_ids))
            if not unique_ids:
                sample_weights.append(self.background_weight)
                continue

            rarest_class_weight = max(
                self.class_weights.get(class_id, min_class_weight)
                for class_id in unique_ids
            )
            diversity_boost = 1.0 + 0.05 * max(len(unique_ids) - 1, 0)
            sample_weights.append(rarest_class_weight * diversity_boost)

        self.weights = torch.tensor(sample_weights, dtype=torch.float32)
        self._print_sampler_stats(image_class_counts)

    def _class_name(self, class_id):
        if hasattr(self.dataset, '_class_name'):
            return self.dataset._class_name(class_id)

        class_names = getattr(self.dataset, 'class_names', None) or []
        if 0 <= class_id < len(class_names):
            return class_names[class_id]

        return CLASS_ID_TO_NAME.get(class_id, f'class_{class_id}')

    def _print_sampler_stats(self, image_class_counts):
        print("  Balanced sampler:")
        for class_id, count in sorted(image_class_counts.items(), key=lambda item: (item[1], item[0])):
            print(
                f"    {self._class_name(class_id):<16}: "
                f"img_freq={count:<5d} weight={self.class_weights[class_id]:.3f}"
            )

    def __iter__(self):
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


# ── DataLoader builders ───────────────────────────────────────
def build_train_loader(
    json_path:   str = None,
    images_dir:  str = None,
    batch_size:  int = 16,
    image_size:  int = 640,
    prompt_mode: str = 'full',
    num_workers: int = 4,
    balanced:    bool = True,
    data_format: str = 'json',
    labels_dir:  str = None,
    class_names: list = None,
) -> DataLoader:
    if data_format == 'detection':
        dataset = StandardDetectionDataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            image_size=image_size,
            class_names=class_names,
            is_train=True,
        )
    else:
        dataset = AircraftDataset(
            json_path   = json_path,
            images_dir  = images_dir,
            image_size  = image_size,
            prompt_mode = prompt_mode,
            is_train    = True,
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No training images were found for data_format='{data_format}' "
            f"and images_dir='{images_dir}'."
        )

    sampler = ClassBalancedSampler(dataset) if balanced else None

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = len(dataset) > batch_size,
    )


def build_val_loader(
    json_path:   str = None,
    images_dir:  str = None,
    batch_size:  int = 16,
    image_size:  int = 640,
    prompt_mode: str = 'full',
    num_workers: int = 4,
    data_format: str = 'json',
    labels_dir:  str = None,
    class_names: list = None,
) -> DataLoader:
    if data_format == 'detection':
        dataset = StandardDetectionDataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            image_size=image_size,
            class_names=class_names,
            is_train=False,
        )
    else:
        dataset = AircraftDataset(
            json_path   = json_path,
            images_dir  = images_dir,
            image_size  = image_size,
            prompt_mode = prompt_mode,
            is_train    = False,
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No validation images were found for data_format='{data_format}' "
            f"and images_dir='{images_dir}'."
        )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = torch.cuda.is_available(),
    )


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    JSON_PATH  = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset\Aircraft_train.json'
    IMAGES_DIR = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset\images\train'

    if not os.path.exists(JSON_PATH):
        print(f"JSON not found: {JSON_PATH}")
        print("Update JSON_PATH and IMAGES_DIR at the bottom of loader.py")
        sys.exit(0)

    print("\n── Building train loader ──")
    loader = build_train_loader(
        json_path   = JSON_PATH,
        images_dir  = IMAGES_DIR,
        batch_size  = 4,
        prompt_mode = 'full',
        num_workers = 0,
        balanced    = True,
    )

    print(f"\nDataset size : {len(loader.dataset)}")
    print(f"Batches      : {len(loader)}")

    print("\n── Fetching first batch ──")
    images, targets, image_ids = next(iter(loader))

    print(f"images shape : {images.shape}")
    print(f"batch size   : {len(targets)}")
    print(f"\nFirst item:")
    t = targets[0]
    print(f"  boxes      : {t['boxes'].shape}  {t['boxes'][:2]}")
    print(f"  labels     : {t['labels']}")
    print(f"  metrics    : {t['metrics'].shape}")
    print(f"  severities : {t['severities']}")
    print(f"  zones      : {t['zones']}")
    print(f"  num_anns   : {t['num_anns']}")
    print(f"\nFirst prompt preview:")
    print(f"  {t['prompts'][0][:120]}...")
    print("\nloader.py works correctly")
