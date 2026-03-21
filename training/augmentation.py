# training/augmentation.py
"""
Copy-paste augmentation for rare damage classes.
Oversamples scratch, paint-peel-off, corrosion, missing-head
to reduce class imbalance before training SLIM-Det.
"""

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Target counts after augmentation ─────────────────────────
TARGET_COUNTS = {
    'crack':          8759,    # leave as is
    'dent':           11716,   # leave as is
    'corrosion':      3500,    # oversample: 2522 → 3500
    'scratch':        3500,    # oversample: 1384 → 3500 (critical)
    'missing-head':   5000,    # oversample: 4327 → 5000
    'paint-peel-off': 3500,    # oversample: 1777 → 3500
}

CURRENT_COUNTS = {
    'crack': 8759, 'dent': 11716, 'corrosion': 2522,
    'scratch': 1384, 'missing-head': 4327, 'paint-peel-off': 1777,
}


def build_patch_bank(json_path, images_dir):
    """
    Extract object patches from JSON annotations.
    Returns {class_name: [patch_entry, ...]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    bank = defaultdict(list)

    for item in data['images']:
        img_path = Path(images_dir) / item['file_name']
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for ann in item.get('annotations', []):
            cat  = ann['category_name']
            bb   = ann['bounding_box_normalized']
            cx, cy, bw, bh = bb['x_center'], bb['y_center'], bb['width'], bb['height']

            x1 = max(0, int((cx - bw/2) * W))
            y1 = max(0, int((cy - bh/2) * H))
            x2 = min(W, int((cx + bw/2) * W))
            y2 = min(H, int((cy + bh/2) * H))

            patch = img[y1:y2, x1:x2]
            if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue

            bank[cat].append({
                'patch':   patch,
                'metrics': ann.get('damage_metrics', {}),
                'desc':    ann.get('description', ''),
                'zone':    ann.get('zone_estimation', 'central'),
                'sev':     ann.get('risk_assessment', {}).get('severity_level', 'low'),
                'box':     (cx, cy, bw, bh),
            })

    print("Patch bank built:")
    for cat, patches in sorted(bank.items()):
        print(f"  {cat:<18}: {len(patches)} patches")
    return bank


def iou_cxcywh(b1, b2):
    """IoU between two (cx,cy,w,h) boxes."""
    def to_xyxy(b):
        return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ax1,ay1,ax2,ay2 = to_xyxy(b1)
    bx1,by1,bx2,by2 = to_xyxy(b2)
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1 = (ax2-ax1)*(ay2-ay1)
    a2 = (bx2-bx1)*(by2-by1)
    return inter/(a1+a2-inter+1e-6)


def paste_patch(image, existing_boxes, patch_entry, n=2):
    """
    Paste n copies of patch onto image in non-overlapping locations.
    Returns (augmented_image, new_annotation_entries)
    """
    H, W  = image.shape[:2]
    new_anns = []

    for _ in range(n):
        patch = patch_entry['patch'].copy()
        ph, pw = patch.shape[:2]

        # Random scale jitter
        scale = random.uniform(0.7, 1.3)
        pw_new = max(8, int(pw * scale))
        ph_new = max(8, int(ph * scale))
        patch  = cv2.resize(patch, (pw_new, ph_new))

        # Random flip
        if random.random() > 0.5:
            patch = cv2.flip(patch, 1)

        # Try to place without heavy overlap
        placed = False
        for _ in range(15):
            x1 = random.randint(0, max(0, W - pw_new))
            y1 = random.randint(0, max(0, H - ph_new))
            x2, y2 = x1+pw_new, y1+ph_new

            cx = (x1+x2)/2/W
            cy = (y1+y2)/2/H
            bw = pw_new/W
            bh = ph_new/H
            new_box = (cx, cy, bw, bh)

            if all(iou_cxcywh(new_box, eb) < 0.2 for eb in existing_boxes):
                alpha = random.uniform(0.8, 1.0)
                roi   = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = cv2.addWeighted(
                    patch, alpha, roi, 1-alpha, 0
                )
                existing_boxes.append(new_box)
                new_anns.append({
                    'category_name': 'augmented',
                    'bounding_box_normalized': {
                        'x_center': cx, 'y_center': cy,
                        'width': bw,   'height': bh,
                    },
                    'damage_metrics':  patch_entry['metrics'],
                    'description':     patch_entry['desc'],
                    'zone_estimation': patch_entry['zone'],
                    'risk_assessment': {'severity_level': patch_entry['sev']},
                })
                placed = True
                break

    return image, new_anns


def verify_bank(bank):
    """Print coverage check — how many pastes each class needs."""
    print("\n── Augmentation coverage check ──")
    print(f"{'Class':<18} {'Current':>8} {'Target':>8} {'Needed':>8} {'Patches':>8} {'OK?':>6}")
    print("-" * 60)
    for cat in TARGET_COUNTS:
        current = CURRENT_COUNTS.get(cat, 0)
        target  = TARGET_COUNTS[cat]
        needed  = max(0, target - current)
        patches = len(bank.get(cat, []))
        ok      = "✅" if patches * 4 >= needed else "⚠️"
        print(f"  {cat:<16} {current:>8} {target:>8} {needed:>8} {patches:>8} {ok:>6}")


if __name__ == '__main__':
    import sys

    JSON_PATH  = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset\Aircraft_train.json'
    IMAGES_DIR = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset\images\train'

    if not os.path.exists(JSON_PATH):
        print(f"JSON not found: {JSON_PATH}")
        sys.exit(0)

    bank = build_patch_bank(JSON_PATH, IMAGES_DIR)
    verify_bank(bank)
    print("\naugmentation.py works correctly")
