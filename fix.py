# fix.py
"""
Universal path fixer for SLIM-Det.
Run this if your dataset path changes or you move the project.

Usage:
    python fix.py                                    # interactive
    python fix.py --dataset_root D:\datasets\Aircraft_dataset\content\Aircraft_dataset
"""

import os
import argparse

# Files that contain hardcoded dataset paths
PATH_FILES = [
    'train.py',
    'evaluate.py',
    'data/loader.py',
    'training/augmentation.py',
    'configs/slim_det.yaml',
]

# Current hardcoded root (what to search for)
OLD_ROOT = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset'


def replace_in_file(filepath, old, new):
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old not in content:
        print(f"  SKIP (path not found in file): {filepath}")
        return False
    content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  UPDATED: {filepath}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root', type=str, default=None,
                   help='New dataset root path')
    p.add_argument('--old_root', type=str, default=OLD_ROOT,
                   help='Old dataset root to replace')
    args = p.parse_args()

    if args.dataset_root:
        new_root = args.dataset_root
    else:
        print(f"Current dataset root:")
        print(f"  {args.old_root}")
        print(f"\nEnter new dataset root path (or press Enter to cancel):")
        new_root = input("> ").strip().strip("'\"")
        if not new_root:
            print("Cancelled.")
            return

    if not os.path.exists(new_root):
        print(f"Warning: path does not exist: {new_root}")
        confirm = input("Continue anyway? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

    print(f"\nReplacing:")
    print(f"  OLD: {args.old_root}")
    print(f"  NEW: {new_root}")
    print()

    updated = 0
    for f in PATH_FILES:
        if replace_in_file(f, args.old_root, new_root):
            updated += 1

    print(f"\nDone — updated {updated}/{len(PATH_FILES)} files.")

    # Verify
    print("\nVerifying train.py paths...")
    for line in open('train.py', encoding='utf-8'):
        if 'DATASET_ROOT' in line or 'JSON' in line or 'IMAGES' in line:
            print(f"  {line.rstrip()}")


if __name__ == '__main__':
    main()
