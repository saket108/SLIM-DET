# train.py
"""
SLIM-Det training script.

Usage:
    python train.py                          # default config
    python train.py --prompt_mode cat_only   # ablation
    python train.py --epochs 100 --batch 8   # quick run
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model.slim_det      import SLIMDet
from data.loader         import build_train_loader, build_val_loader
from training.total_loss import TotalLoss


# ── Dataset paths ─────────────────────────────────────────────
DATASET_ROOT = r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset'
TRAIN_JSON   = os.path.join(DATASET_ROOT, 'Aircraft_train.json')
VAL_JSON     = os.path.join(DATASET_ROOT, 'Aircraft_val.json')
TRAIN_IMAGES = os.path.join(DATASET_ROOT, 'images', 'train')
VAL_IMAGES   = os.path.join(DATASET_ROOT, 'images', 'val')


def parse_args():
    p = argparse.ArgumentParser(description='Train SLIM-Det')
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--batch',        type=int,   default=4)
    p.add_argument('--lr',           type=float, default=4e-4)
    p.add_argument('--imgsz',        type=int,   default=640)
    p.add_argument('--num_queries',  type=int,   default=90)
    p.add_argument('--num_layers',   type=int,   default=4)
    p.add_argument('--prompt_mode',  type=str,   default='full',
                   choices=['full', 'no_desc', 'minimal', 'cat_only'])
    p.add_argument('--workers',      type=int,   default=0)
    p.add_argument('--save_dir',     type=str,   default='runs/slim_det')
    p.add_argument('--resume',       type=str,   default=None)
    p.add_argument('--no_scf',       action='store_true')
    p.add_argument('--freeze_text',  action='store_true', default=True)
    return p.parse_args()


def collate_targets(targets, device):
    """
    Extracts batch tensors from list-of-dict targets.
    Returns unified metrics tensor and prompt/zone lists.
    """
    prompts = []
    zones   = []
    metrics = []

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


def train_one_epoch(model, loader, optimizer, loss_fn,
                    scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, (images, targets, _) in enumerate(loader):
        images = images.to(device)

        # Extract prompts, zones, metrics from targets
        prompts, zones, metrics = collate_targets(targets, device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images, prompts, zones, metrics)
            loss, loss_dict = loss_fn(outputs, targets, device)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        # Print progress every 50 batches
        if (i + 1) % 50 == 0 or i == 0:
            avg = total_loss / (i + 1)
            print(
                f"  Ep {epoch:3d} [{i+1:4d}/{n_batches}] "
                f"loss={avg:.4f} "
                f"cls={loss_dict.get('cls',0):.3f} "
                f"box={loss_dict.get('box',0):.3f} "
                f"giou={loss_dict.get('giou',0):.3f} "
                f"sev={loss_dict.get('sev',0):.3f}"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for images, targets, _ in loader:
        images  = images.to(device)
        prompts, zones, metrics = collate_targets(targets, device)

        outputs = model(images, prompts, zones, metrics)
        loss, _ = loss_fn(outputs, targets, device)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_dir, tag='last'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'slim_det_{tag}.pt')
    torch.save({
        'epoch':       epoch,
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'scheduler':   scheduler.state_dict(),
        'val_loss':    val_loss,
    }, path)
    return path


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device       : {device}")
    print(f"Prompt mode  : {args.prompt_mode}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch}")

    # ── Build model ───────────────────────────────────────────
    print("\nBuilding SLIM-Det...")
    model = SLIMDet(
        num_classes  = 6,
        hidden_dim   = 256,
        num_queries  = args.num_queries,
        num_layers   = args.num_layers,
        text_model   = 'minilm',
        prompt_mode  = args.prompt_mode,
        freeze_text  = args.freeze_text,
        use_scf      = not args.no_scf,
    ).to(device)

    pc = model.param_count()
    print(f"  Total params     : {pc['total']:,}")
    print(f"  Trainable params : {pc['trainable']:,}")

    # ── Data loaders ──────────────────────────────────────────
    print("\nBuilding data loaders...")
    train_loader = build_train_loader(
        json_path   = TRAIN_JSON,
        images_dir  = TRAIN_IMAGES,
        batch_size  = args.batch,
        image_size  = args.imgsz,
        prompt_mode = args.prompt_mode,
        num_workers = args.workers,
        balanced    = True,
    )
    val_loader = build_val_loader(
        json_path   = VAL_JSON,
        images_dir  = VAL_IMAGES,
        batch_size  = args.batch,
        image_size  = args.imgsz,
        prompt_mode = args.prompt_mode,
        num_workers = args.workers,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Loss, optimizer, scheduler ────────────────────────────
    loss_fn   = TotalLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = 0.0005,
        betas        = (0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = args.lr * 0.01,
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── Resume from checkpoint ─────────────────────────────────
    start_epoch = 1
    best_val    = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('val_loss', float('inf'))
        print(f"\nResumed from epoch {ckpt['epoch']}")

    # ── Training loop ─────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, epoch, args
        )

        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Save last checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler,
                        epoch, val_loss, args.save_dir, tag='last')

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler,
                            epoch, val_loss, args.save_dir, tag='best')
            print(f"  ✅ New best val loss: {best_val:.4f}")

        # Save periodic checkpoint every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(model, optimizer, scheduler,
                            epoch, val_loss, args.save_dir, tag=f'ep{epoch}')

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
