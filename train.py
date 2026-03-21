"""
SLIM-Det training script.

Usage:
    python train.py
    python train.py --config configs/slim_det.yaml
    python train.py --dataset_root C:\path\to\Aircraft_dataset
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import build_batch_context, build_train_loader, build_val_loader
from evaluate import print_results as print_eval_results
from evaluate import run_evaluation
from model.slim_det import SLIMDet
from training.total_loss import TotalLoss
from utils.runtime import (
    coalesce,
    load_yaml_config,
    require_existing_paths,
    resolve_dataset_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SLIM-Det')
    parser.add_argument('--config', type=str, default='configs/slim_det.yaml')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--train_json', type=str, default=None)
    parser.add_argument('--val_json', type=str, default=None)
    parser.add_argument('--train_images', type=str, default=None)
    parser.add_argument('--val_images', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_queries', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_every_epochs', type=int, default=None)
    parser.add_argument('--eval_max_batches', type=int, default=None)
    parser.add_argument('--eval_conf', type=float, default=None)
    parser.add_argument('--eval_iou', type=float, default=None)
    parser.add_argument('--eval_nms_iou', type=float, default=None)

    parser.add_argument(
        '--prompt_mode',
        type=str,
        default=None,
        choices=['full', 'no_desc', 'minimal', 'cat_only'],
    )
    parser.add_argument('--text_model', type=str, default=None)
    parser.add_argument('--backbone_name', type=str, default=None)
    parser.add_argument('--no_scf', action='store_true')
    parser.add_argument('--image_only', dest='image_only', action='store_true')
    parser.add_argument('--multimodal', dest='image_only', action='store_false')
    parser.add_argument('--freeze_text', dest='freeze_text', action='store_true')
    parser.add_argument('--train_text', dest='freeze_text', action='store_false')
    parser.add_argument('--save_every_batches', type=int, default=None)
    parser.add_argument(
        '--no_pretrained_backbone',
        action='store_true',
        help='Disable pretrained timm weights and use random-init/fallback backbone.',
    )
    parser.set_defaults(freeze_text=None, image_only=None)
    return parser.parse_args()


def resolve_args(args):
    config = load_yaml_config(args.config)
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    prompt_cfg = config.get('prompt', {})
    train_cfg = config.get('train', {})
    optimizer_cfg = config.get('optimizer', {})
    checkpoint_cfg = config.get('checkpoint', {})
    eval_cfg = config.get('eval', {})

    dataset_root, train_json, train_images = resolve_dataset_paths(
        dataset_root=args.dataset_root,
        data_config=data_cfg,
        split='train',
        json_path=args.train_json,
        images_dir=args.train_images,
    )
    _, val_json, val_images = resolve_dataset_paths(
        dataset_root=dataset_root,
        data_config=data_cfg,
        split='val',
        json_path=args.val_json,
        images_dir=args.val_images,
    )

    resolved = {
        'config': args.config,
        'dataset_root': dataset_root,
        'train_json': train_json,
        'val_json': val_json,
        'train_images': train_images,
        'val_images': val_images,
        'epochs': coalesce(args.epochs, train_cfg.get('epochs'), 300),
        'batch': coalesce(args.batch, train_cfg.get('batch_size'), 4),
        'lr': coalesce(args.lr, optimizer_cfg.get('lr'), 4e-4),
        'imgsz': coalesce(args.imgsz, train_cfg.get('image_size'), data_cfg.get('image_size'), 640),
        'hidden_dim': coalesce(args.hidden_dim, model_cfg.get('hidden_dim'), 256),
        'num_queries': coalesce(args.num_queries, model_cfg.get('num_queries'), 90),
        'num_layers': coalesce(args.num_layers, model_cfg.get('num_layers'), 4),
        'prompt_mode': coalesce(args.prompt_mode, prompt_cfg.get('mode'), 'full'),
        'workers': (
            args.workers
            if args.workers is not None
            else (0 if os.name == 'nt' else coalesce(train_cfg.get('num_workers'), 0))
        ),
        'save_dir': coalesce(args.save_dir, checkpoint_cfg.get('save_dir'), 'runs/slim_det'),
        'resume': args.resume,
        'text_model': coalesce(args.text_model, model_cfg.get('text_model'), 'minilm'),
        'backbone_name': coalesce(
            args.backbone_name,
            model_cfg.get('backbone_name'),
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
        ),
        'image_only': coalesce(args.image_only, model_cfg.get('image_only'), False),
        'freeze_text': coalesce(args.freeze_text, model_cfg.get('freeze_text'), True),
        'use_scf': not args.no_scf if args.no_scf else coalesce(model_cfg.get('use_scf'), True),
        'pretrained_backbone': (
            False if args.no_pretrained_backbone
            else coalesce(model_cfg.get('pretrained_backbone'), True)
        ),
        'save_every_batches': coalesce(
            args.save_every_batches,
            checkpoint_cfg.get('save_period_batches'),
            0,
        ),
        'eval_every_epochs': coalesce(
            args.eval_every_epochs,
            eval_cfg.get('during_train_every_epochs'),
            1,
        ),
        'eval_max_batches': coalesce(
            args.eval_max_batches,
            eval_cfg.get('during_train_max_batches'),
            None,
        ),
        'eval_conf': coalesce(args.eval_conf, eval_cfg.get('conf_thresh'), 0.25),
        'eval_iou': coalesce(args.eval_iou, eval_cfg.get('iou_thresh'), 0.5),
        'eval_nms_iou': coalesce(args.eval_nms_iou, eval_cfg.get('nms_iou'), 0.6),
    }

    require_existing_paths(
        train_json=resolved['train_json'],
        val_json=resolved['val_json'],
        train_images=resolved['train_images'],
        val_images=resolved['val_images'],
    )
    if resolved['resume'] is not None:
        require_existing_paths(resume=resolved['resume'])
    return argparse.Namespace(**resolved)


def build_model_config(args):
    return {
        'num_classes': 6,
        'hidden_dim': args.hidden_dim,
        'num_queries': args.num_queries,
        'num_layers': args.num_layers,
        'text_model': args.text_model,
        'backbone_name': args.backbone_name,
        'prompt_mode': args.prompt_mode,
        'freeze_text': args.freeze_text,
        'pretrained_backbone': args.pretrained_backbone,
        'image_only': args.image_only,
        'use_scf': args.use_scf,
    }


def append_csv_row(path, fieldnames, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    epoch,
    total_epochs=None,
    save_every_batches=0,
    checkpoint_callback=None,
):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    iterator = loader
    progress = None

    if tqdm is not None:
        progress = tqdm(
            loader,
            total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs or '?'}",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        iterator = progress

    for i, (images, targets, _) in enumerate(iterator):
        images = images.to(device)
        prompts, zones, metrics = build_batch_context(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images, prompts, zones, metrics)
            loss, loss_dict = loss_fn(outputs, targets, device, epoch=epoch)

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

        avg = total_loss / (i + 1)
        if progress is not None:
            progress.set_postfix({
                'loss': f"{avg:.4f}",
                'cls': f"{loss_dict.get('cls', 0):.3f}",
                'box': f"{loss_dict.get('box', 0):.3f}",
                'giou': f"{loss_dict.get('giou', 0):.3f}",
                'sev': f"{loss_dict.get('sev', 0):.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            })
        elif (i + 1) % 50 == 0 or i == 0:
            print(
                f"  Ep {epoch:3d} [{i+1:4d}/{n_batches}] "
                f"loss={avg:.4f} "
                f"cls={loss_dict.get('cls', 0):.3f} "
                f"box={loss_dict.get('box', 0):.3f} "
                f"giou={loss_dict.get('giou', 0):.3f} "
                f"sev={loss_dict.get('sev', 0):.3f}"
            )

        if (
            checkpoint_callback is not None
            and save_every_batches
            and (i + 1) % save_every_batches == 0
            and (i + 1) < n_batches
        ):
            checkpoint_callback(
                epoch=epoch,
                batch_in_epoch=i + 1,
                total_batches=n_batches,
                train_loss=total_loss / (i + 1),
            )

    if progress is not None:
        progress.close()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)
    iterator = loader
    progress = None

    if tqdm is not None:
        progress = tqdm(
            loader,
            total=n_batches,
            desc=f"Val {epoch}",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        iterator = progress

    for batch_idx, (images, targets, _) in enumerate(iterator):
        images = images.to(device)
        prompts, zones, metrics = build_batch_context(targets, device)

        outputs = model(images, prompts, zones, metrics)
        loss, _ = loss_fn(outputs, targets, device, epoch=epoch)
        total_loss += loss.item()

        if progress is not None:
            progress.set_postfix({'loss': f"{total_loss / (batch_idx + 1):.4f}"})

    if progress is not None:
        progress.close()

    return total_loss / max(len(loader), 1)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    save_dir,
    tag='last',
    model_config=None,
    train_loss=None,
    batch_in_epoch=None,
    total_batches=None,
    global_step=None,
    is_mid_epoch=False,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'slim_det_{tag}.pt')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'batch_in_epoch': batch_in_epoch,
        'total_batches': total_batches,
        'global_step': global_step,
        'is_mid_epoch': is_mid_epoch,
        'model_config': model_config,
    }, path)
    return path


def mean_metric(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    args = resolve_args(parse_args())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device       : {device}")
    print(f"Prompt mode  : {args.prompt_mode}")
    print(f"Image only   : {args.image_only}")
    print(f"Epochs       : {args.epochs}")
    print(f"Hidden dim   : {args.hidden_dim}")
    print(f"Batch size   : {args.batch}")
    print(f"Dataset root : {args.dataset_root}")

    print("\nBuilding SLIM-Det...")
    model_config = build_model_config(args)
    model = SLIMDet(**model_config).to(device)

    pc = model.param_count()
    print(f"  Total params     : {pc['total']:,}")
    print(f"  Trainable params : {pc['trainable']:,}")

    print("\nBuilding data loaders...")
    train_loader = build_train_loader(
        json_path=args.train_json,
        images_dir=args.train_images,
        batch_size=args.batch,
        image_size=args.imgsz,
        prompt_mode=args.prompt_mode,
        num_workers=args.workers,
        balanced=True,
    )
    val_loader = build_val_loader(
        json_path=args.val_json,
        images_dir=args.val_images,
        batch_size=args.batch,
        image_size=args.imgsz,
        prompt_mode=args.prompt_mode,
        num_workers=args.workers,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    loss_fn = TotalLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.0005,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler = amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    start_epoch = 1
    best_val = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        if ckpt.get('is_mid_epoch'):
            start_epoch = ckpt['epoch']
            batch_in_epoch = ckpt.get('batch_in_epoch')
            print(
                f"\nResumed from mid-epoch checkpoint at epoch {ckpt['epoch']}, "
                f"step {batch_in_epoch}. The epoch will restart from the beginning."
            )
        else:
            print(f"\nResumed from epoch {ckpt['epoch']}")
        best_val = ckpt.get('val_loss', float('inf'))

    print(f"\nStarting training for {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    history_path = os.path.join(args.save_dir, 'train_history.csv')
    history_fields = [
        'epoch',
        'train_loss',
        'val_loss',
        'lr',
        'elapsed_sec',
        'best_val',
        'map50',
        'map5095',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'micro_precision',
        'micro_recall',
        'micro_f1',
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        def checkpoint_callback(epoch, batch_in_epoch, total_batches, train_loss):
            nonlocal global_step
            global_step = (epoch - 1) * total_batches + batch_in_epoch
            path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                args.save_dir,
                tag='last',
                model_config=model_config,
                train_loss=train_loss,
                batch_in_epoch=batch_in_epoch,
                total_batches=total_batches,
                global_step=global_step,
                is_mid_epoch=True,
            )
            print(
                f"  Saved mid-epoch checkpoint: {path} "
                f"(epoch {epoch}, step {batch_in_epoch}/{total_batches})"
            )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            epoch,
            total_epochs=args.epochs,
            save_every_batches=args.save_every_batches,
            checkpoint_callback=checkpoint_callback,
        )
        val_loss = validate(model, val_loader, loss_fn, device, epoch)
        map50 = ''
        map5095 = ''
        summary = None

        if args.eval_every_epochs and epoch % args.eval_every_epochs == 0:
            ap50, ap5095, pr_metrics, summary = run_evaluation(
                model,
                val_loader,
                device,
                conf_thresh=args.eval_conf,
                match_iou=args.eval_iou,
                nms_iou=args.eval_nms_iou,
                max_batches=args.eval_max_batches,
            )
            map50 = mean_metric(list(ap50.values()))
            map5095 = mean_metric(list(ap5095.values()))
            print_eval_results(ap50, ap5095, pr_metrics, summary, args.eval_iou)

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"mAP50={map50 if map50 != '' else 'n/a'} | "
            f"mAP50-95={map5095 if map5095 != '' else 'n/a'} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s"
        )
        append_csv_row(
            history_path,
            history_fields,
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': scheduler.get_last_lr()[0],
                'elapsed_sec': elapsed,
                'best_val': best_val,
                'map50': map50,
                'map5095': map5095,
                'macro_precision': '' if summary is None else summary['macro_precision'],
                'macro_recall': '' if summary is None else summary['macro_recall'],
                'macro_f1': '' if summary is None else summary['macro_f1'],
                'micro_precision': '' if summary is None else summary['micro_precision'],
                'micro_recall': '' if summary is None else summary['micro_recall'],
                'micro_f1': '' if summary is None else summary['micro_f1'],
            },
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            args.save_dir,
            tag='last',
            model_config=model_config,
            train_loss=train_loss,
            batch_in_epoch=len(train_loader),
            total_batches=len(train_loader),
            global_step=epoch * len(train_loader),
            is_mid_epoch=False,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                args.save_dir,
                tag='best',
                model_config=model_config,
                train_loss=train_loss,
                batch_in_epoch=len(train_loader),
                total_batches=len(train_loader),
                global_step=epoch * len(train_loader),
                is_mid_epoch=False,
            )
            print(f"  New best val loss: {best_val:.4f}")

        if epoch % 50 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                args.save_dir,
                tag=f'ep{epoch}',
                model_config=model_config,
                train_loss=train_loss,
                batch_in_epoch=len(train_loader),
                total_batches=len(train_loader),
                global_step=epoch * len(train_loader),
                is_mid_epoch=False,
            )

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
