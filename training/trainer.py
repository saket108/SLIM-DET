# training/trainer.py
"""
Trainer utility class for SLIM-Det.
Wraps training loop, validation, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.loader import build_batch_context
from utils.metrics import LossTracker, MetricLogger


class Trainer:
    """
    SLIM-Det trainer.

    Args:
        model       : SLIMDet model
        train_loader: training DataLoader
        val_loader  : validation DataLoader
        loss_fn     : TotalLoss instance
        config      : dict of training hyperparameters
        save_dir    : directory to save checkpoints
        device      : torch device
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        config,
        save_dir = 'runs/slim_det',
        device   = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.config       = config
        self.save_dir     = save_dir
        self.device       = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        os.makedirs(save_dir, exist_ok=True)

        # Optimizer — only trainable params
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = config.get('lr', 4e-4),
            weight_decay = config.get('weight_decay', 5e-4),
            betas        = config.get('betas', (0.9, 0.999)),
        )

        epochs = config.get('epochs', 300)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max   = epochs,
            eta_min = config.get('lr', 4e-4) * 0.01,
        )

        self.scaler = amp.GradScaler(
            self.device.type,
            enabled=(self.device.type == 'cuda'),
        )
        self.logger = MetricLogger()

        self.best_val = float('inf')
        self.start_epoch = 1

    def _collate_targets(self, targets):
        """Extract prompts, zones, metrics from batch targets."""
        return build_batch_context(targets, self.device)

    def train_epoch(self, epoch):
        self.model.train()
        tracker = LossTracker()
        n       = len(self.train_loader)

        for i, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            prompts, zones, metrics = self._collate_targets(targets)

            self.optimizer.zero_grad()

            with amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == 'cuda'),
            ):
                outputs = self.model(images, prompts, zones, metrics)
                loss, loss_dict = self.loss_fn(
                    outputs, targets, self.device, epoch=epoch
                )

            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            tracker.update({**loss_dict, 'total': loss.item()})

            if (i + 1) % 100 == 0:
                print(
                    f"  Ep {epoch} [{i+1}/{n}] "
                    f"loss={tracker.meters['total'].avg:.4f} "
                    f"cls={tracker.meters.get('cls', type('',(),{'avg':0})).avg:.3f}"
                )

        return tracker.summary()

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        tracker = LossTracker()

        for images, targets, _ in self.val_loader:
            images = images.to(self.device)
            prompts, zones, metrics = self._collate_targets(targets)
            outputs = self.model(images, prompts, zones, metrics)
            loss, loss_dict = self.loss_fn(
                outputs, targets, self.device, epoch=epoch
            )
            tracker.update({**loss_dict, 'total': loss.item()})

        return tracker.summary()

    def save(self, epoch, val_loss, tag='last'):
        path = os.path.join(self.save_dir, f'slim_det_{tag}.pt')
        torch.save({
            'epoch':     epoch,
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss':  val_loss,
        }, path)
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val    = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']}")

    def fit(self, epochs=None):
        epochs = epochs or self.config.get('epochs', 300)
        print(f"Training {self.start_epoch} → {epochs} on {self.device}")

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()

            train_stats = self.train_epoch(epoch)
            val_stats   = self.val_epoch(epoch)
            self.scheduler.step()

            val_loss = val_stats.get('total', 0.0)
            lr       = self.scheduler.get_last_lr()[0]
            elapsed  = time.time() - t0

            print(
                f"Ep {epoch:3d}/{epochs} | "
                f"train={train_stats.get('total',0):.4f} | "
                f"val={val_loss:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

            self.logger.log(
                epoch,
                train_loss = train_stats.get('total', 0),
                val_loss   = val_loss,
                lr         = lr,
            )

            # Save checkpoints
            self.save(epoch, val_loss, tag='last')

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save(epoch, val_loss, tag='best')
                print(f"  ✅ Best val loss: {self.best_val:.4f}")

            if epoch % 50 == 0:
                self.save(epoch, val_loss, tag=f'ep{epoch}')

        best_ep, best_val = self.logger.best('val_loss', mode='min')
        print(f"\nDone. Best val loss {best_val:.4f} at epoch {best_ep}")
        self.logger.print_summary()
