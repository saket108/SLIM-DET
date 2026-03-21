# training/total_loss.py
"""
TotalLoss — SLIM-Det training loss.

Components:
    focal_cls : per-class weighted focal loss (handles 8.5x imbalance)
    l1_box    : L1 box regression
    giou      : GIoU box loss
    severity  : MSE severity regression
    aux       : auxiliary losses on intermediate decoder outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.task_aligned_assigner import TaskAlignedAssigner, giou_loss

# Inverse-frequency alpha — higher = more weight for rare classes
CLASS_ALPHA = torch.tensor([0.18, 0.12, 0.48, 0.65, 0.42, 0.58])


class FocalLoss(nn.Module):
    def __init__(self, alpha=CLASS_ALPHA, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, pred_logits, targets):
        bce   = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        pt    = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        return (self.alpha.to(pred_logits.device) * focal).mean()


class TotalLoss(nn.Module):
    def __init__(self, num_classes=6, w_cls=1.0, w_box=5.0,
                 w_giou=2.0, w_sev=1.0, w_aux=1.0):
        super().__init__()
        self.w_cls=w_cls; self.w_box=w_box; self.w_giou=w_giou
        self.w_sev=w_sev; self.w_aux=w_aux
        self.focal  = FocalLoss(CLASS_ALPHA, gamma=2.0)
        self.assign = TaskAlignedAssigner(topk=13, num_classes=num_classes)

    def _single(self, pred_logits, pred_boxes, pred_sev, target):
        gt_boxes = target['boxes'].to(pred_boxes.device)
        gt_labels= target['labels'].to(pred_boxes.device)
        gt_sevs  = target['severities'].to(pred_boxes.device)

        al, ab, asc = self.assign.assign(
            pred_logits.detach(), pred_boxes.detach(), gt_labels, gt_boxes)
        pos = al >= 0

        loss_cls  = self.focal(pred_logits, asc)
        loss_box  = torch.tensor(0., device=pred_boxes.device)
        loss_giou = torch.tensor(0., device=pred_boxes.device)
        loss_sev  = torch.tensor(0., device=pred_boxes.device)

        if pos.any():
            loss_box  = F.l1_loss(pred_boxes[pos], ab[pos])
            loss_giou = giou_loss(pred_boxes[pos], ab[pos]).mean()
            gi = al[pos].clamp(0, gt_sevs.size(0)-1)
            loss_sev = F.mse_loss(pred_sev[pos].sigmoid(), gt_sevs[gi])

        return loss_cls, loss_box, loss_giou, loss_sev

    def forward(self, outputs, targets):
        pl = outputs['pred_logits']   # [B,Q,C]
        pb = outputs['pred_boxes']    # [B,Q,4]
        ps = outputs['pred_severity'] # [B,Q]
        B  = pl.size(0)

        tc=tb=tg=ts = torch.tensor(0., device=pl.device)
        n = 0
        for i in range(B):
            if targets[i]['num_anns'] == 0: continue
            lc,lb,lg,ls = self._single(pl[i],pb[i],ps[i],targets[i])
            tc+=lc; tb+=lb; tg+=lg; ts+=ls; n+=1
        if n > 0: tc/=n; tb/=n; tg/=n; ts/=n

        # Auxiliary losses
        aux = torch.tensor(0., device=pl.device)
        for ao in outputs.get('aux_outputs', []):
            for i in range(B):
                if targets[i]['num_anns'] == 0: continue
                lc,lb,lg,ls = self._single(
                    ao['pred_logits'][i], ao['pred_boxes'][i],
                    ao['pred_severity'][i], targets[i])
                aux = aux + (self.w_cls*lc+self.w_box*lb+
                             self.w_giou*lg+self.w_sev*ls)/B

        total = (self.w_cls*tc + self.w_box*tb +
                 self.w_giou*tg + self.w_sev*ts + self.w_aux*aux)

        loss_dict = {
            'total': total.item(), 'cls': tc.item(),
            'box': tb.item(),   'giou': tg.item(),
            'sev': ts.item(),   'aux': aux.item(),
        }
        return total, loss_dict


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.slim_det import SLIMDet

    print("Testing TotalLoss...")
    model     = SLIMDet(num_classes=6, freeze_text=True)
    criterion = TotalLoss(num_classes=6)
    model.eval()

    B       = 2
    images  = torch.randn(B, 3, 640, 640)
    prompts = ["Damage type: dent. Location: central.", "Damage type: crack. Location: top_left."]
    zones   = ['central', 'top_left']
    metrics = torch.tensor([[0.05,1.2,0.5,0.1],[0.12,2.1,0.8,0.4]])
    targets = [
        {'boxes':torch.tensor([[0.5,0.5,0.2,0.2]]),'labels':torch.tensor([1]),
         'severities':torch.tensor([0.1]),'num_anns':1},
        {'boxes':torch.tensor([[0.2,0.3,0.1,0.15],[0.7,0.6,0.3,0.25]]),
         'labels':torch.tensor([0,3]),'severities':torch.tensor([0.4,0.6]),'num_anns':2},
    ]

    with torch.no_grad():
        out       = model(images, prompts, zones, metrics)
        loss, ld  = criterion(out, targets)

    print(f"Total loss : {loss.item():.4f}")
    for k,v in ld.items(): print(f"  {k:<8}: {v:.4f}")
    print("total_loss.py works correctly")
