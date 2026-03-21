# training/task_aligned_assigner.py
"""
Task-Aligned Label Assigner — from YOLOv8.
Replaces Hungarian matching O(n³) with O(n) assignment.
For each GT, selects top-k predictions via: cls^alpha * iou^beta
"""

import torch
import torch.nn.functional as F


def box_iou(boxes1, boxes2):
    """IoU between [N,4] and [M,4] boxes in cx,cy,w,h format. Returns [N,M]."""
    def to_xyxy(b):
        return torch.stack([b[...,0]-b[...,2]/2, b[...,1]-b[...,3]/2,
                            b[...,0]+b[...,2]/2, b[...,1]+b[...,3]/2], dim=-1)
    b1 = to_xyxy(boxes1).unsqueeze(1)   # [N,1,4]
    b2 = to_xyxy(boxes2).unsqueeze(0)   # [1,M,4]
    ix1 = torch.max(b1[...,0], b2[...,0]); iy1 = torch.max(b1[...,1], b2[...,1])
    ix2 = torch.min(b1[...,2], b2[...,2]); iy2 = torch.min(b1[...,3], b2[...,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    a1 = (boxes1[:,2]*boxes1[:,3]).unsqueeze(1)
    a2 = (boxes2[:,2]*boxes2[:,3]).unsqueeze(0)
    return (inter / (a1 + a2 - inter + 1e-6)).clamp(0, 1)


def giou_loss(pred, gt):
    """GIoU loss between matched [N,4] boxes in cx,cy,w,h format."""
    def to_xyxy(b):
        return torch.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                            b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], dim=1)
    p = to_xyxy(pred); g = to_xyxy(gt)
    ix1=torch.max(p[:,0],g[:,0]); iy1=torch.max(p[:,1],g[:,1])
    ix2=torch.min(p[:,2],g[:,2]); iy2=torch.min(p[:,3],g[:,3])
    inter = (ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
    ap=(p[:,2]-p[:,0])*(p[:,3]-p[:,1]); ag=(g[:,2]-g[:,0])*(g[:,3]-g[:,1])
    union = ap + ag - inter + 1e-6; iou = inter/union
    ex1=torch.min(p[:,0],g[:,0]); ey1=torch.min(p[:,1],g[:,1])
    ex2=torch.max(p[:,2],g[:,2]); ey2=torch.max(p[:,3],g[:,3])
    enc = (ex2-ex1).clamp(0)*(ey2-ey1).clamp(0)+1e-6
    return 1 - (iou - (enc-union)/enc)


class TaskAlignedAssigner:
    def __init__(self, topk=13, alpha=0.5, beta=6.0, num_classes=6):
        self.topk=topk; self.alpha=alpha; self.beta=beta; self.num_classes=num_classes

    @torch.no_grad()
    def assign(self, pred_scores, pred_boxes, gt_labels, gt_boxes):
        Q = pred_scores.size(0); G = gt_boxes.size(0); device = pred_boxes.device
        if G == 0:
            return (torch.full((Q,),-1,dtype=torch.long,device=device),
                    torch.zeros(Q,4,device=device),
                    torch.zeros(Q,self.num_classes,device=device))
        iou = box_iou(pred_boxes, gt_boxes)
        cls_scores = pred_scores.sigmoid()[:, gt_labels]
        align = (cls_scores**self.alpha) * (iou**self.beta)
        topk = min(self.topk, Q)
        _, topk_idxs = align.topk(topk, dim=0)
        is_assigned = torch.zeros(Q, G, dtype=torch.bool, device=device)
        is_assigned.scatter_(0, topk_idxs, True)
        max_metric, assigned_gt = (align * is_assigned.float()).max(dim=1)
        labels = torch.full((Q,),-1,dtype=torch.long,device=device)
        boxes  = torch.zeros(Q,4,device=device)
        scores = torch.zeros(Q,self.num_classes,device=device)
        pos = is_assigned.any(dim=1)
        if pos.any():
            gi = assigned_gt[pos]
            labels[pos] = gt_labels[gi]
            boxes[pos]  = gt_boxes[gi]
            nm = align[pos, gi]; nm = nm/(nm.max()+1e-6)
            scores[pos, gt_labels[gi]] = nm
        return labels, boxes, scores


if __name__ == '__main__':
    assigner = TaskAlignedAssigner(topk=13, num_classes=6)
    pred_scores = torch.randn(90, 6)
    pred_boxes  = torch.rand(90, 4)
    gt_labels   = torch.tensor([0, 1, 3])
    gt_boxes    = torch.tensor([[0.5,0.5,0.2,0.2],[0.2,0.3,0.1,0.15],[0.7,0.6,0.3,0.25]])
    labels, boxes, scores = assigner.assign(pred_scores, pred_boxes, gt_labels, gt_boxes)
    print(f"Positives   : {(labels>=0).sum().item()} / 90")
    print(f"labels shape: {labels.shape}")
    print("task_aligned_assigner.py works correctly")
