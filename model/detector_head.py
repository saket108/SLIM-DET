# model/detector_head.py
"""DetectorHead — predicts class logits + bounding boxes from decoder queries."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DetectorHead(nn.Module):
    """
    Predicts class logits and normalized bounding boxes from queries.

    Args:
        hidden_dim  : query dimension (default 256)
        num_classes : number of damage classes (default 6)
    """
    def __init__(self, hidden_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, queries: torch.Tensor):
        """
        Args:
            queries : [B, Q, hidden_dim]

        Returns:
            pred_logits : [B, Q, num_classes]  raw logits
            pred_boxes  : [B, Q, 4]            sigmoid cx,cy,w,h in [0,1]
        """
        pred_logits = self.cls_head(queries)
        pred_boxes  = self.box_head(queries).sigmoid()
        return pred_logits, pred_boxes


if __name__ == '__main__':
    head = DetectorHead(hidden_dim=256, num_classes=6)
    q    = torch.randn(2, 90, 256)
    logits, boxes = head(q)
    print(f"pred_logits : {logits.shape}")   # [2, 90, 6]
    print(f"pred_boxes  : {boxes.shape}")    # [2, 90, 4]
    print(f"boxes range : [{boxes.min():.3f}, {boxes.max():.3f}]  (should be 0-1)")
    print("detector_head.py works correctly")
