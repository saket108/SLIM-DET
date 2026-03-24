"""Detection heads used by SLIM-Det decoder queries."""

import math

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
    Predict class logits, normalized boxes, and an optional IoU-quality logit.

    The quality branch is the SLIM-Det-native adaptation of the downloaded
    detection stack: it uses query-space task alignment instead of a dense FPN
    head, but preserves the same idea of supervising a separate localization
    quality score and blending it into the inference confidence.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 6,
        use_quality_head: bool = False,
        tower_depth: int = 2,
    ):
        super().__init__()
        self.use_quality_head = use_quality_head

        self.cls_tower = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=tower_depth)
        self.reg_tower = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=tower_depth)

        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        if use_quality_head:
            self.task_align = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.quality_head = nn.Linear(hidden_dim, 1)

        self._init_biases()

    def _init_biases(self, prior_prob: float = 0.01):
        nn.init.normal_(self.cls_head.weight, std=0.01)
        cls_bias = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.cls_head.bias, cls_bias)
        if self.use_quality_head:
            nn.init.zeros_(self.quality_head.weight)
            nn.init.zeros_(self.quality_head.bias)

    def forward(self, queries: torch.Tensor):
        """
        Args:
            queries : [B, Q, hidden_dim]

        Returns:
            pred_logits   : [B, Q, num_classes] raw logits
            pred_boxes    : [B, Q, 4] sigmoid cx,cy,w,h in [0,1]
            pred_quality  : [B, Q] raw IoU-quality logits, or None
        """
        cls_feat = self.cls_tower(queries)
        reg_feat = self.reg_tower(queries)

        pred_logits = self.cls_head(cls_feat)
        pred_boxes = self.box_head(reg_feat).sigmoid()

        pred_quality = None
        if self.use_quality_head:
            aligned = self.task_align(torch.cat([cls_feat, reg_feat], dim=-1))
            pred_quality = self.quality_head(aligned).squeeze(-1)

        return pred_logits, pred_boxes, pred_quality


if __name__ == '__main__':
    head = DetectorHead(hidden_dim=256, num_classes=6, use_quality_head=True)
    q = torch.randn(2, 90, 256)
    logits, boxes, quality = head(q)
    print(f"pred_logits  : {logits.shape}")
    print(f"pred_boxes   : {boxes.shape}")
    print(f"pred_quality : {quality.shape if quality is not None else None}")
    print(f"boxes range  : [{boxes.min():.3f}, {boxes.max():.3f}]")
    print("detector_head.py works correctly")
