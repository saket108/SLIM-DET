# model/severity_head.py
"""SeverityHead — predicts per-query damage severity score."""

import torch
import torch.nn as nn


class SeverityHead(nn.Module):
    """
    Predicts a continuous severity score per query.
    Output: raw logit — apply sigmoid for [0,1] score.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, queries):
        """queries: [B,Q,D] → pred_severity: [B,Q]"""
        return self.net(queries).squeeze(-1)


if __name__ == '__main__':
    head = SeverityHead()
    q    = torch.randn(2, 90, 256)
    sev  = head(q)
    print(f"pred_severity : {sev.shape}")
    print(f"sigmoid range : [{sev.sigmoid().min():.3f}, {sev.sigmoid().max():.3f}]")
    print("severity_head.py works correctly")
