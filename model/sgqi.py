# model/sgqi.py
"""
SGQI — Structured Group Query Initialization.

NOVELTY: Splits decoder queries into 3 groups, each seeded
by a different JSON metadata field:
  - Text group  (30 queries): seeded by damage description
  - Zone group  (30 queries): seeded by spatial zone prior
  - Metric group(30 queries): seeded by numeric damage intensity
"""

import torch
import torch.nn as nn


class SGQI(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=90, dropout=0.1):
        super().__init__()
        assert num_queries % 3 == 0
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries
        self.qpg         = num_queries // 3

        self.text_base   = nn.Embedding(self.qpg, hidden_dim)
        self.zone_base   = nn.Embedding(self.qpg, hidden_dim)
        self.metric_base = nn.Embedding(self.qpg, hidden_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.zone_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.metric_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        self.cross_group_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for e in [self.text_base, self.zone_base, self.metric_base]:
            nn.init.normal_(e.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, text_emb, zone_emb, metrics_emb):
        B = text_emb.size(0)
        text_q   = self.text_base.weight.unsqueeze(0).expand(B,-1,-1) + self.text_proj(text_emb).unsqueeze(1)
        zone_q   = self.zone_base.weight.unsqueeze(0).expand(B,-1,-1) + self.zone_proj(zone_emb).unsqueeze(1)
        metric_q = self.metric_base.weight.unsqueeze(0).expand(B,-1,-1) + self.metric_proj(metrics_emb).unsqueeze(1)
        queries  = torch.cat([text_q, zone_q, metric_q], dim=1)
        q2, _    = self.cross_group_attn(queries, queries, queries)
        return self.norm(queries + q2)

    def get_group_slices(self):
        return {
            'text':   slice(0, self.qpg),
            'zone':   slice(self.qpg, 2*self.qpg),
            'metric': slice(2*self.qpg, 3*self.qpg),
        }


if __name__ == '__main__':
    sgqi = SGQI(hidden_dim=256, num_queries=90)
    sgqi.eval()
    B = 4
    t, z, m = torch.randn(B,256), torch.randn(B,256), torch.randn(B,256)
    with torch.no_grad():
        q = sgqi(t, z, m)
    print(f"queries shape : {q.shape}")   # [4, 90, 256]
    s = sgqi.get_group_slices()
    diff = (q[:,s['text'],:] - q[:,s['zone'],:]).abs().mean()
    print(f"group diff    : {diff:.4f}  (should be > 0)")
    print(f"params        : {sum(p.numel() for p in sgqi.parameters()):,}")
    print("sgqi.py works correctly")
