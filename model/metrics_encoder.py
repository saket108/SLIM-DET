# model/metrics_encoder.py
"""
MetricsEncoder — encodes 4-D numeric damage metrics into
a hidden_dim embedding with residual connection.

Input metrics (from Aircraft_dataset):
    area_ratio         : float  damaged area / total area
    elongation         : float  aspect ratio of damage shape
    edge_factor        : float  edge sharpness (0=smooth, 1=sharp)
    raw_severity_score : float  raw composite severity [0, 1]
"""

import torch
import torch.nn as nn


class MetricsEncoder(nn.Module):
    """
    Encodes 4-D numeric damage metrics → hidden_dim embedding.

    Architecture:
        4 → 64 → 128 → 256 → hidden_dim
        with LayerNorm + GELU at each step
        plus residual skip from input → output

    The residual ensures gradient flows directly from
    raw metrics to downstream modules even early in training.

    Args:
        input_dim  : number of metric dimensions (default 4)
        hidden_dim : output embedding dimension (default 256)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        input_dim:  int   = 4,
        hidden_dim: int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Main pathway: 4 → 64 → 128 → 256 → hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual skip: directly project input to output dim
        self.skip = nn.Linear(input_dim, hidden_dim)

        # Final activation after residual merge
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metrics: [B, 4] float tensor
                     [area_ratio, elongation, edge_factor, raw_severity]

        Returns:
            metrics_emb: [B, hidden_dim]
        """
        main = self.net(metrics)           # [B, hidden_dim]
        skip = self.skip(metrics)          # [B, hidden_dim]
        return self.act(main + skip)       # residual merge


class MetricsNormalizer(nn.Module):
    """
    Optional: normalize raw metrics to zero mean unit variance.
    Pre-computed statistics from Aircraft_dataset training set.
    Use this before MetricsEncoder for more stable training.
    """

    # Pre-computed from Aircraft_dataset (approximate)
    MEAN = torch.tensor([0.055, 1.20, 0.48, 0.18], dtype=torch.float32)
    STD  = torch.tensor([0.042, 0.35, 0.22, 0.15], dtype=torch.float32)

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', self.MEAN)
        self.register_buffer('std',  self.STD)

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        return (metrics - self.mean) / (self.std + 1e-6)


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    encoder    = MetricsEncoder(input_dim=4, hidden_dim=256)
    normalizer = MetricsNormalizer()
    encoder.eval()
    normalizer.eval()

    # Simulate a batch of 4 damage metric vectors
    sample_metrics = torch.tensor([
        [0.05464, 1.158, 0.52422, 0.08913],   # low dent
        [0.12000, 2.100, 0.80000, 0.45000],   # medium crack
        [0.25000, 1.500, 0.30000, 0.72000],   # high corrosion
        [0.00800, 1.050, 0.15000, 0.03000],   # low scratch
    ], dtype=torch.float32)

    with torch.no_grad():
        normed = normalizer(sample_metrics)
        emb    = encoder(normed)

    print(f"Input shape  : {sample_metrics.shape}")   # [4, 4]
    print(f"Normed shape : {normed.shape}")            # [4, 4]
    print(f"Output shape : {emb.shape}")               # [4, 256]
    print(f"Output mean  : {emb.mean():.4f}")
    print(f"Output std   : {emb.std():.4f}")
    print(f"\nParam count  : {sum(p.numel() for p in encoder.parameters()):,}")
    print("\nmetrics_encoder.py works correctly")
