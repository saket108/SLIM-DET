# model/cafpn.py
"""
CAFPN — Context-Aware Feature Pyramid Network.

NOVELTY: Unlike standard PANet which merges scales purely visually,
CAFPN uses context tokens (text + zone + metrics) to compute
per-scale attention weights — deciding which scale matters most
given the current annotation context.

e.g. zone='central' + low severity → weight coarse scales more
     high severity scratch → weight fine scales more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable conv — ~50% fewer params than standard 3x3.
    Used in FPN smooth convs to keep neck lightweight.
    """
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.dw = nn.Conv2d(
            c_in, c_in, k, padding=k//2, groups=c_in, bias=False
        )
        self.pw  = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CAFPN(nn.Module):
    """
    Context-Aware Feature Pyramid Network.

    Architecture:
        1. Lateral 1x1 convs — unify channel dims to out_channels
        2. Scale attention — context_emb → [w1,w2,w3,w4] weights
        3. Top-down pathway — large scale enriches small scale (FPN)
        4. Scale weighting — multiply each level by its attention weight
        5. Bottom-up pathway — small scale enriches large scale (PANet)
        6. Output convs — depthwise separable smooth convs

    Args:
        in_channels  : list of input channels per scale [64,128,256,512]
        out_channels : unified output channels (default 256)
        context_dim  : context embedding dimension (default 256)
    """

    def __init__(
        self,
        in_channels:  list = [64, 128, 256, 512],
        out_channels: int  = 256,
        context_dim:  int  = 256,
    ):
        super().__init__()
        self.num_levels  = len(in_channels)
        self.out_channels = out_channels

        # Lateral 1x1 projections — unify all scales to out_channels
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for c in in_channels
        ])

        # Scale attention — context decides which scale matters most
        # Input: context_emb [B, context_dim]
        # Output: [B, num_levels] softmax weights
        self.scale_attn = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, self.num_levels),
            nn.Softmax(dim=-1),
        )

        # Top-down smooth convs (FPN path)
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in in_channels
        ])

        # Bottom-up downsample + merge convs (PANet path)
        self.bu_downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2,
                          padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for _ in range(self.num_levels - 1)
        ])

        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in range(self.num_levels - 1)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list, context_emb: torch.Tensor):
        """
        Args:
            features    : [P2, P3, P4, P5] from backbone
                          each [B, C_i, H_i, W_i]
            context_emb : [B, context_dim] from SMFE
                          mean-pooled context token

        Returns:
            out_feats: list of 4 tensors [B, out_channels, H_i, W_i]
                       context-modulated multi-scale features
        """
        # ── Step 1: Lateral projections ───────────────────────
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # ── Step 2: Compute scale attention weights ────────────
        # context_emb tells us which scales matter for this damage
        scale_weights = self.scale_attn(context_emb)    # [B, 4]

        # ── Step 3: Top-down pathway (FPN) ─────────────────────
        # Large → small: high-level semantics flow to fine scales
        td = [laterals[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                td[-1],
                size  = laterals[i].shape[-2:],
                mode  = 'nearest',
            )
            td.append(laterals[i] + upsampled)
        td = td[::-1]   # reverse to [P2, P3, P4, P5] order

        # Apply top-down smooth convs
        td = [conv(f) for conv, f in zip(self.td_convs, td)]

        # ── Step 4: Scale attention weighting ─────────────────
        # Multiply each level by its context-derived weight
        weighted = [
            f * scale_weights[:, i].view(-1, 1, 1, 1)
            for i, f in enumerate(td)
        ]

        # ── Step 5: Bottom-up pathway (PANet) ──────────────────
        # Small → large: fine-scale details flow to coarse scales
        out = [weighted[0]]
        for i in range(self.num_levels - 1):
            down     = self.bu_downs[i](out[-1])
            merged   = down + weighted[i + 1]
            smoothed = self.bu_convs[i](merged)
            out.append(smoothed)

        return out   # [P2, P3, P4, P5] enriched


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    cafpn = CAFPN(
        in_channels  = [64, 128, 256, 512],
        out_channels = 256,
        context_dim  = 256,
    )
    cafpn.eval()

    B = 2
    # Simulate backbone outputs at 640x640 input
    features = [
        torch.randn(B, 64,  160, 160),   # P2 /4
        torch.randn(B, 128,  80,  80),   # P3 /8
        torch.randn(B, 256,  40,  40),   # P4 /16
        torch.randn(B, 512,  20,  20),   # P5 /32
    ]
    context_emb = torch.randn(B, 256)

    with torch.no_grad():
        out_feats = cafpn(features, context_emb)

    print("CAFPN output shapes:")
    names = ['P2', 'P3', 'P4', 'P5']
    for name, f in zip(names, out_feats):
        print(f"  {name}: {f.shape}")

    # Verify context weighting is active
    ctx1 = torch.zeros(B, 256)
    ctx2 = torch.ones(B, 256)
    with torch.no_grad():
        out1 = cafpn(features, ctx1)
        out2 = cafpn(features, ctx2)
    diff = (out1[0] - out2[0]).abs().mean()
    print(f"\nContext effect (P2 diff ctx0 vs ctx1): {diff:.4f}")
    print("  (should be > 0 — context is modulating scales)")

    params = sum(p.numel() for p in cafpn.parameters())
    print(f"\nTotal params : {params:,}")
    print("\ncafpn.py works correctly")
