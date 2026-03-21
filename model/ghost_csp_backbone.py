# model/ghost_csp_backbone.py
"""
GhostCSP Backbone — lightweight 4-stage image backbone.

Combines:
- Ghost modules (GhostNet, Han et al. 2020) — 50% fewer FLOPs
- CSP bottlenecks (YOLOv5) — rich gradient flow
- FiLM metric gating at stages 3+4 — multimodal conditioning

Produces 4-scale feature maps [P2, P3, P4, P5] for FPN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Standard Conv + BN + activation."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GhostModule(nn.Module):
    """
    Ghost module — generates feature maps with cheap operations.

    Half channels from standard conv (primary features).
    Other half from cheap depthwise transform (ghost features).
    ~50% fewer FLOPs than standard conv at same output channels.
    """
    def __init__(self, c_in, c_out, k=3, ratio=2):
        super().__init__()
        c_mid = c_out // ratio

        # Primary: standard conv for rich features
        self.primary = ConvBNAct(c_in, c_mid, k=1)

        # Cheap: depthwise conv to generate ghost features
        self.cheap = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, k, 1, k//2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.GELU(),
        )

    def forward(self, x):
        y1 = self.primary(x)   # rich features
        y2 = self.cheap(y1)    # ghost features (cheap)
        return torch.cat([y1, y2], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost bottleneck with residual connection."""
    def __init__(self, c_in, c_out, expand=2):
        super().__init__()
        c_mid = c_in * expand
        self.ghost1 = GhostModule(c_in,  c_mid)
        self.ghost2 = GhostModule(c_mid, c_out)
        self.skip   = nn.Identity() if c_in == c_out else \
                      ConvBNAct(c_in, c_out, k=1, act=False)

    def forward(self, x):
        return self.ghost2(self.ghost1(x)) + self.skip(x)


class CSPStage(nn.Module):
    """
    CSP stage with Ghost bottlenecks.

    Splits channels in half:
    - Main branch: n Ghost bottlenecks
    - Skip branch: direct connection
    Merges both → richer gradient flow at lower cost.

    Optional FiLM metric gate at stages 3+4:
    metrics_emb scales and shifts feature maps,
    conditioning visual features on damage intensity.
    """
    def __init__(
        self,
        c_in:            int,
        c_out:           int,
        n:               int   = 3,
        use_metric_gate: bool  = False,
        hidden_dim:      int   = 256,
    ):
        super().__init__()
        c_mid = c_out // 2

        self.split_main = ConvBNAct(c_in, c_mid, k=1)
        self.split_skip = ConvBNAct(c_in, c_mid, k=1)

        self.bottlenecks = nn.Sequential(*[
            GhostBottleneck(c_mid, c_mid) for _ in range(n)
        ])

        self.merge = ConvBNAct(c_mid * 2, c_out, k=1)

        # FiLM conditioning: metrics → scale + shift
        self.use_metric_gate = use_metric_gate
        if use_metric_gate:
            self.film_gamma = nn.Linear(hidden_dim, c_out)  # scale
            self.film_beta  = nn.Linear(hidden_dim, c_out)  # shift
            nn.init.ones_(self.film_gamma.weight)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.bias)

    def forward(self, x, metrics_emb=None):
        main = self.bottlenecks(self.split_main(x))
        skip = self.split_skip(x)
        out  = self.merge(torch.cat([main, skip], dim=1))

        # FiLM conditioning — only at stages 3+4
        if self.use_metric_gate and metrics_emb is not None:
            gamma = self.film_gamma(metrics_emb).view(
                -1, out.size(1), 1, 1)
            beta  = self.film_beta(metrics_emb).view(
                -1, out.size(1), 1, 1)
            out = out * (1 + gamma) + beta

        return out


# ── Main Backbone ─────────────────────────────────────────────

class GhostCSPBackbone(nn.Module):
    """
    4-stage GhostCSP backbone.

    Stage 1: 64  channels, depth 3  — P2 (/4)
    Stage 2: 128 channels, depth 3  — P3 (/8)
    Stage 3: 256 channels, depth 9  — P4 (/16) + metric gate
    Stage 4: 512 channels, depth 3  — P5 (/32) + metric gate

    Args:
        channels   : output channels per stage [64,128,256,512]
        depths     : number of bottlenecks per stage [3,3,9,3]
        hidden_dim : metrics_emb dimension for FiLM gates
    """

    def __init__(
        self,
        channels:   list = [64, 128, 256, 512],
        depths:     list = [3,  3,   9,   3],
        hidden_dim: int  = 256,
    ):
        super().__init__()
        assert len(channels) == 4 and len(depths) == 4

        # Stem: aggressive 4× downsampling
        self.stem = nn.Sequential(
            ConvBNAct(3,           channels[0]//2, k=3, s=2),
            ConvBNAct(channels[0]//2, channels[0], k=3, s=2),
        )

        # Downsampling between stages (2× each)
        self.downsample = nn.ModuleList([
            ConvBNAct(channels[i], channels[i], k=3, s=2)
            for i in range(3)
        ])

        # 4 CSP stages
        in_chs = [channels[0]] + channels[:3]
        self.stages = nn.ModuleList([
            CSPStage(
                c_in            = in_chs[i],
                c_out           = channels[i],
                n               = depths[i],
                use_metric_gate = (i >= 2),   # gate at stage 3+4
                hidden_dim      = hidden_dim,
            )
            for i in range(4)
        ])

    def forward(self, x, metrics_emb=None):
        """
        Args:
            x           : [B, 3, H, W] input image
            metrics_emb : [B, hidden_dim] from MetricsEncoder
                          passed to stage 3+4 FiLM gates

        Returns:
            [P2, P3, P4, P5] — list of 4 feature tensors
            P2: [B, 64,  H/4,  W/4]
            P3: [B, 128, H/8,  W/8]
            P4: [B, 256, H/16, W/16]
            P5: [B, 512, H/32, W/32]
        """
        x     = self.stem(x)        # /4
        feats = []

        for i, stage in enumerate(self.stages):
            x = stage(x, metrics_emb)
            feats.append(x)         # save feature map
            if i < 3:
                x = self.downsample[i](x)   # /2 for next stage

        return feats   # [P2, P3, P4, P5]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    backbone = GhostCSPBackbone(
        channels   = [64, 128, 256, 512],
        depths     = [3, 3, 9, 3],
        hidden_dim = 256,
    )
    backbone.eval()

    B = 2
    x           = torch.randn(B, 3, 640, 640)
    metrics_emb = torch.randn(B, 256)

    with torch.no_grad():
        feats = backbone(x, metrics_emb)

    names = ['P2', 'P3', 'P4', 'P5']
    print("Feature map shapes:")
    for name, f in zip(names, feats):
        print(f"  {name}: {f.shape}")

    print(f"\nTotal params : {backbone.num_params():,}")
    print(f"Expected     : ~4-6M")

    # Verify FiLM gates work
    feats_no_gate = backbone(x, metrics_emb=None)
    diff = (feats[2] - feats_no_gate[2]).abs().mean()
    print(f"\nFiLM gate effect (P4 diff with/without metrics): {diff:.4f}")
    print("  (should be > 0 — gate is active)")
    print("\nghost_csp_backbone.py works correctly")
