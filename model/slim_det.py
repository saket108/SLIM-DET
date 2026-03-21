# model/slim_det.py  — FIXED VERSION
"""
SLIM-Det with pretrained ConvNeXt-Tiny backbone.
Key fix: replaced random-init GhostCSP with pretrained timm backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

from model.text_encoder    import TextEncoder
from model.zone_encoder    import ZoneEncoder
from model.metrics_encoder import MetricsEncoder, MetricsNormalizer
from model.smfe            import SMFE
from model.cafpn           import CAFPN
from model.ghost_csp_backbone import GhostCSPBackbone
from model.sgqi            import SGQI
from model.rwda_decoder    import RWDADecoder
from model.detector_head   import DetectorHead
from model.severity_head   import SeverityHead
from model.scf             import SCF


# ── Pretrained backbone wrapper ───────────────────────────────
class PretrainedBackbone(nn.Module):
    """
    Wraps a pretrained timm backbone to produce 4-scale features.
    Default: ConvNeXtV2-Tiny pretrained on ImageNet-22k
    — strong features, 28M params, 4 clean hierarchical stages.
    """
    def __init__(
        self,
        model_name: str = 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        hidden_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.using_timm = False
        self.apply_post_film = False

        in_channels = [64, 128, 256, 512]
        if timm is not None:
            try:
                print(f"  Loading backbone: {model_name} (pretrained={pretrained})")
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                )
                in_channels = list(self.backbone.feature_info.channels())
                self.using_timm = True
                self.apply_post_film = True
            except Exception as exc:
                if pretrained:
                    try:
                        print(f"  Retrying backbone without pretrained weights: {exc}")
                        self.backbone = timm.create_model(
                            model_name,
                            pretrained=False,
                            features_only=True,
                            out_indices=(0, 1, 2, 3),
                        )
                        in_channels = list(self.backbone.feature_info.channels())
                        self.using_timm = True
                        self.apply_post_film = True
                    except Exception as fallback_exc:
                        print(f"  Falling back to GhostCSPBackbone: {fallback_exc}")
                        self.backbone = GhostCSPBackbone(
                            channels=in_channels,
                            hidden_dim=hidden_dim,
                        )
                else:
                    print(f"  Falling back to GhostCSPBackbone: {exc}")
                    self.backbone = GhostCSPBackbone(
                        channels=in_channels,
                        hidden_dim=hidden_dim,
                    )
        else:
            print("  timm not installed - using GhostCSPBackbone fallback")
            self.backbone = GhostCSPBackbone(
                channels=in_channels,
                hidden_dim=hidden_dim,
            )

        # Lateral projections: backbone channels → hidden_dim
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for c in in_channels
        ])

        # FiLM conditioning for stages 2+3 (metric gating)
        self.film_gamma = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        self.film_beta = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])

        for g, b in zip(self.film_gamma, self.film_beta):
            nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)

    def forward(self, x, metrics_emb=None):
        if self.using_timm:
            raw_feats = self.backbone(x)
        else:
            raw_feats = self.backbone(x, metrics_emb)
        feats = [l(f) for l, f in zip(self.laterals, raw_feats)]

        # FiLM condition last 2 stages with damage metrics
        if self.apply_post_film and metrics_emb is not None:
            for i, (g, b) in enumerate(zip(self.film_gamma, self.film_beta)):
                idx   = i + 2   # stages 2 and 3
                gamma = g(metrics_emb).view(-1, feats[idx].size(1), 1, 1)
                beta  = b(metrics_emb).view(-1, feats[idx].size(1), 1, 1)
                feats[idx] = feats[idx] * (1 + gamma) + beta

        return feats   # [P2, P3, P4, P5]


# ── Main model ────────────────────────────────────────────────
class SLIMDet(nn.Module):
    def __init__(
        self,
        num_classes:   int  = 6,
        hidden_dim:    int  = 256,
        num_queries:   int  = 90,
        num_layers:    int  = 4,
        text_model:    str  = 'minilm',
        backbone_name: str  = 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        prompt_mode:   str  = 'full',
        freeze_text:   bool = True,
        text_local_files_only: bool = False,
        pretrained_backbone: bool = True,
        image_only:    bool = False,
        use_scf:       bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries
        self.image_only  = image_only
        self.use_scf     = use_scf

        # ── Encoders / image-only baseline branch ─────────────
        if image_only:
            self.image_only_queries = nn.Embedding(num_queries, hidden_dim)
            self.image_only_context = nn.Parameter(torch.zeros(3, hidden_dim))
            nn.init.normal_(self.image_only_queries.weight, std=0.02)
            nn.init.normal_(self.image_only_context, std=0.02)
        else:
            self.text_encoder    = TextEncoder(
                model_name = text_model,
                hidden_dim = hidden_dim,
                pooling    = 'mean',
                freeze     = freeze_text,
                local_files_only = text_local_files_only,
            )
            self.zone_encoder    = ZoneEncoder(hidden_dim=hidden_dim)
            self.metrics_norm    = MetricsNormalizer()
            self.metrics_encoder = MetricsEncoder(input_dim=4, hidden_dim=hidden_dim)
            self.smfe            = SMFE(hidden_dim=hidden_dim, num_heads=4)

        # ── Pretrained backbone ────────────────────────────────
        self.backbone = PretrainedBackbone(
            model_name = backbone_name,
            hidden_dim = hidden_dim,
            pretrained = pretrained_backbone,
        )
        self.cafpn = CAFPN(
            in_channels  = [hidden_dim] * 4,
            out_channels = hidden_dim,
            context_dim  = hidden_dim,
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Query init + decoder ───────────────────────────────
        self.sgqi    = SGQI(hidden_dim=hidden_dim, num_queries=num_queries)
        self.decoder = RWDADecoder(
            hidden_dim = hidden_dim,
            num_heads  = 8,
            num_points = 4,
            num_layers = num_layers,
            ffn_dim    = hidden_dim * 4,
        )

        # ── Heads ──────────────────────────────────────────────
        self.detect_head   = DetectorHead(hidden_dim=hidden_dim, num_classes=num_classes)
        self.severity_head = SeverityHead(hidden_dim=hidden_dim)
        if use_scf:
            self.scf = SCF(num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, images, prompts, zones, metrics):
        # 1. Encode modalities or use learned visual-only context
        if self.image_only:
            batch_size = images.size(0)
            metrics_emb = None
            ctx_tokens = self.image_only_context.unsqueeze(0).expand(batch_size, -1, -1)
            ctx_emb = ctx_tokens.mean(dim=1)
        else:
            text_emb    = self.text_encoder(prompts)
            zone_emb    = self.zone_encoder(zones)
            metrics_emb = self.metrics_encoder(self.metrics_norm(metrics))
            ctx_tokens, _ = self.smfe(text_emb, zone_emb, metrics_emb)
            ctx_emb     = ctx_tokens.mean(dim=1)

        # 2. Encode image
        feats = self.backbone(images, metrics_emb)
        feats = self.cafpn(feats, ctx_emb)
        p4    = feats[2]
        B, D, H, W = p4.shape
        vis_mem = self.visual_proj(p4.flatten(2).permute(0, 2, 1))

        # 3. Initialize queries
        if self.image_only:
            queries = self.image_only_queries.weight.unsqueeze(0).expand(B, -1, -1)
        else:
            queries = self.sgqi(text_emb, zone_emb, metrics_emb)

        # 4. Decode
        queries, aux_list = self.decoder(queries, vis_mem, ctx_tokens)

        # 5. Predict
        logits, boxes = self.detect_head(queries)
        severity      = self.severity_head(queries)
        scores        = self.scf(logits, severity, boxes)['scores'] \
                        if self.use_scf else logits.sigmoid()

        aux_outputs = []
        for aq in aux_list:
            al, ab = self.detect_head(aq)
            asev   = self.severity_head(aq)
            aux_outputs.append({
                'pred_logits': al, 'pred_boxes': ab, 'pred_severity': asev
            })

        return {
            'pred_logits':   logits,
            'pred_boxes':    boxes,
            'pred_severity': severity,
            'scores':        scores,
            'aux_outputs':   aux_outputs,
        }

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


if __name__ == '__main__':
    model = SLIMDet(num_classes=6, hidden_dim=256, num_queries=90)
    model.eval()
    B = 2
    images  = torch.randn(B, 3, 640, 640)
    prompts = ["Damage: dent. Zone: central. Severity: low."] * B
    zones   = ['central'] * B
    metrics = torch.rand(B, 4)
    with torch.no_grad():
        out = model(images, prompts, zones, metrics)
    print(f"pred_logits  : {out['pred_logits'].shape}")
    print(f"pred_boxes   : {out['pred_boxes'].shape}")
    print(f"aux_outputs  : {len(out['aux_outputs'])}")
    pc = model.param_count()
    print(f"total params : {pc['total']:,}")
    print(f"trainable    : {pc['trainable']:,}")
    print("slim_det.py works correctly")
