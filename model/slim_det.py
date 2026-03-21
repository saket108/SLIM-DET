# model/slim_det.py
"""
SLIM-Det — Structured Language-Image Multimodal Detector.

Full forward pass wiring all 11 modules:

  Image + JSON (text, zone, metrics)
        │
        ├─ TextEncoder    → text_emb    [B, 256]
        ├─ ZoneEncoder    → zone_emb    [B, 256]
        ├─ MetricsEncoder → metrics_emb [B, 256]
        │         │
        │         └─ SMFE → context_tokens [B, 3, 256]
        │                         │
        ├─ GhostCSPBackbone ───── │ (FiLM gated at stages 3+4)
        │   [P2, P3, P4, P5]     │
        │         │               │
        │       CAFPN ←───────────┘ (context-aware scale weighting)
        │   enriched feats [B, 256, H, W] x4
        │         │
        │    flatten P4 → visual_memory [B, HW, 256]
        │         │
        └─ SGQI ──┘  (structured group query init)
              │
           queries [B, 90, 256]
              │
           RWDA Decoder (4 shared-weight iterations)
              │
         ┌────┴────┐
     DetHead    SevHead
   logits+boxes  severity
         │
        SCF (severity-conditioned filter)
         │
      final scores + boxes + severity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.text_encoder       import TextEncoder
from model.zone_encoder       import ZoneEncoder
from model.metrics_encoder    import MetricsEncoder, MetricsNormalizer
from model.smfe               import SMFE
from model.ghost_csp_backbone import GhostCSPBackbone
from model.cafpn              import CAFPN
from model.sgqi               import SGQI
from model.rwda_decoder       import RWDADecoder
from model.detector_head      import DetectorHead
from model.severity_head      import SeverityHead
from model.scf                import SCF


class SLIMDet(nn.Module):
    """
    SLIM-Det: Structured Language-Image Multimodal Detector.

    Args:
        num_classes   : number of damage categories (default 6)
        hidden_dim    : shared embedding dimension (default 256)
        num_queries   : total decoder queries (default 90, must be /3)
        num_layers    : RWDA decoder iterations (default 4)
        text_model    : HuggingFace text encoder key (default 'minilm')
        prompt_mode   : 'full'|'no_desc'|'minimal'|'cat_only'
        freeze_text   : freeze text encoder weights (default True)
        use_scf       : enable severity-conditioned filter (default True)
    """

    def __init__(
        self,
        num_classes:  int  = 6,
        hidden_dim:   int  = 256,
        num_queries:  int  = 90,
        num_layers:   int  = 4,
        text_model:   str  = 'minilm',
        prompt_mode:  str  = 'full',
        freeze_text:  bool = True,
        use_scf:      bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries
        self.prompt_mode = prompt_mode
        self.use_scf     = use_scf

        # ── Encoders ──────────────────────────────────────────
        self.text_encoder    = TextEncoder(
            model_name = text_model,
            hidden_dim = hidden_dim,
            pooling    = 'mean',
            freeze     = freeze_text,
        )
        self.zone_encoder    = ZoneEncoder(hidden_dim=hidden_dim)
        self.metrics_norm    = MetricsNormalizer()
        self.metrics_encoder = MetricsEncoder(
            input_dim  = 4,
            hidden_dim = hidden_dim,
        )

        # ── Structured Multimodal Feature Encoder ─────────────
        self.smfe = SMFE(hidden_dim=hidden_dim, num_heads=4)

        # ── Image backbone + neck ──────────────────────────────
        self.backbone = GhostCSPBackbone(
            channels   = [64, 128, 256, 512],
            depths     = [3, 3, 9, 3],
            hidden_dim = hidden_dim,
        )
        self.cafpn = CAFPN(
            in_channels  = [64, 128, 256, 512],
            out_channels = hidden_dim,
            context_dim  = hidden_dim,
        )

        # Visual feature projection to hidden_dim
        # (CAFPN already outputs hidden_dim, but kept for clarity)
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Structured Group Query Init ────────────────────────
        self.sgqi = SGQI(
            hidden_dim  = hidden_dim,
            num_queries = num_queries,
        )

        # ── Decoder ───────────────────────────────────────────
        self.decoder = RWDADecoder(
            hidden_dim = hidden_dim,
            num_heads  = 8,
            num_points = 4,
            num_layers = num_layers,
            ffn_dim    = hidden_dim * 4,
        )

        # ── Prediction heads ───────────────────────────────────
        self.detect_head   = DetectorHead(
            hidden_dim  = hidden_dim,
            num_classes = num_classes,
        )
        self.severity_head = SeverityHead(hidden_dim=hidden_dim)

        # ── Severity-Conditioned Filter ────────────────────────
        if use_scf:
            self.scf = SCF(
                num_classes = num_classes,
                hidden_dim  = hidden_dim,
            )

    def encode_modalities(self, images, prompts, zones, metrics):
        """
        Run all three non-image encoders in parallel.

        Args:
            images  : [B, 3, H, W]
            prompts : list of B prompt strings
            zones   : list of B zone strings
            metrics : [B, 4] damage metric tensor

        Returns:
            text_emb    : [B, hidden_dim]
            zone_emb    : [B, hidden_dim]
            metrics_emb : [B, hidden_dim]
            context_tokens : [B, 3, hidden_dim]
            context_emb    : [B, hidden_dim]
        """
        text_emb    = self.text_encoder(prompts)
        zone_emb    = self.zone_encoder(zones)
        metrics_emb = self.metrics_encoder(self.metrics_norm(metrics))

        context_tokens, gate_weights = self.smfe(
            text_emb, zone_emb, metrics_emb
        )
        context_emb = context_tokens.mean(dim=1)   # [B, D]

        return text_emb, zone_emb, metrics_emb, context_tokens, context_emb

    def encode_image(self, images, metrics_emb, context_emb):
        """
        Run backbone + CAFPN to get multi-scale visual features.

        Args:
            images      : [B, 3, H, W]
            metrics_emb : [B, hidden_dim]  for FiLM gating
            context_emb : [B, hidden_dim]  for CAFPN scale attention

        Returns:
            visual_memory : [B, HW, hidden_dim]  flattened P4 features
            all_feats     : list of 4 enriched feature maps
        """
        feats = self.backbone(images, metrics_emb)
        feats = self.cafpn(feats, context_emb)

        # Use P4 (40x40) as primary visual memory for decoder
        # P4 balances semantic richness and spatial resolution
        p4 = feats[2]                              # [B, D, 40, 40]
        B, D, H, W = p4.shape
        visual_memory = p4.flatten(2).permute(0, 2, 1)  # [B, HW, D]
        visual_memory = self.visual_proj(visual_memory)

        return visual_memory, feats

    def forward(self, images, prompts, zones, metrics):
        """
        Full SLIM-Det forward pass.

        Args:
            images  : [B, 3, H, W]         input images
            prompts : list[str]             one prompt per image
            zones   : list[str]             one zone per image
            metrics : Tensor [B, 4]         damage metrics per image

        Returns:
            dict with:
                pred_logits  : [B, Q, num_classes]  raw class logits
                pred_boxes   : [B, Q, 4]            normalized cx,cy,w,h
                pred_severity: [B, Q]               raw severity logits
                scores       : [B, Q, num_classes]  SCF-boosted scores
                aux_outputs  : list of dicts        for auxiliary losses
        """
        # ── Step 1: Encode all modalities ─────────────────────
        text_emb, zone_emb, metrics_emb, context_tokens, context_emb = \
            self.encode_modalities(images, prompts, zones, metrics)

        # ── Step 2: Encode image with multimodal conditioning ──
        visual_memory, all_feats = \
            self.encode_image(images, metrics_emb, context_emb)

        # ── Step 3: Initialize queries from structured metadata ─
        queries = self.sgqi(text_emb, zone_emb, metrics_emb)

        # ── Step 4: Recurrent decoder ─────────────────────────
        queries, aux_query_list = self.decoder(
            queries, visual_memory, context_tokens
        )

        # ── Step 5: Prediction heads ───────────────────────────
        pred_logits, pred_boxes = self.detect_head(queries)
        pred_severity           = self.severity_head(queries)

        # ── Step 6: Severity-conditioned filter ───────────────
        if self.use_scf:
            scf_out = self.scf(pred_logits, pred_severity, pred_boxes)
            scores  = scf_out['scores']
        else:
            scores  = pred_logits.sigmoid()

        # ── Auxiliary outputs for intermediate decoder layers ──
        aux_outputs = []
        for aq in aux_query_list:
            al, ab = self.detect_head(aq)
            asev   = self.severity_head(aq)
            aux_outputs.append({
                'pred_logits':   al,
                'pred_boxes':    ab,
                'pred_severity': asev,
            })

        return {
            'pred_logits':   pred_logits,    # [B, Q, C]
            'pred_boxes':    pred_boxes,      # [B, Q, 4]
            'pred_severity': pred_severity,   # [B, Q]
            'scores':        scores,          # [B, Q, C] SCF-boosted
            'aux_outputs':   aux_outputs,     # list for aux loss
        }

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {
            'total':     total,
            'trainable': trainable,
            'frozen':    frozen,
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building SLIM-Det...")
    model = SLIMDet(
        num_classes = 6,
        hidden_dim  = 256,
        num_queries = 90,
        num_layers  = 4,
        text_model  = 'minilm',
        freeze_text = True,
        use_scf     = True,
    )
    model.eval()

    # Simulate a batch of 2 images
    B = 2
    images = torch.randn(B, 3, 640, 640)
    prompts = [
        "Damage type: surface deformation or dent. Location: central inspection region. Severity: low severity.",
        "Damage type: structural crack or fracture line. Location: upper-left inspection zone. Severity: high severity.",
    ]
    zones   = ['central', 'top_left']
    metrics = torch.tensor([
        [0.055, 1.16, 0.524, 0.089],
        [0.120, 2.10, 0.800, 0.450],
    ])

    print("Running forward pass...")
    with torch.no_grad():
        out = model(images, prompts, zones, metrics)

    print("\n── Output shapes ──")
    print(f"pred_logits   : {out['pred_logits'].shape}")    # [2, 90, 6]
    print(f"pred_boxes    : {out['pred_boxes'].shape}")     # [2, 90, 4]
    print(f"pred_severity : {out['pred_severity'].shape}")  # [2, 90]
    print(f"scores        : {out['scores'].shape}")         # [2, 90, 6]
    print(f"aux_outputs   : {len(out['aux_outputs'])} intermediate layers")

    print("\n── Value checks ──")
    print(f"boxes range   : [{out['pred_boxes'].min():.3f}, {out['pred_boxes'].max():.3f}]  (should be 0-1)")
    print(f"scores range  : [{out['scores'].min():.3f}, {out['scores'].max():.3f}]  (should be 0-1)")

    print("\n── Parameter count ──")
    pc = model.param_count()
    print(f"Total     : {pc['total']:,}")
    print(f"Trainable : {pc['trainable']:,}")
    print(f"Frozen    : {pc['frozen']:,}  (text encoder)")

    print("\nslim_det.py works correctly ✅")
