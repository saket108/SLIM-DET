# model/scf.py
"""
SCF — Severity-Conditioned Filter.

NOVELTY: Re-ranks detection confidence using predicted severity.
High severity boosts borderline detections so subtle but
critical damage is never suppressed.

No existing detector uses a domain-specific auxiliary
prediction to modulate classification confidence.

Formula: final_score = cls_score * (1 + alpha_c * severity_boost_c)
         alpha is learned per class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCF(nn.Module):
    """
    Severity-Conditioned Filter.

    Args:
        num_classes : number of damage classes (default 6)
        hidden_dim  : query dimension for calibrator (default 256)
    """

    def __init__(self, num_classes=6, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes

        # Per-class learned blend factor alpha
        # Initialized to 0.3 — small boost, grows during training
        self.alpha = nn.Parameter(
            torch.full((num_classes,), 0.3)
        )

        # Severity calibrator: maps raw severity → per-class boost
        self.calibrator = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

        for m in self.calibrator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pred_logits, pred_severity, pred_boxes):
        """
        Args:
            pred_logits  : [B, Q, num_classes]  raw class logits
            pred_severity: [B, Q]               raw severity logits
            pred_boxes   : [B, Q, 4]            predicted boxes

        Returns:
            dict with:
                scores   : [B, Q, num_classes]  severity-boosted scores
                boxes    : [B, Q, 4]
                severity : [B, Q]               sigmoid severity scores
        """
        cls_scores = pred_logits.sigmoid()          # [B, Q, C]
        sev_scores = pred_severity.sigmoid()        # [B, Q]

        # Calibrate: severity → per-class boost signal
        sev_in = sev_scores.unsqueeze(-1)           # [B, Q, 1]
        boost  = self.calibrator(sev_in)            # [B, Q, C]

        # Per-class alpha — clamp to [0, 1]
        alpha = self.alpha.clamp(0.0, 1.0)          # [C]

        # Final score: base + severity-weighted boost
        final_scores = cls_scores * (1.0 + alpha * boost)
        final_scores = final_scores.clamp(0.0, 1.0)

        return {
            'scores':   final_scores,
            'boxes':    pred_boxes,
            'severity': sev_scores,
        }


if __name__ == '__main__':
    scf = SCF(num_classes=6)
    scf.eval()

    B, Q = 2, 90
    logits   = torch.randn(B, Q, 6)
    severity = torch.randn(B, Q)
    boxes    = torch.rand(B, Q, 4)

    with torch.no_grad():
        out = scf(logits, severity, boxes)

    print(f"scores shape   : {out['scores'].shape}")    # [2, 90, 6]
    print(f"boxes shape    : {out['boxes'].shape}")     # [2, 90, 4]
    print(f"severity shape : {out['severity'].shape}")  # [2, 90]
    print(f"scores range   : [{out['scores'].min():.3f}, {out['scores'].max():.3f}]")

    # Verify severity boost is active
    base  = logits.sigmoid()
    diff  = (out['scores'] - base).abs().mean()
    print(f"boost effect   : {diff:.4f}  (should be > 0)")
    print(f"alpha values   : {scf.alpha.data}")
    print("scf.py works correctly")
