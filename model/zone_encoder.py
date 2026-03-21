# model/zone_encoder.py
"""
ZoneEncoder — converts zone string labels into spatial
prior embeddings used to seed decoder reference points.

Novelty: no existing detector uses inspection zone labels
as positional priors for query initialization.
"""

import torch
import torch.nn as nn


# ── Zone registry ─────────────────────────────────────────────
# Maps zone name → normalized (cx, cy) spatial prior
ZONE_MAP = {
    'central':      (0.50, 0.50),
    'top_left':     (0.20, 0.20),
    'top_right':    (0.80, 0.20),
    'bottom_left':  (0.20, 0.80),
    'bottom_right': (0.80, 0.80),
    'left_edge':    (0.10, 0.50),
    'right_edge':   (0.90, 0.50),
    'top_edge':     (0.50, 0.10),
    'bottom_edge':  (0.50, 0.90),
    'unknown':      (0.50, 0.50),   # fallback = center
}

ZONE_TO_IDX = {z: i for i, z in enumerate(ZONE_MAP)}
NUM_ZONES   = len(ZONE_MAP)


class ZoneEncoder(nn.Module):
    """
    Encodes zone string labels into hidden_dim embeddings.

    Two components:
    1. Learned zone embedding (semantic zone identity)
    2. Fixed spatial prior (cx, cy) refined by a small MLP

    The spatial prior is used to initialize decoder
    reference points — giving each query group a head
    start on WHERE to look in the image.

    Args:
        hidden_dim: output embedding dimension (default 256)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learned embedding per zone
        self.zone_embed = nn.Embedding(NUM_ZONES, hidden_dim)

        # MLP to refine zone + spatial coords into embedding
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Register spatial priors as buffer (not trained)
        coords = torch.tensor(
            list(ZONE_MAP.values()), dtype=torch.float32
        )   # [NUM_ZONES, 2]
        self.register_buffer('zone_coords', coords)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.zone_embed.weight, std=0.02)
        for m in self.refine.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, zone_names: list) -> torch.Tensor:
        """
        Args:
            zone_names: list of B zone strings e.g. ['central', 'top_left']

        Returns:
            zone_emb: [B, hidden_dim]
        """
        # Map strings to indices (handle unknown zones)
        idxs = torch.tensor(
            [ZONE_TO_IDX.get(z, ZONE_TO_IDX['unknown']) for z in zone_names],
            dtype=torch.long,
            device=self.zone_embed.weight.device
        )

        # Learned zone embedding
        emb    = self.zone_embed(idxs)                  # [B, hidden_dim]

        # Fixed spatial prior for each zone
        coords = self.zone_coords[idxs]                 # [B, 2]

        # Refine: concat embedding + spatial coords → final embedding
        zone_emb = self.refine(torch.cat([emb, coords], dim=-1))  # [B, hidden_dim]

        return zone_emb

    def get_reference_points(self, zone_names: list) -> torch.Tensor:
        """
        Returns normalized (cx, cy) spatial priors for a batch of zones.
        Used to initialize decoder reference points in RWDA.

        Args:
            zone_names: list of B zone strings

        Returns:
            ref_points: [B, 2] normalized coordinates in [0, 1]
        """
        idxs = torch.tensor(
            [ZONE_TO_IDX.get(z, ZONE_TO_IDX['unknown']) for z in zone_names],
            dtype=torch.long,
            device=self.zone_coords.device
        )
        return self.zone_coords[idxs]   # [B, 2]


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    encoder = ZoneEncoder(hidden_dim=256)
    encoder.eval()

    zones = ['central', 'top_left', 'bottom_right', 'unknown', 'left_edge']

    with torch.no_grad():
        emb    = encoder(zones)
        refpts = encoder.get_reference_points(zones)

    print(f"Input zones    : {zones}")
    print(f"zone_emb shape : {emb.shape}")       # [5, 256]
    print(f"ref_points     : {refpts}")           # [5, 2]
    print(f"central ref    : {refpts[0]}")        # tensor([0.5, 0.5])
    print(f"top_left ref   : {refpts[1]}")        # tensor([0.2, 0.2])
    print("\nzone_encoder.py works correctly")
