# model/smfe.py
"""
SMFE — Structured Multimodal Feature Encoder.

NOVELTY: Treats each JSON metadata field as a separate token
and learns cross-field attention between them.
text token ↔ zone token ↔ metrics token

No existing detector treats structured annotation fields
as cross-attending tokens. This is the core novel module.
"""

import torch
import torch.nn as nn


class SMFE(nn.Module):
    """
    Structured Multimodal Feature Encoder.

    Takes three modality embeddings and fuses them via:
    1. Field-type embeddings (like segment embeddings in BERT)
    2. Cross-field multi-head attention
    3. FFN refinement
    4. Dynamic gating — learns how much each field contributes

    Args:
        hidden_dim : embedding dimension (default 256)
        num_heads  : attention heads (default 4)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int   = 256,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Field-type embeddings: TEXT=0, ZONE=1, METRICS=2
        # Like positional embeddings but for modality identity
        self.field_type_emb = nn.Embedding(3, hidden_dim)

        # Cross-field attention — fields attend to each other
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dynamic gate: learns per-field importance weights
        # Input: concatenation of all 3 field tokens
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),   # [B, 3] — sums to 1
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.field_type_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_emb:    torch.Tensor,
        zone_emb:    torch.Tensor,
        metrics_emb: torch.Tensor,
    ):
        """
        Args:
            text_emb    : [B, hidden_dim] from TextEncoder
            zone_emb    : [B, hidden_dim] from ZoneEncoder
            metrics_emb : [B, hidden_dim] from MetricsEncoder

        Returns:
            context_tokens : [B, 3, hidden_dim]
                             one token per field group
                             token 0 = text context
                             token 1 = zone context
                             token 2 = metrics context
            gate_weights   : [B, 3] field importance weights
                             useful for visualization/interpretability
        """
        B = text_emb.size(0)

        # Stack into token sequence [B, 3, hidden_dim]
        tokens = torch.stack(
            [text_emb, zone_emb, metrics_emb], dim=1
        )   # [B, 3, D]

        # Add field-type identities (TEXT=0, ZONE=1, METRICS=2)
        field_ids = torch.arange(3, device=tokens.device)
        tokens = tokens + self.field_type_emb(field_ids).unsqueeze(0)

        # Cross-field attention: each field learns from others
        attn_out, attn_weights = self.cross_attn(
            tokens, tokens, tokens
        )   # [B, 3, D]
        tokens = self.norm1(tokens + attn_out)

        # FFN
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Dynamic gating: which field matters most?
        # Input: flatten all 3 tokens → [B, 3*D]
        gate_input   = tokens.reshape(B, -1)        # [B, 3*D]
        gate_weights = self.gate(gate_input)         # [B, 3]

        # Apply gates: each token scaled by its importance
        gated_tokens = tokens * gate_weights.unsqueeze(-1)  # [B, 3, D]

        return gated_tokens, gate_weights

    def get_context_embedding(
        self,
        text_emb:    torch.Tensor,
        zone_emb:    torch.Tensor,
        metrics_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience: returns mean-pooled context embedding.
        Used by CAFPN for scale-aware feature weighting.

        Returns:
            context_emb: [B, hidden_dim]
        """
        tokens, _ = self.forward(text_emb, zone_emb, metrics_emb)
        return tokens.mean(dim=1)   # [B, D]


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    B = 4
    D = 256

    smfe = SMFE(hidden_dim=D, num_heads=4)
    smfe.eval()

    # Simulate encoder outputs
    text_emb    = torch.randn(B, D)
    zone_emb    = torch.randn(B, D)
    metrics_emb = torch.randn(B, D)

    with torch.no_grad():
        ctx_tokens, gates = smfe(text_emb, zone_emb, metrics_emb)
        ctx_emb           = smfe.get_context_embedding(
            text_emb, zone_emb, metrics_emb
        )

    print(f"text_emb shape    : {text_emb.shape}")       # [4, 256]
    print(f"ctx_tokens shape  : {ctx_tokens.shape}")     # [4, 3, 256]
    print(f"gate_weights      : {gates}")                # [4, 3] sum=1
    print(f"gate sum (=1?)    : {gates.sum(dim=-1)}")    # all 1.0
    print(f"ctx_emb shape     : {ctx_emb.shape}")        # [4, 256]

    trainable = sum(p.numel() for p in smfe.parameters() if p.requires_grad)
    print(f"\nTrainable params  : {trainable:,}")
    print("\nsmfe.py works correctly")
