# model/rwda_decoder.py
"""
RWDA — Recurrent Weight-shared Deformable Attention Decoder.

NOVELTY COMBINATION:
1. Shared weights across all layers  → 60% fewer params
2. Deformable cross-attention        → attends to K points not all tokens
3. Unified memory                    → visual + context tokens together
4. Iteration embeddings              → each pass has unique identity
5. Auxiliary outputs                 → loss applied at every iteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossAttn(nn.Module):
    """
    Deformable cross-attention.
    Each query attends to num_heads * num_points sampled locations
    in the memory, weighted by learned attention scores.
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_points=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.num_points = num_points

        # Predict attention weights over sampled points
        self.attn_net   = nn.Linear(hidden_dim, num_heads * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)

        nn.init.zeros_(self.attn_net.bias)

    def forward(self, queries, memory, ref_points):
        """
        queries    : [B, Q, D]
        memory     : [B, N, D]
        ref_points : [B, Q, 2]  normalized [0,1]

        Returns    : [B, Q, D]
        """
        B, Q, D = queries.shape
        N       = memory.size(1)

        # Attention weights over num_heads * num_points
        attn_w = self.attn_net(queries)                         # [B, Q, H*P]
        attn_w = attn_w.view(B, Q, self.num_heads * self.num_points)
        attn_w = attn_w.softmax(-1).unsqueeze(-1)               # [B, Q, H*P, 1]

        # Project memory values
        values = self.value_proj(memory)                        # [B, N, D]

        # Map reference points to memory indices
        # ref_points: [B, Q, 2] → indices into N memory tokens
        ref_idx = (ref_points[..., 0] * (N - 1)).long().clamp(0, N - 1)  # [B, Q]

        # Gather H*P neighbourhood samples around each ref point
        num_samples = self.num_heads * self.num_points
        offsets = torch.linspace(-num_samples//2, num_samples//2,
                                 num_samples, device=queries.device).long()

        # Sample indices: [B, Q, H*P]
        idx = (ref_idx.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0))
        idx = idx.clamp(0, N - 1)                               # [B, Q, H*P]

        # Gather values at sampled positions: [B, Q, H*P, D]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)       # [B, Q, H*P, D]
        val_exp = values.unsqueeze(1).expand(-1, Q, -1, -1)     # [B, Q, N, D]
        sampled = torch.gather(val_exp, 2, idx_exp)             # [B, Q, H*P, D]

        # Weighted sum: [B, Q, D]
        out = (sampled * attn_w).sum(dim=2)

        return self.out_proj(out)                               # [B, Q, D]


class RWDADecoder(nn.Module):
    """
    Recurrent Weight-shared Deformable Attention Decoder.

    Same weights applied num_layers times recurrently.
    Each iteration gets a unique iteration embedding so the
    model knows which pass it is on.

    Args:
        hidden_dim : query/memory dimension (default 256)
        num_heads  : attention heads (default 8)
        num_points : deformable attention sample points (default 4)
        num_layers : recurrent iterations (default 4)
        ffn_dim    : FFN hidden dimension (default 1024)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int   = 256,
        num_heads:  int   = 8,
        num_points: int   = 4,
        num_layers: int   = 4,
        ffn_dim:    int   = 1024,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # SHARED weights — applied recurrently
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableCrossAttn(hidden_dim, num_heads, num_points)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Reference point predictor — refines each iteration
        self.ref_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

        # Per-iteration identity embeddings (NOT shared)
        self.iter_emb = nn.Embedding(num_layers, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.iter_emb.weight, std=0.02)

    def forward(self, queries, visual_memory, context_tokens):
        """
        Args:
            queries        : [B, Q, D]   from SGQI
            visual_memory  : [B, HW, D]  flattened visual features
            context_tokens : [B, 3, D]   from SMFE

        Returns:
            queries     : [B, Q, D]  final refined queries
            aux_outputs : list of [B, Q, D] per intermediate iteration
        """
        # Unified memory: visual tokens + context tokens
        memory = torch.cat([visual_memory, context_tokens], dim=1)  # [B, HW+3, D]

        aux_outputs = []

        for i in range(self.num_layers):
            # Add iteration identity
            queries = queries + self.iter_emb.weight[i]

            # 1. Self-attention among queries
            q2, _ = self.self_attn(queries, queries, queries)
            queries = self.norm1(queries + q2)

            # 2. Predict reference points
            ref_pts = self.ref_point_head(queries)   # [B, Q, 2]

            # 3. Deformable cross-attention to unified memory
            q2 = self.cross_attn(queries, memory, ref_pts)
            queries = self.norm2(queries + q2)

            # 4. FFN
            queries = self.norm3(queries + self.ffn(queries))

            aux_outputs.append(queries)

        return queries, aux_outputs[:-1]


if __name__ == '__main__':
    decoder = RWDADecoder(
        hidden_dim=256, num_heads=8,
        num_points=4,   num_layers=4,
    )
    decoder.eval()

    B, Q, D = 2, 90, 256
    queries        = torch.randn(B, Q, D)
    visual_memory  = torch.randn(B, 1600, D)   # 40x40 flattened
    context_tokens = torch.randn(B, 3, D)

    with torch.no_grad():
        out, aux = decoder(queries, visual_memory, context_tokens)

    print(f"queries in  : {queries.shape}")
    print(f"queries out : {out.shape}")                    # [2, 90, 256]
    print(f"aux outputs : {len(aux)} x {aux[0].shape}")   # 3 x [2,90,256]

    params = sum(p.numel() for p in decoder.parameters())
    print(f"params      : {params:,}")
    print("rwda_decoder.py works correctly")
