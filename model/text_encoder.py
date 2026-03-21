# model/text_encoder.py
"""
TextEncoder — encodes damage description prompts into
hidden_dim embeddings using a lightweight pretrained
sentence transformer.

Default model: all-MiniLM-L6-v2 (22M params)
  — 90% of BERT quality at 6% of the size
  — perfect for structured damage descriptions

Supports 4 pooling strategies for ablation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ── Supported lightweight models ─────────────────────────────
SUPPORTED_MODELS = {
    'minilm':    'sentence-transformers/all-MiniLM-L6-v2',   # 22M  ← default
    'bge-small': 'BAAI/bge-small-en-v1.5',                   # 33M
    'distilbert':'distilbert-base-uncased',                   # 66M
    'gte-small': 'Alibaba-NLP/gte-small',                    # 33M
}


class TextEncoder(nn.Module):
    """
    Encodes text prompts → hidden_dim embeddings.

    Pipeline:
        prompt str
          → tokenizer
          → transformer (frozen or fine-tuned)
          → pooling (mean / cls / max)
          → linear projection → hidden_dim

    Args:
        model_name  : HuggingFace model id or key from SUPPORTED_MODELS
        hidden_dim  : output projection dimension (default 256)
        pooling     : 'mean' | 'cls' | 'max' (default 'mean')
        max_length  : max token length (default 128)
        freeze      : freeze transformer weights (default True)
        dropout     : dropout on projection (default 0.1)
    """

    def __init__(
        self,
        model_name:  str   = 'minilm',
        hidden_dim:  int   = 256,
        pooling:     str   = 'mean',
        max_length:  int   = 128,
        freeze:      bool  = True,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.pooling    = pooling
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # Resolve model name
        hf_name = SUPPORTED_MODELS.get(model_name, model_name)
        print(f"  Loading text encoder: {hf_name}")

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self.transformer = AutoModel.from_pretrained(hf_name)

        # Freeze transformer weights (save GPU memory + speed)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print(f"  Transformer frozen — only projection trains")

        # Get transformer output dim
        transformer_dim = self.transformer.config.hidden_size

        # Projection: transformer_dim → hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._init_proj()

    def _init_proj(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings to a single vector.

        mean: weighted average ignoring padding tokens (best for sentences)
        cls:  use [CLS] token embedding
        max:  max over non-padding tokens
        """
        if self.pooling == 'cls':
            return token_embeddings[:, 0, :]

        # Expand mask: [B, T] → [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling == 'mean':
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        if self.pooling == 'max':
            # Set padding to very negative value before max
            token_embeddings = token_embeddings.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )
            return token_embeddings.max(dim=1).values

        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, prompts: list) -> torch.Tensor:
        """
        Args:
            prompts: list of B prompt strings

        Returns:
            text_emb: [B, hidden_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        # Move to same device as projection layer
        device = next(self.proj.parameters()).device
        input_ids      = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Transformer forward
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.transformer.parameters())
        ):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Pool to single vector
        pooled = self._pool(
            outputs.last_hidden_state, attention_mask
        )   # [B, transformer_dim]

        # Project to hidden_dim
        text_emb = self.proj(pooled)    # [B, hidden_dim]

        return text_emb

    def encode_single(self, prompt: str) -> torch.Tensor:
        """Convenience: encode a single prompt string."""
        return self.forward([prompt])[0]


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading TextEncoder (downloading ~22MB model)...")
    encoder = TextEncoder(
        model_name = 'minilm',
        hidden_dim = 256,
        pooling    = 'mean',
        freeze     = True,
    )
    encoder.eval()

    test_prompts = [
        "Damage type: surface deformation or dent. Location: central inspection region. Severity: low severity, monitoring recommended.",
        "Damage type: structural crack or fracture line. Location: upper-left inspection zone. Severity: high severity, immediate attention required.",
        "Damage type: missing fastener or rivet head. Location: right edge boundary. Severity: moderate severity, maintenance required.",
        "Damage type: surface scratch or abrasion. Location: central inspection region. Severity: low severity, monitoring recommended.",
    ]

    with torch.no_grad():
        emb = encoder(test_prompts)

    print(f"\nInput prompts  : {len(test_prompts)}")
    print(f"Output shape   : {emb.shape}")           # [4, 256]
    print(f"Output mean    : {emb.mean():.4f}")
    print(f"Output std     : {emb.std():.4f}")

    # Check similarity — dent and scratch should be closer
    # than dent and crack (different damage mechanisms)
    from torch.nn.functional import cosine_similarity
    sim_dent_scratch = cosine_similarity(emb[0:1], emb[3:4]).item()
    sim_dent_crack   = cosine_similarity(emb[0:1], emb[1:2]).item()
    print(f"\nSimilarity dent↔scratch : {sim_dent_scratch:.3f}")
    print(f"Similarity dent↔crack   : {sim_dent_crack:.3f}")

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in encoder.parameters())
    print(f"\nTrainable params : {trainable:,} / {total:,}")
    print("\ntext_encoder.py works correctly")
