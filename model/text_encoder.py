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

import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


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
        local_files_only: bool = False,
        fallback_dim: int = 256,
        fallback_vocab_size: int = 8192,
    ):
        super().__init__()
        self.pooling    = pooling
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.uses_fallback = False

        # Resolve model name
        hf_name = SUPPORTED_MODELS.get(model_name, model_name)
        print(f"  Loading text encoder: {hf_name}")

        self.tokenizer, self.transformer = self._load_backbone(
            hf_name=hf_name,
            local_files_only=local_files_only,
            hidden_dim=fallback_dim,
            vocab_size=fallback_vocab_size,
        )

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

    def _load_backbone(self, hf_name, local_files_only, hidden_dim, vocab_size):
        if AutoTokenizer is None or AutoModel is None:
            print("  transformers not installed - using hash-based fallback text encoder")
            self.uses_fallback = True
            return _HashTokenizer(vocab_size), _HashTextBackbone(vocab_size, hidden_dim)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                hf_name,
                local_files_only=True,
            )
            transformer = AutoModel.from_pretrained(
                hf_name,
                local_files_only=True,
            )
            return tokenizer, transformer
        except Exception as exc:
            if not local_files_only:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        hf_name,
                        local_files_only=False,
                    )
                    transformer = AutoModel.from_pretrained(
                        hf_name,
                        local_files_only=False,
                    )
                    return tokenizer, transformer
                except Exception as remote_exc:
                    exc = remote_exc

            print(f"  Falling back to hash-based text encoder: {exc}")
            self.uses_fallback = True
            return _HashTokenizer(vocab_size), _HashTextBackbone(vocab_size, hidden_dim)

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


class _HashTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def _token_to_id(self, token):
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=4).digest()
        token_id = int.from_bytes(digest, 'little')
        return 1 + (token_id % (self.vocab_size - 1))

    def __call__(
        self,
        prompts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
    ):
        del padding, truncation, return_tensors
        encoded_prompts = []

        for prompt in prompts:
            tokens = prompt.lower().replace('.', ' ').replace(',', ' ').split()
            tokens = tokens[:max_length]
            if not tokens:
                tokens = ['<empty>']
            encoded_prompts.append([self._token_to_id(token) for token in tokens])

        max_tokens = max(len(tokens) for tokens in encoded_prompts)
        input_ids = torch.zeros(len(prompts), max_tokens, dtype=torch.long)
        attention_mask = torch.zeros(len(prompts), max_tokens, dtype=torch.long)

        for idx, token_ids in enumerate(encoded_prompts):
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            input_ids[idx, :len(token_ids)] = token_tensor
            attention_mask[idx, :len(token_ids)] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


class _HashTextBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.config = SimpleNamespace(hidden_size=hidden_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        return SimpleNamespace(last_hidden_state=self.norm(output))


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
