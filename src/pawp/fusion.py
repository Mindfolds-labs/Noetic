from __future__ import annotations

import torch
from torch import Tensor, nn


class PAWPFusion(nn.Module):
    """Multimodal fusion block for text/phonetic/root/language signals."""

    def __init__(
        self,
        text_vocab_size: int,
        phonetic_vocab_size: int,
        root_vocab_size: int,
        language_vocab_size: int,
        text_dim: int = 96,
        phonetic_dim: int = 64,
        root_dim: int = 48,
        language_dim: int = 16,
        model_dim: int = 128,
        num_heads: int = 4,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, text_dim, padding_idx=pad_idx)
        self.phonetic_embedding = nn.Embedding(phonetic_vocab_size, phonetic_dim, padding_idx=pad_idx)
        self.root_embedding = nn.Embedding(root_vocab_size, root_dim, padding_idx=pad_idx)
        self.language_embedding = nn.Embedding(language_vocab_size, language_dim, padding_idx=pad_idx)

        self.text_proj = nn.Linear(text_dim, model_dim)
        self.phonetic_proj = nn.Linear(phonetic_dim, model_dim)
        self.root_proj = nn.Linear(root_dim, model_dim)
        self.language_proj = nn.Linear(language_dim, model_dim)

        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, text_ids: Tensor, phonetic_ids: Tensor, root_ids: Tensor, language_ids: Tensor) -> Tensor:
        modalities = torch.stack(
            [
                self.text_proj(self.text_embedding(text_ids)),
                self.phonetic_proj(self.phonetic_embedding(phonetic_ids)),
                self.root_proj(self.root_embedding(root_ids)),
                self.language_proj(self.language_embedding(language_ids)),
            ],
            dim=2,
        )  # [B, T, 4, D]

        batch, seq_len, modal_count, dim = modalities.shape
        flattened = modalities.reshape(batch * seq_len, modal_count, dim)
        attended, _ = self.attention(flattened, flattened, flattened, need_weights=False)
        fused = attended.mean(dim=1).reshape(batch, seq_len, dim)
        return self.out(self.norm(fused))
