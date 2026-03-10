from __future__ import annotations

import torch


class SparseAssociativeAttention:
    """Memory-efficient attention with optional sparse top-k activation."""

    def __init__(self, topk: int = 8) -> None:
        self.topk = topk

    def score(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(query, dim=-1)
        k = torch.nn.functional.normalize(keys, dim=-1)
        return q @ k.transpose(-1, -2)

    def retrieve(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sims = self.score(query, keys)
        k = min(self.topk, sims.shape[-1])
        top_vals, top_idx = torch.topk(sims, k=k, dim=-1)
        sparse_scores = torch.full_like(sims, float("-inf"))
        sparse_scores.scatter_(-1, top_idx, top_vals)
        attn = torch.softmax(sparse_scores, dim=-1)
        return attn @ values, attn
