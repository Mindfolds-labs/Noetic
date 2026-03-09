from __future__ import annotations

import hashlib

import torch


class AssociativeMemory:
    """Deterministic embedding lookup by concept key."""

    def __init__(self, embedding_dim: int = 256) -> None:
        self.embedding_dim = embedding_dim

    def retrieve(self, key: str) -> torch.Tensor:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        vals = [digest[i % len(digest)] / 255.0 for i in range(self.embedding_dim)]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
