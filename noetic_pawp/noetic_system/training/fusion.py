from __future__ import annotations

import torch


def fuse_batch(text_embeddings: torch.Tensor, memory_embeddings: torch.Tensor) -> torch.Tensor:
    return 0.7 * text_embeddings + 0.3 * memory_embeddings
