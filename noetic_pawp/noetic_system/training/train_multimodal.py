from __future__ import annotations

import torch

from .fusion import fuse_batch


class MultimodalTrainingLoop:
    def step(self, text_embeddings: torch.Tensor, memory_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = fuse_batch(text_embeddings, memory_embeddings)
        loss = torch.mean((fused - text_embeddings) ** 2)
        return {"fused": fused, "loss": loss}
