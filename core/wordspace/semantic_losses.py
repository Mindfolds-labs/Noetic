from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _cosine_gram(x: Tensor) -> Tensor:
    xn = F.normalize(x, p=2, dim=-1)
    return xn @ xn.transpose(0, 1)


class SemanticStabilityLoss(nn.Module):
    """Penalizes relative anchor geometry drift from initialization."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, current_anchor_embeddings: Tensor, initial_anchor_embeddings: Tensor) -> Tensor:
        if current_anchor_embeddings.shape != initial_anchor_embeddings.shape:
            raise ValueError("anchor tensors must have identical shape")
        if current_anchor_embeddings.dim() != 2:
            raise ValueError("anchor tensors must be rank-2 [num_anchors, dim]")
        cur_gram = _cosine_gram(current_anchor_embeddings)
        init_gram = _cosine_gram(initial_anchor_embeddings)
        loss = (cur_gram - init_gram).pow(2)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
