from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticDriftMetrics:
    semantic_drift_index: float
    avg_cosine_drift: float
    knn_stability: float


class SemanticDriftMonitor:
    def __init__(self, knn_k: int = 2) -> None:
        self.knn_k = knn_k

    @staticmethod
    def _cosine_drift(current: Tensor, initial: Tensor) -> Tensor:
        sim = F.cosine_similarity(current, initial, dim=-1)
        return 1.0 - sim

    def _knn_indices(self, x: Tensor) -> Tensor:
        sims = F.normalize(x, p=2, dim=-1) @ F.normalize(x, p=2, dim=-1).transpose(0, 1)
        _, idx = torch.topk(sims, k=min(self.knn_k + 1, sims.size(1)), dim=-1)
        return idx[:, 1:]

    def measure(self, current_anchor_embeddings: Tensor, initial_anchor_embeddings: Tensor) -> SemanticDriftMetrics:
        if current_anchor_embeddings.shape != initial_anchor_embeddings.shape:
            raise ValueError("anchor tensors must have identical shape")
        cos_drift = self._cosine_drift(current_anchor_embeddings, initial_anchor_embeddings)
        cur_knn = self._knn_indices(current_anchor_embeddings)
        init_knn = self._knn_indices(initial_anchor_embeddings)
        knn_stability = (cur_knn == init_knn).float().mean().item()
        avg = float(cos_drift.mean().item())
        return SemanticDriftMetrics(
            semantic_drift_index=avg + (1.0 - knn_stability),
            avg_cosine_drift=avg,
            knn_stability=float(knn_stability),
        )
