from __future__ import annotations

import torch


class AssociativeMemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, torch.Tensor] = {}

    def put(self, key: str, value: torch.Tensor) -> None:
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> torch.Tensor | None:
        return self._store.get(key)

    def similarity_search(self, query: torch.Tensor, topk: int = 3) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        q = torch.nn.functional.normalize(query.flatten(), dim=0)
        for k, v in self._store.items():
            score = torch.nn.functional.cosine_similarity(q, torch.nn.functional.normalize(v.flatten(), dim=0), dim=0)
            out.append((k, float(score.item())))
        return sorted(out, key=lambda x: x[1], reverse=True)[:topk]
