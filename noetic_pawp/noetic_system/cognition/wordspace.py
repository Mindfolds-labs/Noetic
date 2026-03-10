from __future__ import annotations

import torch

from pyfolds.leibreg.wordspace import WordSpace, WordSpaceConfig


class WordSpaceAdapter:
    def __init__(self, config: WordSpaceConfig | None = None) -> None:
        self.model = WordSpace(config or WordSpaceConfig())

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.project_text(x)

    def retrieve(self, query: torch.Tensor, candidates: torch.Tensor, k: int = 5):
        return self.model.retrieve_topk(query, candidates, k=k)
