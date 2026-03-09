from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImaginationConfig:
    dim: int = 4
    with_confidence: bool = True


class Imagination(nn.Module):
    def __init__(self, config: ImaginationConfig | None = None) -> None:
        super().__init__()
        self.config = config or ImaginationConfig()
        self.hypothesis = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim),
            nn.Tanh(),
        )
        self.conf_head = nn.Sequential(nn.Linear(self.config.dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.hypothesis(x)
        out = {"hypothesis": h}
        if self.config.with_confidence:
            out["confidence"] = self.conf_head(h)
        return out
