from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class REGCoreConfig:
    dim: int = 4
    hidden_mult: int = 2


class REGCore(nn.Module):
    def __init__(self, config: REGCoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or REGCoreConfig()
        hidden = self.config.dim * max(self.config.hidden_mult, 1)
        self.net = nn.Sequential(
            nn.Linear(self.config.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.config.dim),
        )

    def forward(self, x):
        return self.net(x)
