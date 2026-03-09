from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WordSpaceConfig:
    text_input_dim: int = 768
    image_input_dim: int = 72
    memory_input_dim: int = 256
    target_dim: int = 4


class WordSpace(nn.Module):
    def __init__(self, config: WordSpaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or WordSpaceConfig()
        self.text_head = nn.Linear(self.config.text_input_dim, self.config.target_dim)
        self.image_head = nn.Linear(self.config.image_input_dim, self.config.target_dim)
        self.memory_head = nn.Linear(self.config.memory_input_dim, self.config.target_dim)

    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.float()

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_head(self._normalize_shape(x))

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_head(self._normalize_shape(x))

    def project_memory(self, x: torch.Tensor) -> torch.Tensor:
        return self.memory_head(self._normalize_shape(x))
