from __future__ import annotations

import torch


class RIVEEncoder:
    """Minimal tensor encoder returning 72 coefficients per image."""

    def __init__(self, output_dim: int = 72) -> None:
        self.output_dim = output_dim

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        flat = image.float().reshape(image.shape[0], -1)
        if flat.shape[1] >= self.output_dim:
            return flat[:, : self.output_dim]
        pad = torch.zeros(flat.shape[0], self.output_dim - flat.shape[1], device=flat.device)
        return torch.cat([flat, pad], dim=1)
