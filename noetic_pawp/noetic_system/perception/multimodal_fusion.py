from __future__ import annotations

import torch


class MultimodalFusion:
    """Simple weighted fusion used by PerceptionController."""

    def __init__(self, semantic_weight: float = 0.5, visual_weight: float = 0.3, phonetic_weight: float = 0.2) -> None:
        self.semantic_weight = semantic_weight
        self.visual_weight = visual_weight
        self.phonetic_weight = phonetic_weight

    def fuse(self, semantic: torch.Tensor, visual: torch.Tensor, phonetic: torch.Tensor) -> torch.Tensor:
        phonetic_vec = phonetic.float().mean().expand_as(semantic)
        visual_resized = visual.float()
        if visual_resized.shape != semantic.shape:
            visual_resized = torch.nn.functional.pad(visual_resized, (0, max(0, semantic.numel() - visual.numel())))[: semantic.numel()]
            visual_resized = visual_resized.view_as(semantic)
        return (
            self.semantic_weight * semantic.float()
            + self.visual_weight * visual_resized
            + self.phonetic_weight * phonetic_vec
        )
