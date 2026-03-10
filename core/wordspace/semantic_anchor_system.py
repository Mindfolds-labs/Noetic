from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if torch is not None:
    from torch import Tensor
else:
    Tensor = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class AnchorSpec:
    concept_id: str
    is_anchor: bool


class SemanticAnchorSystem:
    """Manages fixed semantic anchors and slow/frozen identity updates."""

    def __init__(self, concepts_path: str | Path) -> None:
        self.concepts_path = Path(concepts_path)
        self.anchor_specs = self._load_specs(self.concepts_path)
        self.anchor_ids = {spec.concept_id for spec in self.anchor_specs.values() if spec.is_anchor}

    @staticmethod
    def _load_specs(path: Path) -> Dict[str, AnchorSpec]:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        specs: Dict[str, AnchorSpec] = {}
        for rec in records:
            concept_id = str(rec.get("concept_id", "")).strip()
            if not concept_id:
                continue
            specs[concept_id] = AnchorSpec(concept_id=concept_id, is_anchor=bool(rec.get("is_anchor", False)))
        return specs

    def is_anchor(self, concept_id: str | None) -> bool:
        return bool(concept_id and concept_id in self.anchor_ids)

    def apply_identity_update(
        self,
        concept_id: str,
        core_identity: Tensor,
        delta: Tensor,
        base_lr: float = 1.0,
        anchor_lr_scale: float = 0.05,
    ) -> Tensor:
        lr = base_lr * (anchor_lr_scale if self.is_anchor(concept_id) else 1.0)
        return core_identity + (lr * delta)

    def anchor_mask(self, concept_ids: Iterable[str]) -> Tensor:
        if torch is None:
            raise RuntimeError("PyTorch is required for anchor_mask")
        values = [1.0 if cid in self.anchor_ids else 0.0 for cid in concept_ids]
        return torch.tensor(values, dtype=torch.float32)
