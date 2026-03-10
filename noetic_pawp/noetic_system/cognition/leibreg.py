from __future__ import annotations

import torch

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge


class LeibRegAdapter:
    def __init__(self, model: NoeticLeibregBridge | None = None) -> None:
        self.model = model or NoeticLeibregBridge()

    def infer(self, text: str, memory_keys: str | None = None) -> dict[str, torch.Tensor | None]:
        return self.model(text=text, memory_keys=memory_keys)
