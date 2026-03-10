from __future__ import annotations

from torch import nn

from .contracts import CognitionState, PerceptionOutput


class CognitiveController:
    def __init__(self, core: nn.Module) -> None:
        self._core = core

    def run(self, perception: PerceptionOutput, *, reset_state: bool = False) -> CognitionState:
        out = self._core(perception.cn, reset_state=reset_state)
        return CognitionState(
            z_noetic=out["z_noetic"],
            spikes=out["spikes"],
            membrane=out["membrane"],
            surprise=out["surprise"],
            surprise_trace=out["surprise_trace"],
        )
