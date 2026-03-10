from __future__ import annotations

from torch import nn

from .contracts import BridgeState, MemoryQuery


class BridgeController:
    def __init__(self, to_prs: nn.Module) -> None:
        self._to_prs = to_prs

    def project(self, memory_query: MemoryQuery) -> BridgeState:
        prs = self._to_prs(memory_query.z_noetic)
        return BridgeState(z_noetic=memory_query.z_noetic, prs=prs)
