from __future__ import annotations

from .contracts import CognitionState, MemoryQuery


class MemoryController:
    def query(self, cognition: CognitionState) -> MemoryQuery:
        return MemoryQuery(z_noetic=cognition.z_noetic)
