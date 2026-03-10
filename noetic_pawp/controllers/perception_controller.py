from __future__ import annotations

from .contracts import PerceptionOutput


class PerceptionController:
    def prepare(self, cn):
        return PerceptionOutput(cn=cn)
