from __future__ import annotations

from typing import List


class RootHeuristics:
    _pt_suffixes = ("mente", "ções", "ção", "dade", "ismo", "ista", "izar", "ável", "ível", "eiro")

    def split(self, word: str) -> List[str]:
        for suffix in self._pt_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return [word[: -len(suffix)], suffix]
        return [word]
