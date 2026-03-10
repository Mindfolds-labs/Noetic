from __future__ import annotations

from typing import Dict


class RootHeuristics:
    _suffixes: Dict[str, tuple[str, ...]] = {
        "pt": ("mente", "ções", "ção", "dade", "ismo", "ista", "izar", "ável", "ível", "eiro"),
        "en": ("ing", "ed", "ly", "ness", "ment", "tion", "s"),
    }

    def extract(self, word: str, language: str = "en") -> str:
        suffixes = self._suffixes.get(language, self._suffixes["en"])
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        return word

    def split(self, word: str, language: str = "en") -> list[str]:
        root = self.extract(word, language=language)
        return [root] if root == word else [root, word[len(root) :]]
