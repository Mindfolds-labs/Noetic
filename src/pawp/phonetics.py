from __future__ import annotations

from typing import Dict, List


class PhoneticAdapter:
    """Prototype G2P adapter. PT heuristics for v0.4."""

    _digraphs: Dict[str, str] = {
        "ch": "ʃ",
        "lh": "ʎ",
        "nh": "ɲ",
        "qu": "k",
        "gu": "g",
        "rr": "ʁ",
        "ss": "s",
    }
    _chars: Dict[str, str] = {
        "a": "a",
        "á": "a",
        "â": "ɐ",
        "ã": "ɐ̃",
        "e": "e",
        "é": "ɛ",
        "ê": "e",
        "i": "i",
        "í": "i",
        "o": "o",
        "ó": "ɔ",
        "ô": "o",
        "õ": "õ",
        "u": "u",
        "ú": "u",
        "ç": "s",
        "c": "k",
        "g": "g",
        "j": "ʒ",
        "r": "ɾ",
        "s": "s",
        "x": "ʃ",
        "z": "z",
    }

    def word_to_ipa_units(self, word: str, language: str = "pt") -> List[str]:
        if language != "pt":
            return list(word)

        units: List[str] = []
        i = 0
        while i < len(word):
            pair = word[i : i + 2]
            if len(pair) == 2 and pair in self._digraphs:
                units.append(self._digraphs[pair])
                i += 2
                continue
            units.append(self._chars.get(word[i], word[i]))
            i += 1
        return [u for u in units if u.strip()]
