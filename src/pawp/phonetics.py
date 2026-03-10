from __future__ import annotations

from functools import lru_cache
from typing import Dict, List


class PhoneticAdapter:
    """Lightweight heuristic grapheme-to-IPA adapter with LRU caching."""

    _digraphs: Dict[str, str] = {
        "ch": "tʃ",
        "sh": "ʃ",
        "th": "θ",
        "ph": "f",
        "ng": "ŋ",
        "lh": "ʎ",
        "nh": "ɲ",
        "rr": "ʁ",
        "ss": "s",
        "qu": "k",
        "gu": "g",
    }

    _chars: Dict[str, str] = {
        "a": "a",
        "á": "a",
        "â": "ɐ",
        "ã": "ɐ̃",
        "b": "b",
        "c": "k",
        "ç": "s",
        "d": "d",
        "e": "e",
        "é": "ɛ",
        "ê": "e",
        "f": "f",
        "g": "g",
        "h": "h",
        "i": "i",
        "í": "i",
        "j": "ʒ",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "o",
        "ó": "ɔ",
        "ô": "o",
        "õ": "õ",
        "p": "p",
        "q": "k",
        "r": "ɾ",
        "s": "s",
        "t": "t",
        "u": "u",
        "ú": "u",
        "v": "v",
        "w": "w",
        "x": "ʃ",
        "y": "j",
        "z": "z",
    }

    def __init__(self, cache_size: int = 8192) -> None:
        self._cache_size = cache_size

    @lru_cache(maxsize=8192)
    def word_to_ipa(self, word: str, language: str = "en") -> str:
        if not word:
            return ""
        if language not in {"en", "pt"}:
            return word

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
        return "".join(u for u in units if u.strip())

    def word_to_ipa_units(self, word: str, language: str = "en") -> List[str]:
        return list(self.word_to_ipa(word, language=language))
