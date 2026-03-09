from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from .align import align_subwords_to_ipa
from .g2p import word_to_ipa

IPA_UNK_TOKEN = "[IPA_UNK]"

# Fixed IPA inventory to guarantee deterministic ids across runs.
_IPA_TOKENS = [
    IPA_UNK_TOKEN,
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɐ",
    "ɑ",
    "ɔ",
    "ə",
    "ɛ",
    "ɜ",
    "ɡ",
    "ɪ",
    "ɲ",
    "ɹ",
    "ɾ",
    "ʁ",
    "ʃ",
    "ʊ",
    "ʎ",
    "ʒ",
    "θ",
    "ˈ",
    "ˌ",
]

IPA_TOKEN_TO_ID: Dict[str, int] = {token: idx for idx, token in enumerate(_IPA_TOKENS)}


def _ipa_units(sequence: str) -> List[str]:
    cleaned = sequence.strip().strip("/").strip("[]")
    return [ch for ch in cleaned if not ch.isspace()]


def _looks_like_ipa(text: str) -> bool:
    if "/" in text or "[" in text or "]" in text:
        return True

    units = _ipa_units(text)
    if not units:
        return False

    ipa_specific = {"æ", "ð", "ŋ", "ɐ", "ɑ", "ɔ", "ə", "ɛ", "ɜ", "ɡ", "ɪ", "ɲ", "ɹ", "ɾ", "ʁ", "ʃ", "ʊ", "ʎ", "ʒ", "θ", "ˈ", "ˌ"}
    return any(unit in ipa_specific for unit in units)


def text_to_ipa(text: str, language: str = "pt") -> str:
    normalized = unicodedata.normalize("NFC", text).lower().strip()
    if not normalized:
        return ""
    if _looks_like_ipa(normalized):
        return "".join(_ipa_units(normalized))

    words = re.findall(r"[\wÀ-ÿ]+", normalized, flags=re.UNICODE)
    ipa_words = [word_to_ipa(word, language=language) for word in words]
    return " ".join(ipa_words)


def ipa_to_ids(ipa_sequence: str | Sequence[str]) -> List[int]:
    units = list(ipa_sequence) if not isinstance(ipa_sequence, str) else _ipa_units(ipa_sequence)
    unk_id = IPA_TOKEN_TO_ID[IPA_UNK_TOKEN]
    return [IPA_TOKEN_TO_ID.get(unit, unk_id) for unit in units]


def align_text_ipa(tokens: Sequence[str], ipa_sequence: str | Sequence[str]) -> List[Tuple[int, int]]:
    units = list(ipa_sequence) if not isinstance(ipa_sequence, str) else _ipa_units(ipa_sequence)
    return align_subwords_to_ipa(tokens, units)
