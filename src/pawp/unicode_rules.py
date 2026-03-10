from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

try:
    import regex as _regex  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _regex = None


SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "THAI": ((0x0E00, 0x0E7F),),
    "KHMER": ((0x1780, 0x17FF), (0x19E0, 0x19FF)),
    "MYANMAR": ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
    "CJK": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)),
    "HIRAGANA": ((0x3040, 0x309F),),
    "KATAKANA": ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F)),
}
NO_SPACE_SCRIPTS = tuple(SCRIPT_RANGES.keys())


class UnicodeNormalizer:
    def __init__(self, form: str = "NFC", lowercase: bool = True) -> None:
        self.form = form
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.form, text)
        return text.lower() if self.lowercase else text


class BasicUnicodeRules:
    @staticmethod
    def char_class(ch: str) -> str:
        category = unicodedata.category(ch)
        if category.startswith("L"):
            return "LETTER"
        if category.startswith("N"):
            return "NUMBER"
        if category.startswith("P"):
            return "PUNCT"
        if category.startswith("Z"):
            return "SPACE"
        return "OTHER"


def _script_from_codepoint(cp: int) -> str:
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= cp <= end for start, end in ranges):
            return script
    return "OTHER"


def _char_script(ch: str) -> str:
    cp = ord(ch)
    script = _script_from_codepoint(cp)
    if script != "OTHER":
        return script

    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "LATIN"
    if "ARABIC" in name:
        return "ARABIC"
    return "OTHER"


def _dominant_script(text: str) -> str:
    counts: Counter[str] = Counter(_char_script(ch) for ch in text if not ch.isspace())
    if not counts:
        return "OTHER"
    return max(counts.items(), key=lambda item: item[1])[0]


def _iter_graphemes(text: str) -> List[Tuple[str, int, int]]:
    if not text:
        return []
    if _regex is not None:
        return [(match.group(0), match.start(), match.end()) for match in _regex.finditer(r"\X", text)]

    graphemes: List[Tuple[str, int, int]] = []
    start = 0
    idx = 0
    while idx < len(text):
        idx += 1
        while idx < len(text):
            ch = text[idx]
            cp = ord(ch)
            if unicodedata.combining(ch) or unicodedata.category(ch).startswith("M"):
                idx += 1
                continue
            if cp in (0x200D,) or 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
                idx += 1
                continue
            break
        graphemes.append((text[start:idx], start, idx))
        start = idx
    return graphemes




def _grapheme_script(grapheme: str) -> str:
    for ch in grapheme:
        if ch.isspace() or unicodedata.category(ch).startswith("M"):
            continue
        return _char_script(ch)
    return "OTHER"


def _split_mixed_span(span: str, absolute_start: int) -> List[Tuple[str, int, int]]:
    output: List[Tuple[str, int, int]] = []
    graphemes = _iter_graphemes(span)
    chunk = ""
    chunk_start = 0
    chunk_mode = "standard"

    def flush() -> None:
        nonlocal chunk
        if not chunk:
            return
        if chunk_mode == "no_space":
            for grapheme, rel_start, rel_end in _iter_graphemes(chunk):
                if not grapheme.isspace():
                    output.append((grapheme, absolute_start + chunk_start + rel_start, absolute_start + chunk_start + rel_end))
        else:
            for token_match in re.finditer(r"\w+|[^\w\s]", chunk, flags=re.UNICODE):
                output.append((token_match.group(0), absolute_start + chunk_start + token_match.start(), absolute_start + chunk_start + token_match.end()))
        chunk = ""

    for grapheme, rel_start, _ in graphemes:
        mode = "no_space" if _grapheme_script(grapheme) in NO_SPACE_SCRIPTS else "standard"
        if not chunk:
            chunk = grapheme
            chunk_start = rel_start
            chunk_mode = mode
            continue
        if mode != chunk_mode:
            flush()
            chunk = grapheme
            chunk_start = rel_start
            chunk_mode = mode
            continue
        chunk += grapheme
    flush()
    return output


def _split_uax29_no_space_span(span: str, absolute_start: int) -> List[Tuple[str, int, int]]:
    if _regex is not None:
        return [
            (match.group(0), absolute_start + match.start(), absolute_start + match.end())
            for match in _regex.finditer(r"\X", span)
            if not match.group(0).isspace()
        ]
    return [
        (grapheme, absolute_start + rel_start, absolute_start + rel_end)
        for grapheme, rel_start, rel_end in _iter_graphemes(span)
        if not grapheme.isspace()
    ]

class PreTokenizer:
    _token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def split_words(self, text: str) -> List[str]:
        return [token for token, _, _ in self.split_words_with_offsets(text)]

    def split_words_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        output: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", text, flags=re.UNICODE):
            span = match.group(0)
            span_start = match.start()
            dominant_script = _dominant_script(span)
            if dominant_script in NO_SPACE_SCRIPTS:
                output.extend(_split_uax29_no_space_span(span, span_start))
            elif any(_char_script(ch) in NO_SPACE_SCRIPTS for ch in span if not ch.isspace()):
                output.extend(_split_mixed_span(span, span_start))
            else:
                for token_match in self._token_pattern.finditer(span):
                    output.append(
                        (
                            token_match.group(0),
                            span_start + token_match.start(),
                            span_start + token_match.end(),
                        )
                    )
        return output
