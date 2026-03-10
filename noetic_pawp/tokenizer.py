from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

from .align import align_subwords_to_ipa
from .config import PAWPConfig, PAWPToken, TokenAnalysis, TokenizerMode
from .g2p import surface_to_ipa

SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "THAI": ((0x0E00, 0x0E7F),),
    "KHMER": ((0x1780, 0x17FF), (0x19E0, 0x19FF)),
    "MYANMAR": ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
    "CJK": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)),
    "HIRAGANA": ((0x3040, 0x309F),),
    "KATAKANA": ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F)),
}

NO_SPACE_SCRIPTS = tuple(SCRIPT_RANGES.keys())


def _char_script(ch: str) -> str:
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= cp <= end for start, end in ranges):
            return script

    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "LATIN"
    if "ARABIC" in name:
        return "ARABIC"
    if "CJK" in name:
        return "CJK"
    if "HIRAGANA" in name:
        return "HIRAGANA"
    if "KATAKANA" in name:
        return "KATAKANA"
    if "THAI" in name:
        return "THAI"
    if "KHMER" in name:
        return "KHMER"
    if "MYANMAR" in name:
        return "MYANMAR"
    return "OTHER"


def _dominant_script(text: str) -> str:
    counts: Counter[str] = Counter(_char_script(ch) for ch in text if not ch.isspace())
    if not counts:
        return "OTHER"
    return max(counts.items(), key=lambda item: item[1])[0]


def _iter_graphemes(text: str) -> List[Tuple[str, int, int]]:
    if not text:
        return []
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


def _split_no_space_span(span: str, absolute_start: int) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    for grapheme, rel_start, rel_end in _iter_graphemes(span):
        if grapheme.isspace():
            continue
        tokens.append((grapheme, absolute_start + rel_start, absolute_start + rel_end))
    return tokens




def _grapheme_script(grapheme: str) -> str:
    for ch in grapheme:
        if ch.isspace() or unicodedata.category(ch).startswith("M"):
            continue
        return _char_script(ch)
    return "OTHER"


def _split_mixed_span(span: str, absolute_start: int) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    graphemes = _iter_graphemes(span)
    current = ""
    current_start = 0
    current_mode = "standard"

    def flush() -> None:
        nonlocal current, current_start, current_mode
        if not current:
            return
        if current_mode == "no_space":
            tokens.extend(_split_no_space_span(current, absolute_start + current_start))
        else:
            tokens.extend(_split_standard_span(current, absolute_start + current_start))
        current = ""

    for grapheme, rel_start, rel_end in graphemes:
        script = _grapheme_script(grapheme)
        mode = "no_space" if script in NO_SPACE_SCRIPTS else "standard"
        if not current:
            current = grapheme
            current_start = rel_start
            current_mode = mode
            continue
        if mode != current_mode:
            flush()
            current = grapheme
            current_start = rel_start
            current_mode = mode
            continue
        current += grapheme
    flush()
    return tokens

def _split_standard_span(span: str, absolute_start: int) -> List[Tuple[str, int, int]]:
    return [
        (match.group(0), absolute_start + match.start(), absolute_start + match.end())
        for match in re.finditer(r"\w+|[^\w\s]", span, flags=re.UNICODE)
    ]


class PAWPTokenizer:
    def __init__(self, config: PAWPConfig | None = None) -> None:
        self.config = config or PAWPConfig()
        self.vocab: Dict[str, int] = {
            self.config.pad_token: 0,
            self.config.unk_token: 1,
            self.config.cls_token: 2,
            self.config.sep_token: 3,
            self.config.mask_token: 4,
            "[SCRIPT_LATIN]": 5,
            "[SCRIPT_CJK]": 6,
            "[SCRIPT_ARABIC]": 7,
            "[CULTURE_GLOBAL]": 8,
            "[CULTURE_LOCAL]": 9,
        }

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.config.normalize_form, text)
        if self.config.lowercase:
            text = text.lower()
        return text

    def split_words(self, text: str) -> List[str]:
        return [token for token, _, _ in self.split_words_with_offsets(text)]

    def split_words_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        out: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", text, flags=re.UNICODE):
            span = match.group(0)
            start = match.start()
            dominant_script = _dominant_script(span)
            if dominant_script in NO_SPACE_SCRIPTS:
                out.extend(_split_no_space_span(span, start))
            elif any(_char_script(ch) in NO_SPACE_SCRIPTS for ch in span if not ch.isspace()):
                out.extend(_split_mixed_span(span, start))
            else:
                out.extend(_split_standard_span(span, start))
        return out

    def fit_vocab(self, corpus: Iterable[str], min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for line in corpus:
            for word in self.split_words(self.normalize(line)):
                counter.update(self._candidate_pieces(word))

        next_id = max(self.vocab.values()) + 1
        for piece, freq in counter.items():
            if freq >= min_freq and piece not in self.vocab:
                self.vocab[piece] = next_id
                next_id += 1

    def train_vocab(self, corpus: Iterable[str], min_freq: int = 2) -> None:
        self.fit_vocab(corpus, min_freq=min_freq)

    def _candidate_pieces(self, word: str) -> List[str]:
        pieces = [word]
        for i in range(1, len(word)):
            pieces.append(word[:i])
            pieces.append(f"##{word[i:]}")
        for ch in word:
            pieces.append(ch)
            pieces.append(f"##{ch}")
        return pieces

    def wordpiece_tokenize(self, word: str) -> List[str]:
        vocab_tuple = tuple(sorted(self.vocab.items()))
        return list(self._wordpiece_tokenize_cached(word, vocab_tuple))

    @staticmethod
    @lru_cache(maxsize=8192)
    def _wordpiece_tokenize_cached(word: str, vocab_tuple: tuple) -> tuple[str, ...]:
        vocab = dict(vocab_tuple)
        if word in vocab:
            return (word,)

        unk_token = next((token for token, idx in vocab.items() if idx == 1), "[UNK]")

        pieces: List[str] = []
        cursor = 0
        while cursor < len(word):
            end = len(word)
            matched = None
            while end > cursor:
                chunk = word[cursor:end]
                piece = chunk if cursor == 0 else f"##{chunk}"
                if piece in vocab:
                    matched = piece
                    break
                end -= 1
            if matched is None:
                ch = word[cursor]
                matched = ch if cursor == 0 else f"##{ch}"
                if matched not in vocab:
                    matched = unk_token
            pieces.append(matched)
            cursor = end if end > cursor else cursor + 1
        return tuple(pieces)

    def infer_root_segments(self, word: str) -> List[str]:
        suffixes = ("mente", "ção", "ções", "izar", "dade", "ismo")
        return list(self._infer_root_segments_cached(word, suffixes))

    @staticmethod
    @lru_cache(maxsize=4096)
    def _infer_root_segments_cached(word: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
        for suf in suffixes:
            if word.endswith(suf) and len(word) > len(suf) + 2:
                return (word[: -len(suf)], suf)
        return (word,)

    def clear_caches(self) -> None:
        self._wordpiece_tokenize_cached.cache_clear()
        self._infer_root_segments_cached.cache_clear()

    def tokenize(
        self,
        text: str,
        language: str = "pt",
        mode: TokenizerMode | str | None = None,
    ) -> List[TokenAnalysis]:
        resolved_mode = TokenizerMode.from_value(mode or self.config.default_tokenizer_mode)
        normalized = self.normalize(text)
        analyses: List[TokenAnalysis] = []
        for word in self.split_words(normalized):
            pieces = self.wordpiece_tokenize(word)
            ipa = ""
            if resolved_mode in {TokenizerMode.AUDIO, TokenizerMode.MULTIMODAL}:
                ipa = surface_to_ipa(word, lang=language, backend=self.config.g2p_backend_priority)
            analyses.append(
                TokenAnalysis(
                    original_word=word,
                    normalized_word=word,
                    pieces=pieces,
                    ipa=ipa,
                    root_segments=self.infer_root_segments(word),
                    used_phonetic_bias=(
                        self.config.use_phonetic_hints
                        and resolved_mode in {TokenizerMode.AUDIO, TokenizerMode.MULTIMODAL}
                    ),
                )
            )
        return analyses

    def encode(
        self,
        text: str,
        language: str = "pt",
        attach_cn: bool = False,
        mode: TokenizerMode | str | None = None,
    ) -> List[PAWPToken]:
        resolved_mode = TokenizerMode.from_value(mode or self.config.default_tokenizer_mode)
        enable_audio = resolved_mode in {TokenizerMode.AUDIO, TokenizerMode.MULTIMODAL}
        tokens: List[PAWPToken] = []
        for analysis in self.tokenize(text, language=language, mode=resolved_mode):
            ipa_units = list(analysis.ipa) if enable_audio else []
            ipa_sequence = "".join(ipa_units)
            spans = align_subwords_to_ipa(analysis.pieces, ipa_units)
            script = _char_script(analysis.original_word[0]) if analysis.original_word else "OTHER"
            unicode_meta = {
                "nfc": unicodedata.normalize("NFC", analysis.original_word),
                "script": script,
                "codepoints": [ord(ch) for ch in analysis.original_word],
            }

            for idx, piece in enumerate(analysis.pieces):
                start, end = spans[idx]
                root_tag = analysis.root_segments[0] if analysis.root_segments else None
                cn = [0.0] * 72 if (attach_cn and enable_audio) else None
                tokens.append(
                    PAWPToken(
                        wp_piece=piece,
                        wp_id=self.vocab.get(piece, self.vocab[self.config.unk_token]),
                        ipa_units=ipa_units[start:end],
                        ipa_sequence=ipa_sequence[start:end],
                        phoneme_spans=[(start, end)],
                        root_tag=root_tag,
                        lang=language,
                        script=script,
                        unicode_meta=unicode_meta,
                        cn=cn,
                    )
                )
        return tokens


def compare_wordpiece_vs_pawp(tokenizer: PAWPTokenizer, word: str, language: str = "pt") -> Dict[str, Any]:
    analyses = tokenizer.tokenize(word, language=language)
    if not analyses:
        return {"word": word, "error": "no_analysis"}

    item = analyses[0]
    encoded = tokenizer.encode(word, language=language, attach_cn=False)
    return {
        "word": word,
        "normalized": item.normalized_word,
        "wordpiece_only": item.pieces,
        "ipa": item.ipa,
        "root_segments": item.root_segments,
        "pawp": [token.to_dict() for token in encoded],
    }


def review_alignment(tokenizer: PAWPTokenizer, words: List[str], language: str = "pt") -> List[Dict[str, Any]]:
    return [compare_wordpiece_vs_pawp(tokenizer, word, language=language) for word in words]
