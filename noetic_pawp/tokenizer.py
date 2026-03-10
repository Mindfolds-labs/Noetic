from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List

from .align import align_subwords_to_ipa
from .config import PAWPConfig, PAWPToken, TokenAnalysis, TokenizerMode
from .g2p import surface_to_ipa

NO_SPACE_SCRIPTS = ("THAI", "KHMER", "MYANMAR", "CJK", "HIRAGANA", "KATAKANA")


def _char_script(ch: str) -> str:
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
        tokens = re.findall(r"\S+", text, flags=re.UNICODE)
        out: List[str] = []
        for token in tokens:
            scripts = {_char_script(ch) for ch in token if not ch.isspace()}
            if scripts.intersection(NO_SPACE_SCRIPTS):
                out.extend([ch for ch in token if not ch.isspace()])
            else:
                out.extend(re.findall(r"[\wÀ-ÿ]+", token, flags=re.UNICODE))
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
                ipa = surface_to_ipa(word, lang=language)
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
