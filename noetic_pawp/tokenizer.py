from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List

from .align import align_subwords_to_ipa
from .config import PAWPConfig, PAWPToken, TokenAnalysis
from .g2p import word_to_ipa


class PAWPTokenizer:
    def __init__(self, config: PAWPConfig | None = None) -> None:
        self.config = config or PAWPConfig()
        self.vocab: Dict[str, int] = {
            self.config.pad_token: 0,
            self.config.unk_token: 1,
            self.config.cls_token: 2,
            self.config.sep_token: 3,
            self.config.mask_token: 4,
        }

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.config.normalize_form, text)
        if self.config.lowercase:
            text = text.lower()
        return text

    def split_words(self, text: str) -> List[str]:
        return re.findall(r"[\wÀ-ÿ]+", text, flags=re.UNICODE)

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
        if word in self.vocab:
            return [word]

        pieces: List[str] = []
        cursor = 0
        while cursor < len(word):
            end = len(word)
            matched = None
            while end > cursor:
                chunk = word[cursor:end]
                piece = chunk if cursor == 0 else f"##{chunk}"
                if piece in self.vocab:
                    matched = piece
                    break
                end -= 1
            if matched is None:
                ch = word[cursor]
                matched = ch if cursor == 0 else f"##{ch}"
                if matched not in self.vocab:
                    matched = self.config.unk_token
            pieces.append(matched)
            cursor = end if end > cursor else cursor + 1
        return pieces

    def infer_root_segments(self, word: str) -> List[str]:
        suffixes = ["mente", "ção", "ções", "izar", "dade", "ismo"]
        for suf in suffixes:
            if word.endswith(suf) and len(word) > len(suf) + 2:
                return [word[: -len(suf)], suf]
        return [word]

    def tokenize(self, text: str, language: str = "pt") -> List[TokenAnalysis]:
        normalized = self.normalize(text)
        analyses: List[TokenAnalysis] = []
        for word in self.split_words(normalized):
            pieces = self.wordpiece_tokenize(word)
            ipa = word_to_ipa(word, language=language)
            analyses.append(
                TokenAnalysis(
                    original_word=word,
                    normalized_word=word,
                    pieces=pieces,
                    ipa=ipa,
                    root_segments=self.infer_root_segments(word),
                    used_phonetic_bias=self.config.use_phonetic_hints,
                )
            )
        return analyses

    def encode(self, text: str, language: str = "pt", attach_cn: bool = False) -> List[PAWPToken]:
        tokens: List[PAWPToken] = []
        for analysis in self.tokenize(text, language=language):
            ipa_units = list(analysis.ipa)
            spans = align_subwords_to_ipa(analysis.pieces, ipa_units)

            for idx, piece in enumerate(analysis.pieces):
                start, end = spans[idx]
                root_tag = analysis.root_segments[0] if analysis.root_segments else None
                cn = [0.0] * 72 if attach_cn else None
                tokens.append(
                    PAWPToken(
                        wp_piece=piece,
                        wp_id=self.vocab.get(piece, self.vocab[self.config.unk_token]),
                        ipa_units=ipa_units[start:end],
                        phoneme_spans=[(start, end)],
                        root_tag=root_tag,
                        lang=language,
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
