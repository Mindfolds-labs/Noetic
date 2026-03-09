from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pawp.config import PAWPConfig
from pawp.phonetics import PhoneticAdapter
from pawp.roots import RootHeuristics
from pawp.unicode_rules import PreTokenizer, UnicodeNormalizer


@dataclass
class TokenAnalysis:
    original_word: str
    normalized_word: str
    pieces: List[str]
    ipa: Optional[str] = None
    root_segments: List[str] = field(default_factory=list)


@dataclass
class PAWPToken:
    wp_piece: str
    wp_id: int
    ipa_units: List[str] = field(default_factory=list)
    phoneme_spans: List[Tuple[int, int]] = field(default_factory=list)
    root_tag: Optional[str] = None
    lang: Optional[str] = None
    cn: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WordPieceOnlyView:
    def split_word(self, word: str) -> List[str]:
        if len(word) <= 4:
            return [word]
        chunks: List[str] = [word[:3]]
        rest = word[3:]
        while rest:
            chunks.append(f"##{rest[:3]}")
            rest = rest[3:]
        return chunks


class PAWPTokenizer:
    def __init__(self, config: Optional[PAWPConfig] = None) -> None:
        self.config = config or PAWPConfig()
        self.normalizer = UnicodeNormalizer(self.config.normalize_form, self.config.lowercase)
        self.pretokenizer = PreTokenizer()
        self.root_rules = RootHeuristics()
        self.phonetic = PhoneticAdapter()
        self.wordpiece = WordPieceOnlyView()
        self.vocab: Dict[str, int] = {
            self.config.unk_token: 0,
            self.config.cls_token: 1,
            self.config.sep_token: 2,
            self.config.pad_token: 3,
            self.config.mask_token: 4,
        }

    def train_vocab(self, texts: Iterable[str]) -> Dict[str, int]:
        counter: Dict[str, int] = {}
        for text in texts:
            norm = self.normalizer.normalize(text)
            for token in self.pretokenizer.split_words(norm):
                for piece in self.wordpiece.split_word(token):
                    counter[piece] = counter.get(piece, 0) + 1

        next_id = max(self.vocab.values()) + 1
        for piece, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if piece in self.vocab:
                continue
            self.vocab[piece] = next_id
            next_id += 1
            if len(self.vocab) >= self.config.max_vocab_size:
                break
        return self.vocab

    def tokenize(self, text: str, language: str = "pt") -> List[TokenAnalysis]:
        normalized = self.normalizer.normalize(text)
        words = self.pretokenizer.split_words(normalized)
        analyses: List[TokenAnalysis] = []
        for word in words:
            pieces = self.wordpiece.split_word(word)
            ipa_units = self.phonetic.word_to_ipa_units(word, language=language)
            analyses.append(
                TokenAnalysis(
                    original_word=word,
                    normalized_word=word,
                    pieces=pieces,
                    ipa="".join(ipa_units),
                    root_segments=self.root_rules.split(word),
                )
            )
        return analyses

    def align_subwords_to_ipa(self, pieces: List[str], ipa_units: List[str]) -> List[Tuple[int, int]]:
        if not pieces:
            return []
        if not ipa_units:
            return [(0, 0) for _ in pieces]

        lengths = [max(1, len(p.replace(self.config.wordpiece_prefix, ""))) for p in pieces]
        total_len = sum(lengths)
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for idx, piece_len in enumerate(lengths):
            if idx == len(lengths) - 1:
                end = len(ipa_units)
            else:
                expected = round((piece_len / total_len) * len(ipa_units))
                end = max(cursor + 1, min(len(ipa_units), cursor + expected))
            spans.append((cursor, end))
            cursor = end
        if spans[-1][1] != len(ipa_units):
            spans[-1] = (spans[-1][0], len(ipa_units))
        return spans

    def encode(self, text: str, language: str = "pt", attach_cn: bool = False) -> List[PAWPToken]:
        encoded: List[PAWPToken] = []
        for analysis in self.tokenize(text, language=language):
            ipa_units = self.phonetic.word_to_ipa_units(analysis.normalized_word, language=language)
            spans = self.align_subwords_to_ipa(analysis.pieces, ipa_units)
            root_tag = analysis.root_segments[0] if analysis.root_segments else None
            for piece, span in zip(analysis.pieces, spans):
                start, end = span
                encoded.append(
                    PAWPToken(
                        wp_piece=piece,
                        wp_id=self.vocab.get(piece, self.vocab[self.config.unk_token]),
                        ipa_units=ipa_units[start:end],
                        phoneme_spans=[span],
                        root_tag=root_tag,
                        lang=language,
                        cn=[0.0] * 72 if attach_cn else None,
                    )
                )
        return encoded


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
