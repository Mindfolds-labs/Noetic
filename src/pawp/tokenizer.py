from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from pawp.config import PAWPConfig, TokenizerMode
from pawp.phonetics import PhoneticAdapter
from pawp.roots import RootHeuristics
from pawp.unicode_rules import PreTokenizer, UnicodeNormalizer

CONTROL_TOKENS_ORDER = ("unk_token", "cls_token", "sep_token", "pad_token", "mask_token")
SCRIPT_SPECIAL_TOKENS = (
    "[SCRIPT_LATIN]",
    "[SCRIPT_CJK]",
    "[SCRIPT_ARABIC]",
    "[SCRIPT_OTHER]",
)
CULTURE_SPECIAL_TOKENS = ("[CULTURE_GLOBAL]", "[CULTURE_LOCAL]")


@dataclass
class CognitiveToken:
    text: str
    token_id: int
    ipa_representation: str
    root: str
    language_hint: str
    embedding: Optional[List[float]] = None

    @property
    def wp_piece(self) -> str:  # compatibility
        return self.text

    @property
    def wp_id(self) -> int:  # compatibility
        return self.token_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenAnalysis:
    original_word: str
    normalized_word: str
    pieces: List[str]
    ipa: str
    root: str


class WordPieceOnlyView:
    def __init__(self, prefix: str = "##") -> None:
        self.prefix = prefix

    def split_word(self, word: str) -> List[str]:
        if len(word) <= 4:
            return [word]
        chunks: List[str] = [word[:3]]
        rest = word[3:]
        while rest:
            chunks.append(f"{self.prefix}{rest[:3]}")
            rest = rest[3:]
        return chunks


class PAWPTokenizer:
    def __init__(self, config: Optional[PAWPConfig] = None) -> None:
        self.config = config or PAWPConfig()
        self.normalizer = UnicodeNormalizer(self.config.normalize_form, self.config.lowercase)
        self.pretokenizer = PreTokenizer()
        self.root_rules = RootHeuristics()
        self.phonetic = PhoneticAdapter()
        self.wordpiece = WordPieceOnlyView(prefix=self.config.wordpiece_prefix)
        self.vocab: Dict[str, int] = self._init_vocab()

    def _init_vocab(self) -> Dict[str, int]:
        ordered_tokens = [getattr(self.config, attr) for attr in CONTROL_TOKENS_ORDER]
        ordered_tokens.extend(SCRIPT_SPECIAL_TOKENS)
        ordered_tokens.extend(CULTURE_SPECIAL_TOKENS)
        return {token: idx for idx, token in enumerate(ordered_tokens)}

    def _resolve_script_token(self, text: str) -> Optional[str]:
        if not text:
            return None
        script_counts: Dict[str, int] = {}
        for ch in text:
            if ch.isspace():
                continue
            name = unicodedata.name(ch, "")
            script = "OTHER"
            if "LATIN" in name:
                script = "LATIN"
            elif "ARABIC" in name:
                script = "ARABIC"
            elif any(tag in name for tag in ("CJK", "HIRAGANA", "KATAKANA", "HAN")):
                script = "CJK"
            script_counts[script] = script_counts.get(script, 0) + 1
        dominant = max(script_counts.items(), key=lambda item: item[1])[0] if script_counts else "OTHER"
        token = f"[SCRIPT_{dominant}]"
        return token if token in self.vocab else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab": dict(sorted(self.vocab.items(), key=lambda item: item[1])),
            "special_tokens": {
                "control": [getattr(self.config, attr) for attr in CONTROL_TOKENS_ORDER],
                "script": list(SCRIPT_SPECIAL_TOKENS),
                "culture": list(CULTURE_SPECIAL_TOKENS),
            },
        }

    def _resolve_culture_token(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        token = "[CULTURE_LOCAL]" if language.lower() != self.config.default_language.lower() else "[CULTURE_GLOBAL]"
        return token if token in self.vocab else None

    def train_vocab(self, texts: Iterable[str], min_freq: Optional[int] = None) -> Dict[str, int]:
        counter: Dict[str, int] = {}
        for text in texts:
            norm = self.normalizer.normalize(text)
            for token in self.pretokenizer.split_words(norm):
                for piece in self.wordpiece.split_word(token):
                    counter[piece] = counter.get(piece, 0) + 1

        next_id = max(self.vocab.values()) + 1
        threshold = min_freq if min_freq is not None else self.config.min_frequency
        for piece, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < threshold or piece in self.vocab:
                continue
            self.vocab[piece] = next_id
            next_id += 1
            if len(self.vocab) >= self.config.max_vocab_size:
                break
        return self.vocab

    def fit_vocab(self, texts: Iterable[str], min_freq: int = 1) -> Dict[str, int]:
        return self.train_vocab(texts, min_freq=min_freq)

    def tokenize(
        self,
        text: str,
        language: Optional[str] = None,
        mode: TokenizerMode | str | None = None,
    ) -> List[TokenAnalysis]:
        language = language or self.config.default_language
        resolved_mode = TokenizerMode.from_value(mode or self.config.default_tokenizer_mode)
        normalized = self.normalizer.normalize(text)
        words = self.pretokenizer.split_words(normalized)
        analyses: List[TokenAnalysis] = []
        for word in words:
            pieces = self.wordpiece.split_word(word)
            ipa = ""
            if resolved_mode in {TokenizerMode.AUDIO, TokenizerMode.MULTIMODAL}:
                ipa = self.phonetic.word_to_ipa(word, language=language)
            root = self.root_rules.extract(word, language=language)
            analyses.append(TokenAnalysis(word, word, pieces, ipa, root))
        return analyses

    def encode(
        self,
        text: str,
        language: Optional[str] = None,
        attach_embedding: bool = False,
        mode: TokenizerMode | str | None = None,
    ) -> List[CognitiveToken]:
        language = language or self.config.default_language
        resolved_mode = TokenizerMode.from_value(mode or self.config.default_tokenizer_mode)
        enable_audio = resolved_mode in {TokenizerMode.AUDIO, TokenizerMode.MULTIMODAL}
        encoded: List[CognitiveToken] = []
        normalized = self.normalizer.normalize(text)
        for special in (self._resolve_script_token(normalized), self._resolve_culture_token(language)):
            if special is None:
                continue
            encoded.append(
                CognitiveToken(
                    text=special,
                    token_id=self.vocab[special],
                    ipa_representation="",
                    root="",
                    language_hint=language,
                    embedding=[] if attach_embedding else None,
                )
            )
        for analysis in self.tokenize(normalized, language=language, mode=resolved_mode):
            for piece in analysis.pieces:
                encoded.append(
                    CognitiveToken(
                        text=piece,
                        token_id=self.vocab.get(piece, self.vocab[self.config.unk_token]),
                        ipa_representation=analysis.ipa if enable_audio else "",
                        root=analysis.root,
                        language_hint=language,
                        embedding=[] if attach_embedding else None,
                    )
                )
        return encoded


PAWPToken = CognitiveToken


def compare_wordpiece_vs_pawp(tokenizer: PAWPTokenizer, word: str, language: str = "en") -> Dict[str, Any]:
    analyses = tokenizer.tokenize(word, language=language)
    if not analyses:
        return {"word": word, "error": "no_analysis"}

    item = analyses[0]
    encoded = tokenizer.encode(word, language=language)
    return {
        "word": word,
        "normalized": item.normalized_word,
        "wordpiece_only": item.pieces,
        "ipa": item.ipa,
        "root": item.root,
        "pawp": [token.to_dict() for token in encoded],
    }


def review_alignment(tokenizer: PAWPTokenizer, word: str, language: str = "en") -> Dict[str, Any]:
    return compare_wordpiece_vs_pawp(tokenizer, word, language)
