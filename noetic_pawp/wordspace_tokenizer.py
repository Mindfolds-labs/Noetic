from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .concept_normalizer import ConceptNormalizer

from .ipa_encoder import align_text_ipa, ipa_to_ids, text_to_ipa
from .config import PAWPConfig, PAWPToken
from .tokenizer import PAWPTokenizer


@dataclass
class WordSpacePayload:
    token_ids: List[int]
    token_text: List[str]
    token_offsets: List[Tuple[int, int]]
    token_ipa_ids: List[List[int]]
    concept_ids: Optional[List[Optional[str]]] = None


class WordSpaceTokenizer:
    """WordSpace wrapper that reuses the baseline PAWPTokenizer internals."""

    def __init__(
        self,
        tokenizer: Optional[PAWPTokenizer] = None,
        config: Optional[PAWPConfig] = None,
        concept_normalizer: Optional[ConceptNormalizer] = None,
    ) -> None:
        self.tokenizer = tokenizer or PAWPTokenizer(config=config)
        self.config = self.tokenizer.config
        self.concept_normalizer = concept_normalizer or ConceptNormalizer()

    def _piece_offsets(self, word_start: int, pieces: List[str], word_len: int) -> List[Tuple[int, int]]:
        if not pieces:
            return []
        piece_lengths = [max(1, len(piece.replace(self.config.wordpiece_prefix, ""))) for piece in pieces]
        total = sum(piece_lengths)
        cursor = word_start
        offsets: List[Tuple[int, int]] = []
        for idx, plen in enumerate(piece_lengths):
            if idx == len(piece_lengths) - 1:
                end = word_start + word_len
            else:
                delta = max(1, round((plen / total) * word_len))
                end = min(word_start + word_len, cursor + delta)
            offsets.append((cursor, end))
            cursor = end
        if offsets:
            offsets[-1] = (offsets[-1][0], word_start + word_len)
        return offsets

    def encode(
        self,
        text: str,
        language: str = "pt",
        attach_cn: bool = False,
    ) -> Union[List[PAWPToken], WordSpacePayload]:
        # Compatibility mode: preserve baseline PAWP behavior when WordSpace is disabled.
        if not self.config.feature_flags.enable_wordspace:
            return self.tokenizer.encode(text, language=language, attach_cn=attach_cn)

        normalized = self.tokenizer.normalize(text)
        analyses = self.tokenizer.tokenize(normalized, language=language)
        words = self.tokenizer.split_words_with_offsets(normalized)

        token_ids: List[int] = []
        token_text: List[str] = []
        token_offsets: List[Tuple[int, int]] = []
        token_ipa_ids: List[List[int]] = []
        concept_ids: Optional[List[Optional[str]]] = [] if self.config.feature_flags.enable_associative_memory else None

        for analysis, (_, start, end) in zip(analyses, words):
            ipa_sequence = text_to_ipa(analysis.original_word, language=language)
            # Mantemos a forma serial para alinhamento e a forma em unidades para ids,
            # espelhando o contrato PAWPToken (ipa_sequence + ipa_units).
            ipa_units = [ch for ch in ipa_sequence if not ch.isspace()]
            spans = align_text_ipa(analysis.pieces, ipa_sequence)
            piece_offsets = self._piece_offsets(start, analysis.pieces, end - start)

            for idx, piece in enumerate(analysis.pieces):
                token_ids.append(self.tokenizer.vocab.get(piece, self.tokenizer.vocab[self.config.unk_token]))
                token_text.append(piece)
                token_offsets.append(piece_offsets[idx])

                if self.config.feature_flags.enable_ipa_channel:
                    s, e = spans[idx]
                    token_ipa_ids.append(ipa_to_ids(ipa_units[s:e]))
                else:
                    token_ipa_ids.append([])

                if concept_ids is not None:
                    concept_ids.append(self.concept_normalizer.resolve_concept(token_text=piece, language=language))

        return WordSpacePayload(
            token_ids=token_ids,
            token_text=token_text,
            token_offsets=token_offsets,
            token_ipa_ids=token_ipa_ids,
            concept_ids=concept_ids,
        )
