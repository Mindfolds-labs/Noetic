from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .concept_normalizer import ConceptNormalizer


@dataclass(frozen=True)
class RetrievalSample:
    query: str
    language: str
    expected_concept_id: Optional[str]


def rank_concepts(query: str, language: str, normalizer: Optional[ConceptNormalizer] = None) -> List[str]:
    resolver = normalizer or ConceptNormalizer()
    query_key = resolver._normalize_alias(query)

    scored: List[tuple[int, str]] = []
    for concept_id, aliases in resolver._concept_to_aliases.items():
        alias_keys = [resolver._normalize_alias(alias) for alias in aliases]
        best = 0
        if query_key in alias_keys:
            best = 3
        elif any(query_key and query_key in alias for alias in alias_keys):
            best = 2
        elif any(query_key and alias in query_key for alias in alias_keys):
            best = 1

        if best > 0:
            scored.append((best, concept_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [concept_id for _, concept_id in scored]

    exact = resolver.resolve_concept(query, language=language)
    if exact and exact not in ranked:
        ranked.insert(0, exact)
    return ranked


def retrieval_at_k(samples: Sequence[RetrievalSample], k: int, normalizer: Optional[ConceptNormalizer] = None) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not samples:
        return 0.0

    resolver = normalizer or ConceptNormalizer()
    hits = 0
    valid = 0
    for sample in samples:
        if sample.expected_concept_id is None:
            continue
        valid += 1
        top_k = rank_concepts(sample.query, language=sample.language, normalizer=resolver)[:k]
        if sample.expected_concept_id in top_k:
            hits += 1

    return hits / valid if valid else 0.0
