from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


RELATION_KEYS = ("is_a", "sounds_like", "looks_like", "acts_like", "co_occurs_with", "context_of")
SLOT_KEYS = ("visual", "audio", "action", "context")


@dataclass(frozen=True)
class AssociationResult:
    relation: str
    target_concept_id: str
    score: float


class AssociativeMemory:
    """Simple in-memory associative memory backed by JSONL seed files."""

    def __init__(self, concepts_dir: str | Path = "data/concepts") -> None:
        self.concepts_dir = Path(concepts_dir)
        self._by_id: Dict[str, dict] = {}
        self._alias_index: Dict[tuple[str, str], List[str]] = {}
        self.reload()

    def reload(self) -> None:
        self._by_id.clear()
        self._alias_index.clear()
        for path in sorted(self.concepts_dir.glob("*.jsonl")):
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                concept = self._normalize_schema(json.loads(line))
                concept_id = concept["concept_id"]
                self._by_id[concept_id] = concept
                self._index_aliases(concept)

    @staticmethod
    def _normalize_schema(concept: dict) -> dict:
        if "concept_id" not in concept or not concept["concept_id"]:
            raise ValueError("concept record precisa de concept_id")

        aliases = concept.get("aliases") or {}
        ipa_variants = concept.get("ipa_variants") or []
        relations = concept.get("relations") or {}
        slots = concept.get("slots") or {}

        normalized_relations = {key: list(relations.get(key) or []) for key in RELATION_KEYS}
        normalized_slots = {key: slots.get(key) for key in SLOT_KEYS}

        return {
            "concept_id": str(concept["concept_id"]),
            "aliases": {str(lang): [str(item) for item in items] for lang, items in aliases.items()},
            "ipa_variants": [str(item) for item in ipa_variants],
            "relations": normalized_relations,
            "slots": normalized_slots,
        }

    def _index_aliases(self, concept: dict) -> None:
        concept_id = concept["concept_id"]
        for lang, values in concept["aliases"].items():
            for value in values:
                key = (lang.lower(), value.casefold())
                self._alias_index.setdefault(key, []).append(concept_id)

    def get_concept(self, concept_id: str) -> Optional[dict]:
        return self._by_id.get(concept_id)

    def search_by_alias(self, text: str, language: str) -> List[dict]:
        key = (language.lower(), text.casefold())
        concept_ids = self._alias_index.get(key, [])
        return [self._by_id[cid] for cid in concept_ids]

    def retrieve_associations(self, concept_id: str, top_k: int = 8) -> List[AssociationResult]:
        concept = self.get_concept(concept_id)
        if concept is None:
            return []

        results: List[AssociationResult] = []
        for relation in RELATION_KEYS:
            targets = concept["relations"].get(relation, [])
            weight = 2.0 if relation in {"is_a", "context_of"} else 1.0
            for target in targets:
                results.append(AssociationResult(relation=relation, target_concept_id=target, score=weight))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(0, int(top_k))]

    def concept_ids(self) -> Iterable[str]:
        return self._by_id.keys()
