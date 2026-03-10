from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Concept:
    concept_id: str
    label: str
    source: str


class ConceptStore:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}

    def upsert(self, concept: Concept) -> None:
        self._concepts[concept.concept_id] = concept

    def get(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def all(self) -> list[Concept]:
        return list(self._concepts.values())
