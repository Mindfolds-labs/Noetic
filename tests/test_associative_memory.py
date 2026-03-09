from __future__ import annotations

from core.memory.associative_memory import AssociativeMemory


def test_associative_memory_load_and_lookup() -> None:
    memory = AssociativeMemory("data/concepts")

    concept = memory.get_concept("concept.greeting.hello")
    assert concept is not None
    assert concept["slots"]["context"] == "meeting_start"

    by_alias = memory.search_by_alias("olá", language="pt")
    assert by_alias
    assert by_alias[0]["concept_id"] == "concept.greeting.hello"


def test_associative_memory_retrieve_associations_top_k() -> None:
    memory = AssociativeMemory("data/concepts")
    associations = memory.retrieve_associations("concept.food.coffee", top_k=2)

    assert len(associations) == 2
    assert associations[0].score >= associations[1].score
