import pytest

from noetic_pawp.concept_normalizer import ConceptNormalizer


@pytest.mark.parametrize(
    ("alias", "lang", "expected"),
    [
        ("hello", "en", "concept.greeting.hello"),
        ("olá", "pt", "concept.greeting.hello"),
        ("hola", "es", "concept.greeting.hello"),
        ("Cafe\u0301", "pt", "concept.food.coffee"),
        ("café", "pt", "concept.food.coffee"),
    ],
)
def test_concept_normalization_alias_coverage(alias: str, lang: str, expected: str) -> None:
    normalizer = ConceptNormalizer()
    assert normalizer.resolve_concept(alias, language=lang) == expected
