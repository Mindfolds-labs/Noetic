from noetic_pawp import ConceptNormalizer, PAWPTokenizer
from noetic_pawp.feature_flags import FeatureFlags
from noetic_pawp.wordspace_tokenizer import WordSpaceTokenizer


def test_resolve_concept_cross_lingual_aliases() -> None:
    normalizer = ConceptNormalizer()

    assert normalizer.resolve_concept("hello", language="en") == "concept.greeting.hello"
    assert normalizer.resolve_concept("olá", language="pt") == "concept.greeting.hello"
    assert normalizer.resolve_concept("hola", language="es") == "concept.greeting.hello"


def test_resolve_concept_unicode_variation_and_missing_fallback() -> None:
    normalizer = ConceptNormalizer()

    assert normalizer.resolve_concept("Cafe\u0301", language="pt") == "concept.food.coffee"
    assert normalizer.resolve_concept("desconhecido", language="pt") is None


def test_wordspace_attaches_concept_ids_from_normalizer() -> None:
    tokenizer = PAWPTokenizer()
    tokenizer.fit_vocab(["hello olá mundo"], min_freq=1)
    tokenizer.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_associative_memory=True)

    ws = WordSpaceTokenizer(tokenizer=tokenizer)
    payload = ws.encode("hello mundo", language="en")

    assert payload.concept_ids == ["concept.greeting.hello", None]
