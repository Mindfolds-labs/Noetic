from core.wordspace.wordspace_point import (
    WordSpacePoint,
    payload_from_wordspace_points,
    wordspace_points_from_payload,
)
from noetic_pawp.feature_flags import FeatureFlags
from noetic_pawp.wordspace_tokenizer import WordSpacePayload, WordSpaceTokenizer


def test_wordspace_point_shape_and_serialization() -> None:
    point = WordSpacePoint(
        text_vec=(1, 2),
        ipa_vec=(3,),
        context_vec=(4, 5, 6),
        assoc_vec=(),
        concept_id="concept:greeting",
        confidence=0.93,
    )

    assert point.shape == (2, 1, 3, 0)

    encoded = point.to_dict()
    restored = WordSpacePoint.from_dict(encoded)
    assert restored == point


def test_wordspace_converters_with_payload_optionals() -> None:
    payload = WordSpacePayload(
        token_ids=[10, 20],
        token_text=["olá", "mundo"],
        token_offsets=[(0, 3), (4, 9)],
        token_ipa_ids=[[1, 2], [3]],
        concept_ids=[None, "concept:world"],
    )

    points = wordspace_points_from_payload(payload)

    assert [p.concept_id for p in points] == [None, "concept:world"]
    assert [p.confidence for p in points] == [1.0, 1.0]
    assert points[0].text_vec == (10.0,)
    assert points[1].ipa_vec == (3.0,)

    rebuilt = payload_from_wordspace_points(
        points,
        token_text=payload.token_text,
        token_offsets=payload.token_offsets,
    )

    assert rebuilt.token_ids == payload.token_ids
    assert rebuilt.token_ipa_ids == payload.token_ipa_ids
    assert rebuilt.concept_ids == payload.concept_ids


def test_wordspace_from_enriched_tokenizer_output() -> None:
    tokenizer = WordSpaceTokenizer()
    tokenizer.tokenizer.fit_vocab(["karaokê linguística"], min_freq=1)
    tokenizer.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=True)

    payload = tokenizer.encode("karaokê linguística", language="pt")
    assert isinstance(payload, WordSpacePayload)

    points = wordspace_points_from_payload(payload)

    assert len(points) == len(payload.token_ids)
    assert all(len(point.text_vec) == 1 for point in points)
    expected_concepts = payload.concept_ids or [None] * len(payload.token_ids)
    assert [point.concept_id for point in points] == expected_concepts
