import pytest
torch = pytest.importorskip("torch")
from noetic_pawp.interfaces import CognitiveOutput


def _make_valid() -> CognitiveOutput:
    return CognitiveOutput(
        text_embeddings=torch.randn(2, 5, 64),
        phonetic_features=torch.randn(2, 5, 32),
        concept_representation=torch.randn(2, 5, 128),
        confidence=torch.rand(2, 5),
        prediction_error=torch.rand(2, 5),
    )


def test_valid_output_passes_validate():
    assert _make_valid().validate() is True


def test_wrong_rank_fails_validate():
    out = _make_valid()
    out.text_embeddings = torch.randn(2, 64)  # rank 2 instead of 3
    assert out.validate() is False


def test_cognitive_output_is_dataclass():
    from dataclasses import fields
    assert len(fields(CognitiveOutput)) == 5
