import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.noetic_system.perception.pawp_encoder import PAWPEncoder


def test_pawp_points_are_multimodal_and_valid() -> None:
    encoder = PAWPEncoder()
    points = encoder.encode_text("noetic system")
    assert PAWPEncoder.validate_mapping(points)
    if points:
        first = points[0]
        assert first.syntactic_features.shape == (4,)
        assert first.semantic_vector.ndim == 1
        assert first.visual_vector.ndim == 1
