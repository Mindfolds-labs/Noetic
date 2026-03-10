import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.config import PAWPToken
from noetic_pawp.noetic_system.perception.pawp_encoder import PAWPEncoder


class _StubTokenizer:
    def encode(self, text: str, language: str = "pt", attach_cn: bool = False):
        del text, language, attach_cn
        return [
            PAWPToken(wp_piece="ka", wp_id=11, ipa_sequence="ka"),
            PAWPToken(wp_piece="##ra", wp_id=12, ipa_units=["ɾ", "a"]),
        ]


def test_encode_text_accepts_minimal_pawp_token_contract() -> None:
    encoder = PAWPEncoder(tokenizer=_StubTokenizer())
    points = encoder.encode_text("kara")

    assert len(points) == 2
    assert [point.token for point in points] == ["ka", "##ra"]
    assert all(point.phonetic_vector.ndim == 1 for point in points)
    assert all(point.syntactic_features.shape == (4,) for point in points)
    assert PAWPEncoder.validate_mapping(points)
