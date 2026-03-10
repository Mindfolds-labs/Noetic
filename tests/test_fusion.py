import pytest

torch = pytest.importorskip("torch")

from pawp.fusion import PAWPFusion


def test_fusion_outputs_fused_cognitive_embedding() -> None:
    fusion = PAWPFusion(
        text_vocab_size=100,
        phonetic_vocab_size=60,
        root_vocab_size=40,
        language_vocab_size=8,
        model_dim=32,
        num_heads=4,
    )
    text_ids = torch.randint(0, 99, (2, 5))
    phonetic_ids = torch.randint(0, 59, (2, 5))
    root_ids = torch.randint(0, 39, (2, 5))
    language_ids = torch.randint(0, 7, (2, 5))

    out = fusion(text_ids, phonetic_ids, root_ids, language_ids)
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()
