import pytest

torch = pytest.importorskip("torch")

from core.attention.associative_attention import AssociativeIncrementalAttention


def test_ab_zero_bias_matches_baseline_output() -> None:
    torch.manual_seed(7)
    attn = AssociativeIncrementalAttention(d_model=8, nhead=2, dropout=0.0, concept_bias=0.0)
    attn.eval()

    x = torch.randn(2, 4, 8)
    concept_ids = torch.tensor([[1, 1, 2, -1], [3, 4, 3, 5]])
    zeros = torch.zeros(2, 4, 4)

    y_base, w_base = attn(x, return_attention=True)
    y_assoc, w_assoc = attn(
        x,
        concept_ids=concept_ids,
        ipa_affinity_bias=zeros,
        assoc_bias=zeros,
        return_attention=True,
    )

    assert torch.allclose(y_base, y_assoc, atol=1e-6)
    assert torch.allclose(w_base, w_assoc, atol=1e-6)


def test_ab_concept_match_reinforces_attention() -> None:
    torch.manual_seed(13)
    attn = AssociativeIncrementalAttention(d_model=8, nhead=2, dropout=0.0, concept_bias=2.5)
    attn.eval()

    x = torch.zeros(1, 3, 8)
    concept_ids = torch.tensor([[10, 10, 99]])

    _, w_no_bias = attn(x, return_attention=True)
    _, w_with_bias = attn(x, concept_ids=concept_ids, return_attention=True)

    assert w_with_bias[0, 0, 1] > w_no_bias[0, 0, 1]
    assert w_with_bias[0, 1, 0] > w_no_bias[0, 1, 0]


def test_encoder_flag_fallback_matches_baseline_path_shape() -> None:
    from pawp.model import PAWPEncoderModel

    torch.manual_seed(3)
    model = PAWPEncoderModel(
        word_vocab_size=32,
        ipa_vocab_size=32,
        root_vocab_size=16,
        lang_vocab_size=8,
        d_model=16,
        nhead=4,
        num_layers=2,
        num_classes=5,
        enable_associative_attention=False,
    )
    model.eval()

    bsz, seq = 2, 4
    wp = torch.randint(0, 31, (bsz, seq))
    ipa = torch.randint(0, 31, (bsz, seq))
    root = torch.randint(0, 15, (bsz, seq))
    lang = torch.randint(0, 7, (bsz, seq))
    cn = torch.randn(bsz, seq, 72)
    mask = torch.ones(bsz, seq, dtype=torch.bool)

    out = model(wp, ipa, root, lang, cn, attention_mask=mask)
    assert out.shape == (bsz, 5)
