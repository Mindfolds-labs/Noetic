from noetic_pawp import PAWPTokenizer, compare_wordpiece_vs_pawp


def build_tokenizer() -> PAWPTokenizer:
    tok = PAWPTokenizer()
    tok.fit_vocab([
        "karaokê linguística pronúncia",
        "computação multimodal tokenização",
    ], min_freq=1)
    return tok


def test_encode_returns_tokens() -> None:
    tok = build_tokenizer()
    encoded = tok.encode("karaokê", language="pt")
    assert encoded
    assert encoded[0].wp_piece


def test_compare_has_required_keys() -> None:
    tok = build_tokenizer()
    row = compare_wordpiece_vs_pawp(tok, "pronúncia", language="pt")
    assert "wordpiece_only" in row
    assert "pawp" in row
    assert isinstance(row["pawp"], list)
