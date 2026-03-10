from pawp import PAWPTokenizer, compare_wordpiece_vs_pawp


def _build_tokenizer() -> PAWPTokenizer:
    tok = PAWPTokenizer()
    tok.train_vocab([
        "karaokê linguística pronúncia computação multimodal",
        "tokenização fonética",
    ])
    return tok


def test_encode_produces_cognitive_tokens() -> None:
    tok = _build_tokenizer()
    out = tok.encode("karaokê", language="pt")
    assert out
    assert all(item.text for item in out)
    assert all(item.ipa_representation for item in out)


def test_phonetic_lru_cache_hits() -> None:
    tok = _build_tokenizer()
    tok.phonetic.word_to_ipa.cache_clear()
    tok.phonetic.word_to_ipa("linguística", language="pt")
    tok.phonetic.word_to_ipa("linguística", language="pt")
    assert tok.phonetic.word_to_ipa.cache_info().hits >= 1


def test_compare_structure() -> None:
    tok = _build_tokenizer()
    row = compare_wordpiece_vs_pawp(tok, "pronúncia", language="pt")
    assert "wordpiece_only" in row
    assert "pawp" in row
    assert row["pawp"]
