from pawp import PAWPTokenizer, compare_wordpiece_vs_pawp


def _build_tokenizer() -> PAWPTokenizer:
    tok = PAWPTokenizer()
    tok.train_vocab([
        "karaokê linguística pronúncia computação multimodal",
        "tokenização fonética",
    ])
    return tok


def test_encode_produces_tokens() -> None:
    tok = _build_tokenizer()
    out = tok.encode("karaokê", language="pt")
    assert out
    assert all(item.wp_piece for item in out)


def test_alignment_covers_all_ipa_units() -> None:
    tok = _build_tokenizer()
    analysis = tok.tokenize("linguística", language="pt")[0]
    ipa_units = tok.phonetic.word_to_ipa_units(analysis.normalized_word, language="pt")
    spans = tok.align_subwords_to_ipa(analysis.pieces, ipa_units)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(ipa_units)


def test_compare_structure() -> None:
    tok = _build_tokenizer()
    row = compare_wordpiece_vs_pawp(tok, "pronúncia", language="pt")
    assert "wordpiece_only" in row
    assert "pawp" in row
    assert row["pawp"]
