from pawp import PAWPTokenizer, TokenizerMode, compare_wordpiece_vs_pawp


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
    lexical = [item for item in out if not item.text.startswith("[")]
    assert out
    assert all(item.text for item in out)
    assert lexical
    assert all(item.ipa_representation for item in lexical)


def test_tokenizer_modes_contract() -> None:
    tok = _build_tokenizer()

    text_tokens = tok.encode("karaokê", language="pt", mode=TokenizerMode.TEXT)
    audio_tokens = tok.encode("karaokê", language="pt", mode=TokenizerMode.AUDIO)
    multimodal_tokens = tok.encode("karaokê", language="pt", mode=TokenizerMode.MULTIMODAL)

    assert text_tokens and audio_tokens and multimodal_tokens
    assert all(item.ipa_representation == "" for item in text_tokens)
    assert any(item.ipa_representation for item in audio_tokens)
    assert any(item.ipa_representation for item in multimodal_tokens)


def test_invalid_mode_raises() -> None:
    tok = _build_tokenizer()
    try:
        tok.encode("karaokê", language="pt", mode="invalid")
    except ValueError as exc:
        assert "Invalid tokenizer mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid mode")


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


def test_vocab_serialization_contains_script_and_culture_tokens() -> None:
    tok = _build_tokenizer()
    exported = tok.to_dict()

    assert exported["special_tokens"]["script"] == [
        "[SCRIPT_LATIN]",
        "[SCRIPT_CJK]",
        "[SCRIPT_ARABIC]",
        "[SCRIPT_OTHER]",
    ]
    assert exported["special_tokens"]["culture"] == ["[CULTURE_GLOBAL]", "[CULTURE_LOCAL]"]


def test_encode_prefixes_metadata_tokens_when_available() -> None:
    tok = _build_tokenizer()
    out = tok.encode("hello mundo", language="pt")

    assert len(out) >= 2
    assert out[0].text.startswith("[SCRIPT_")
    assert out[1].text == "[CULTURE_LOCAL]"
