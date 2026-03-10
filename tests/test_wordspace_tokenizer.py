from noetic_pawp import PAWPTokenizer, TokenizerMode
from noetic_pawp.feature_flags import FeatureFlags
from noetic_pawp.wordspace_tokenizer import WordSpacePayload, WordSpaceTokenizer


def _baseline_tokenizer() -> PAWPTokenizer:
    tok = PAWPTokenizer()
    tok.fit_vocab(
        [
            "karaokê linguística pronúncia computação multimodal",
            "tokenização fonética hello mundo café noir",
        ],
        min_freq=1,
    )
    return tok


def _serialize(tokens):
    return [item.to_dict() for item in tokens]


def test_wordspace_disabled_matches_baseline_monolingual() -> None:
    base = _baseline_tokenizer()
    ws = WordSpaceTokenizer(tokenizer=base)

    baseline = _serialize(base.encode("pronúncia computação", language="pt"))
    wrapped = _serialize(ws.encode("pronúncia computação", language="pt"))

    assert wrapped == baseline


def test_wordspace_disabled_matches_baseline_multilingual() -> None:
    base = _baseline_tokenizer()
    ws = WordSpaceTokenizer(tokenizer=base)

    text = "hello mundo café noir"
    baseline = _serialize(base.encode(text, language="en"))
    wrapped = _serialize(ws.encode(text, language="en"))

    assert wrapped == baseline


def test_wordspace_payload_contract() -> None:
    base = _baseline_tokenizer()
    base.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=True)
    ws = WordSpaceTokenizer(tokenizer=base)

    payload = ws.encode("karaokê linguística", language="pt")

    assert isinstance(payload, WordSpacePayload)
    assert payload.token_ids
    assert payload.token_text
    assert payload.token_offsets
    assert payload.token_ipa_ids
    assert payload.concept_ids is None
    assert len(payload.token_ids) == len(payload.token_text) == len(payload.token_offsets) == len(payload.token_ipa_ids)


def test_wordspace_ipa_ids_are_deterministic_across_instances() -> None:
    base_a = _baseline_tokenizer()
    base_a.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=True)
    base_b = _baseline_tokenizer()
    base_b.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=True)

    ws_a = WordSpaceTokenizer(tokenizer=base_a)
    ws_b = WordSpaceTokenizer(tokenizer=base_b)

    payload_a = ws_a.encode("karaokê linguística", language="pt")
    payload_b = ws_b.encode("karaokê linguística", language="pt")

    assert isinstance(payload_a, WordSpacePayload)
    assert isinstance(payload_b, WordSpacePayload)
    assert payload_a.token_ipa_ids == payload_b.token_ipa_ids


def test_noetic_tokenizer_modes_contract() -> None:
    base = _baseline_tokenizer()

    text_tokens = base.encode("pronúncia", language="pt", mode=TokenizerMode.TEXT)
    audio_tokens = base.encode("pronúncia", language="pt", mode=TokenizerMode.AUDIO)
    multimodal_tokens = base.encode("pronúncia", language="pt", mode=TokenizerMode.MULTIMODAL)

    assert text_tokens and audio_tokens and multimodal_tokens
    assert all(not item.ipa_sequence and item.ipa_units == [] for item in text_tokens)
    assert any(item.ipa_sequence for item in audio_tokens)
    assert any(item.ipa_sequence for item in multimodal_tokens)


def test_mode_text_skips_cn_attachment() -> None:
    base = _baseline_tokenizer()
    out = base.encode("pronúncia", language="pt", mode=TokenizerMode.TEXT, attach_cn=True)
    assert out
    assert all(item.cn is None for item in out)


def test_invalid_mode_raises_for_noetic_tokenizer() -> None:
    base = _baseline_tokenizer()
    try:
        base.encode("pronúncia", language="pt", mode="invalid")
    except ValueError as exc:
        assert "Invalid tokenizer mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid mode")


def test_wordspace_offsets_stable_for_multiscript_text() -> None:
    base = _baseline_tokenizer()
    base.fit_vocab(["abcไทย 日本語かなカナ"], min_freq=1)
    base.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=False)
    ws = WordSpaceTokenizer(tokenizer=base)

    text = "abcไทย 日本語"
    payload = ws.encode(text, language="pt")

    assert isinstance(payload, WordSpacePayload)
    assert payload.token_text == ["abc", "ไ", "ท", "ย", "日", "本", "語"]
    assert payload.token_offsets == [(0, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10)]
    assert [text[s:e] for s, e in payload.token_offsets] == payload.token_text


def test_wordspace_offsets_stable_for_really_mixed_no_space_scripts() -> None:
    base = _baseline_tokenizer()
    base.fit_vocab(["alphabeticalไทย日本"], min_freq=1)
    base.config.feature_flags = FeatureFlags(enable_wordspace=True, enable_ipa_channel=False)
    ws = WordSpaceTokenizer(tokenizer=base)

    text = "alphabeticalไทย日本"
    payload = ws.encode(text, language="pt")

    assert isinstance(payload, WordSpacePayload)
    assert payload.token_text == ["alphabetical", "ไ", "ท", "ย", "日", "本"]
    assert [text[s:e] for s, e in payload.token_offsets] == payload.token_text


def test_noetic_vocab_serialization_contains_script_and_culture_tokens() -> None:
    base = _baseline_tokenizer()
    exported = base.to_dict()

    assert exported["special_tokens"]["script"] == [
        "[SCRIPT_LATIN]",
        "[SCRIPT_CJK]",
        "[SCRIPT_ARABIC]",
        "[SCRIPT_HIRAGANA]",
        "[SCRIPT_KATAKANA]",
        "[SCRIPT_THAI]",
        "[SCRIPT_KHMER]",
        "[SCRIPT_MYANMAR]",
        "[SCRIPT_OTHER]",
    ]
    assert exported["special_tokens"]["culture"] == ["[CULTURE_GLOBAL]", "[CULTURE_LOCAL]"]


def test_noetic_encode_prefixes_metadata_tokens_when_available() -> None:
    base = _baseline_tokenizer()
    out = base.encode("pronúncia", language="pt", mode=TokenizerMode.TEXT)

    assert len(out) >= 2
    assert out[0].wp_piece.startswith("[SCRIPT_")
    assert out[1].wp_piece == "[CULTURE_LOCAL]"
