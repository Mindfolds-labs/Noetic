from noetic_pawp import PAWPTokenizer
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
