from pawp.unicode_rules import PreTokenizer


def test_pretokenizer_no_space_script_uses_grapheme_offsets() -> None:
    tokenizer = PreTokenizer()
    text = "ภาษาไทย日本語"

    tokens = tokenizer.split_words_with_offsets(text)

    assert [tok for tok, _, _ in tokens] == ["ภ", "า", "ษ", "า", "ไ", "ท", "ย", "日", "本", "語"]
    assert [text[s:e] for _, s, e in tokens] == [tok for tok, _, _ in tokens]


def test_pretokenizer_mixed_script_splits_mode_switch_and_preserves_offsets() -> None:
    tokenizer = PreTokenizer()
    text = "abcไทย-日本def"

    tokens = tokenizer.split_words_with_offsets(text)

    assert [tok for tok, _, _ in tokens] == ["abc", "ไ", "ท", "ย", "-", "日", "本", "def"]
    assert [(s, e) for _, s, e in tokens] == [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)]
    assert [text[s:e] for _, s, e in tokens] == [tok for tok, _, _ in tokens]
