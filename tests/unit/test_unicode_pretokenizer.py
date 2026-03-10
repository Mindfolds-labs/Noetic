from pawp.unicode_rules import NO_SPACE_SCRIPTS, PreTokenizer


def test_no_space_scripts_registry_contains_expected_blocks() -> None:
    assert {"THAI", "KHMER", "MYANMAR", "CJK", "HIRAGANA", "KATAKANA"}.issubset(set(NO_SPACE_SCRIPTS))


def test_pretokenizer_splits_cjk_and_japanese_by_grapheme() -> None:
    pre = PreTokenizer()
    text = "日本語かなカナ"
    assert pre.split_words(text) == ["日", "本", "語", "か", "な", "カ", "ナ"]


def test_pretokenizer_preserves_offsets_for_mixed_script_span() -> None:
    pre = PreTokenizer()
    text = "abcไทย 日本語"

    pieces = pre.split_words_with_offsets(text)

    rebuilt = [text[s:e] for _, s, e in pieces]
    assert [token for token, _, _ in pieces] == rebuilt
    assert pieces == [
        ("abc", 0, 3),
        ("ไ", 3, 4),
        ("ท", 4, 5),
        ("ย", 5, 6),
        ("日", 7, 8),
        ("本", 8, 9),
        ("語", 9, 10),
    ]
