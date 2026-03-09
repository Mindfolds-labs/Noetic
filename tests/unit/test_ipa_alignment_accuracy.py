from noetic_pawp.ipa_encoder import align_text_ipa, text_to_ipa


def test_ipa_alignment_has_monotonic_full_coverage() -> None:
    word = "pronúncia"
    tokens = ["pro", "##nún", "##cia"]
    ipa = text_to_ipa(word, language="pt")

    spans = align_text_ipa(tokens, ipa)

    assert spans[0][0] == 0
    assert spans[-1][1] == len([ch for ch in ipa if not ch.isspace()])
    assert all(spans[i][1] <= spans[i + 1][1] for i in range(len(spans) - 1))


def test_alignment_handles_non_uniform_grapheme_phoneme_mapping() -> None:
    # "ca" -> /k a/ and "os" -> /u s/: non-uniform split should still map by sound.
    spans = align_text_ipa(["ca", "##os"], ["k", "a", "u", "s"])

    assert spans == [(0, 2), (2, 4)]


def test_alignment_handles_silent_letter_segments() -> None:
    # "gh" in "thought" is silent in this simplified sequence, so middle token may map to empty span.
    spans = align_text_ipa(["th", "##ough", "##t"], ["θ", "ɔ", "t"])

    assert spans[0][0] == 0
    assert spans[-1][1] == 3
    assert spans[1][0] <= spans[1][1]
