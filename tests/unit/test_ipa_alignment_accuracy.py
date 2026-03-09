from noetic_pawp.ipa_encoder import align_text_ipa, text_to_ipa


def test_ipa_alignment_has_monotonic_full_coverage() -> None:
    word = "pronúncia"
    tokens = ["pro", "##nún", "##cia"]
    ipa = text_to_ipa(word, language="pt")

    spans = align_text_ipa(tokens, ipa)

    assert spans[0][0] == 0
    assert spans[-1][1] == len([ch for ch in ipa if not ch.isspace()])
    assert all(spans[i][1] <= spans[i + 1][1] for i in range(len(spans) - 1))
