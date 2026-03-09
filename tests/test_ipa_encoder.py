from noetic_pawp.ipa_encoder import IPA_TOKEN_TO_ID, IPA_UNK_TOKEN, align_text_ipa, ipa_to_ids, text_to_ipa


def test_text_to_ipa_multilingual_portuguese_and_english() -> None:
    pt = text_to_ipa("ninho", language="pt")
    en = text_to_ipa("thinking", language="en")

    assert "ɲ" in pt
    assert en == "thinking"


def test_text_to_ipa_accepts_preencoded_ipa() -> None:
    ipa = text_to_ipa("/kɐˈza/", language="pt")

    assert ipa == "kɐˈza"


def test_ipa_id_mapping_is_stable() -> None:
    sequence = "kɐza"

    first = ipa_to_ids(sequence)
    second = ipa_to_ids(sequence)

    assert first == second
    assert IPA_TOKEN_TO_ID[IPA_UNK_TOKEN] == 0


def test_align_text_ipa_reuses_alignment_contract() -> None:
    spans = align_text_ipa(["pro", "##nún", "##cia"], text_to_ipa("pronúncia", language="pt"))

    assert len(spans) == 3
    assert spans[0][0] == 0
    assert spans[-1][1] >= spans[-1][0]
