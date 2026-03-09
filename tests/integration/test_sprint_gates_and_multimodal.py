from noetic_pawp.feature_flags import FeatureFlags
from noetic_pawp.gating import SprintGateStatus, can_enable_multimodal


def test_sprint1_gate_blocks_multimodal_when_text_ipa_concept_fail() -> None:
    status = SprintGateStatus(sprint=1, tokenizer_ok=True, ipa_alignment_ok=False, concept_ok=True)
    flags = FeatureFlags(enable_multimodal=True)

    assert flags.enable_multimodal
    assert can_enable_multimodal(status) is False


def test_sprint2_gate_requires_assoc_attention_gain_or_neutrality() -> None:
    status = SprintGateStatus(
        sprint=2,
        tokenizer_ok=True,
        ipa_alignment_ok=True,
        concept_ok=True,
        assoc_attention_ok=True,
    )

    assert can_enable_multimodal(status) is True
