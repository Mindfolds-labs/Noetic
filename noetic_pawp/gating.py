from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SprintGateStatus:
    sprint: int
    tokenizer_ok: bool = True
    ipa_alignment_ok: bool = True
    concept_ok: bool = True
    assoc_attention_ok: bool = True

    def checks(self) -> Dict[str, bool]:
        base = {
            "tokenizer": self.tokenizer_ok,
            "ipa_alignment": self.ipa_alignment_ok,
            "concept": self.concept_ok,
        }
        if self.sprint >= 2:
            base["assoc_attention"] = self.assoc_attention_ok
        return base

    def passed(self) -> bool:
        return all(self.checks().values())


def can_enable_multimodal(status: SprintGateStatus) -> bool:
    return status.passed()


def gate_name_for_sprint(sprint: int) -> str:
    if sprint <= 1:
        return "sprint1-tokenizer-ipa-concept"
    return "sprint2-assoc-attention"
