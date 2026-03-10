from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
for candidate in (repo_root, repo_root / "src", repo_root / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse

from noetic_pawp.gating import SprintGateStatus, can_enable_multimodal, gate_name_for_sprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação de gates por sprint")
    parser.add_argument("--sprint", type=int, required=True)
    parser.add_argument("--tokenizer-ok", action="store_true")
    parser.add_argument("--ipa-ok", action="store_true")
    parser.add_argument("--concept-ok", action="store_true")
    parser.add_argument("--assoc-attention-ok", action="store_true")
    args = parser.parse_args()

    status = SprintGateStatus(
        sprint=args.sprint,
        tokenizer_ok=args.tokenizer_ok,
        ipa_alignment_ok=args.ipa_ok,
        concept_ok=args.concept_ok,
        assoc_attention_ok=args.assoc_attention_ok,
    )

    print(f"gate={gate_name_for_sprint(args.sprint)}")
    print(f"checks={status.checks()}")
    print(f"passed={status.passed()}")
    print(f"multimodal_enabled={can_enable_multimodal(status)}")


if __name__ == "__main__":
    main()
