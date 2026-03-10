#!/usr/bin/env python
"""Verifica saúde mínima do repositório Noetic antes de um release."""
from __future__ import annotations
import sys, time, importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _check(label: str, fn) -> bool:
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except Exception as exc:
        print(f"  ✗ {label}: {exc}")
        return False


def main() -> int:
    print("=== Noetic Health Check ===")
    results = []

    results.append(_check(
        "import noetic_pawp",
        lambda: importlib.import_module("noetic_pawp")
    ))
    results.append(_check(
        "import noetic_pawp.interfaces",
        lambda: importlib.import_module("noetic_pawp.interfaces")
    ))
    results.append(_check(
        "PAWPTokenizer encode PT",
        lambda: __import__("noetic_pawp").PAWPTokenizer().encode("pronúncia", language="pt")
    ))
    results.append(_check(
        "PAWPTokenizer encode EN",
        lambda: __import__("noetic_pawp").PAWPTokenizer().encode("hello world", language="en")
    ))
    results.append(_check(
        "ConceptNormalizer resolve PT",
        lambda: __import__("noetic_pawp").ConceptNormalizer().resolve_concept("olá", "pt")
    ))
    results.append(_check(
        "SprintGates sprint1 passed",
        lambda: __import__("noetic_pawp").SprintGateStatus(
            sprint=1, tokenizer_ok=True, ipa_alignment_ok=True, concept_ok=True
        ).passed()
    ))

    ok = sum(results)
    total = len(results)
    print(f"\n{'PASS' if ok == total else 'FAIL'} — {ok}/{total} checks ok")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
