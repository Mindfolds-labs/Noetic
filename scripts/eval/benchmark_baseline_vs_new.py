from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
for candidate in (repo_root, repo_root / "src", repo_root / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import statistics
import time

from noetic_pawp.concept_normalizer import ConceptNormalizer
from noetic_pawp.ipa_encoder import align_text_ipa, text_to_ipa
from noetic_pawp.retrieval import RetrievalSample, retrieval_at_k


QUERIES = [
    RetrievalSample(query="hello", language="en", expected_concept_id="concept.greeting.hello"),
    RetrievalSample(query="olá", language="pt", expected_concept_id="concept.greeting.hello"),
    RetrievalSample(query="cafe\u0301", language="pt", expected_concept_id="concept.food.coffee"),
]

WORDS = ["pronúncia", "linguística", "computação", "karaokê"]


def _timeit(fn, rounds: int = 1000) -> float:
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.mean(samples)


def baseline_concept_eval(normalizer: ConceptNormalizer) -> float:
    def run() -> None:
        for q in QUERIES:
            normalizer.resolve_concept(q.query, language=q.language)

    return _timeit(run)


def new_concept_eval(normalizer: ConceptNormalizer) -> tuple[float, float]:
    def run() -> None:
        retrieval_at_k(QUERIES, k=1, normalizer=normalizer)

    return _timeit(run), retrieval_at_k(QUERIES, k=1, normalizer=normalizer)


def ipa_alignment_eval() -> tuple[float, float]:
    def baseline() -> None:
        for word in WORDS:
            ipa = text_to_ipa(word, language="pt")
            align_text_ipa([word[:3], f"##{word[3:]}"], ipa)

    elapsed = _timeit(baseline)
    # proxy score: % of words with monotonic non-empty end span
    ok = 0
    for word in WORDS:
        ipa = text_to_ipa(word, language="pt")
        spans = align_text_ipa([word[:3], f"##{word[3:]}"], ipa)
        if spans and spans[0][0] == 0 and spans[-1][1] >= spans[-1][0]:
            ok += 1
    return elapsed, ok / len(WORDS)


def main() -> None:
    normalizer = ConceptNormalizer()
    baseline_t = baseline_concept_eval(normalizer)
    new_t, r_at_1 = new_concept_eval(normalizer)
    ipa_t, ipa_score = ipa_alignment_eval()

    print("== Benchmark baseline vs novo módulo ==")
    print(f"baseline.resolve_concept.mean_s={baseline_t:.8f}")
    print(f"novo.retrieval_at_1.mean_s={new_t:.8f}")
    print(f"novo.retrieval_at_1.score={r_at_1:.4f}")
    print(f"ipa.alignment.mean_s={ipa_t:.8f}")
    print(f"ipa.alignment.score={ipa_score:.4f}")


if __name__ == "__main__":
    main()
