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
import statistics
import time

from noetic_pawp.feature_flags import FeatureFlags


def _timeit(fn, rounds: int = 200) -> float:
    vals = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return statistics.mean(vals)


def experimental_stub_step() -> None:
    # Stub de benchmark experimental (RIVE/PGE) para comparação isolada.
    total = 0
    for i in range(500):
        total += (i * i) % 11
    _ = total


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark experimental RIVE/PGE")
    parser.add_argument("--enable-experimental-rive-pge", action="store_true")
    args = parser.parse_args()

    flags = FeatureFlags(enable_experimental_rive_pge=bool(args.enable_experimental_rive_pge))
    if not flags.enable_experimental_rive_pge:
        print("RIVE/PGE experimental está desabilitado. Use --enable-experimental-rive-pge.")
        return

    t = _timeit(experimental_stub_step)
    print("== Benchmark experimental RIVE/PGE ==")
    print(f"experimental.step.mean_s={t:.8f}")


if __name__ == "__main__":
    main()
