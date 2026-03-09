from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def benchmark_tf_dataset(dataset, num_steps: int = 10) -> dict[str, float]:
    start = time.perf_counter()
    seen = 0
    for batch in dataset.take(num_steps):
        seen += int(batch["image"].shape[0])
    elapsed = time.perf_counter() - start
    return {"steps": num_steps, "samples": seen, "seconds": elapsed, "throughput": seen / max(elapsed, 1e-9)}


def benchmark_rive(preprocessor, image_batch, num_steps: int = 10) -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(num_steps):
        _ = preprocessor(image_batch)
    elapsed = time.perf_counter() - start
    return {"steps": num_steps, "seconds": elapsed, "latency_ms": (elapsed / num_steps) * 1000.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Noetic benchmarks")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    results = {"message": "Use benchmark_tf_dataset/benchmark_rive from Python for custom runs."}
    print(json.dumps(results, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
