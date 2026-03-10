from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pawp import PAWPTokenizer

try:
    import torch
    from pawp.config import CognitiveCoreConfig
    from pawp.model import NoeticCognitiveCore
except Exception:  # pragma: no cover
    torch = None
    CognitiveCoreConfig = None
    NoeticCognitiveCore = None


SAMPLE_TEXT = "Neural cognitive systems learn linguistic structure through phonetic and semantic grounding."


def main(iterations: int = 500) -> None:
    tokenizer = PAWPTokenizer()
    tokenizer.train_vocab([SAMPLE_TEXT] * 50)

    start = time.perf_counter()
    token_count = 0
    for _ in range(iterations):
        token_count += len(tokenizer.encode(SAMPLE_TEXT, language="en"))
    elapsed = time.perf_counter() - start
    print(f"tokens/sec: {token_count / elapsed:.2f}")

    if torch is not None and NoeticCognitiveCore is not None and CognitiveCoreConfig is not None:
        core = NoeticCognitiveCore(CognitiveCoreConfig(input_dim=64, hidden_dim=64, output_dim=64))
        fused = torch.randn(16, 20, 64)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = core(fused)
        elapsed = time.perf_counter() - start
        embeddings = iterations * fused.size(0)
        print(f"embedding throughput (embeddings/sec): {embeddings / elapsed:.2f}")
    else:
        print("embedding throughput (embeddings/sec): unavailable (torch not installed)")

    tracemalloc.start()
    _ = tokenizer.encode(SAMPLE_TEXT, language="en")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"memory usage current={current / 1024:.1f}KB peak={peak / 1024:.1f}KB")


if __name__ == "__main__":
    main()
