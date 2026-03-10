from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path
from typing import Dict, List

from noetic_pyfolds_bridge import NoeticPyFoldsBridge


def _stdp_update(weights: List[float], pre_synaptic: List[float], post_synaptic: List[float], lr: float) -> List[float]:
    updated = []
    for i, w in enumerate(weights):
        dw = lr * (pre_synaptic[i] * post_synaptic[i] - 0.5 * (1.0 - post_synaptic[i]))
        updated.append(max(0.0, min(2.0, w + dw)))
    return updated


def train(dataset: List[str], language: str, epochs: int, lr: float) -> List[Dict[str, float]]:
    bridge = NoeticPyFoldsBridge()
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        cumulative_error = 0.0
        cumulative_spikes = 0.0

        for text in dataset:
            symbolic_state = bridge.encode_text(text, language=language)
            neural_input = bridge.convert_to_neural_input(symbolic_state)
            neural_state = bridge.run_pyfolds(neural_input)

            predicted_state = [
                (symbolic_state.concept_embeddings[i] + symbolic_state.phonetic_features[i]) / 2.0
                for i in range(bridge.embedding_dim)
            ]
            decoded = bridge.decode_neural_state(predicted_state, neural_state)
            bridge.attention_weights = _stdp_update(
                bridge.attention_weights,
                pre_synaptic=neural_input,
                post_synaptic=neural_state.spikes,
                lr=lr,
            )

            cumulative_error += float(decoded["prediction_error"])
            cumulative_spikes += sum(neural_state.spikes)

        history.append(
            {
                "epoch": float(epoch),
                "mean_prediction_error": cumulative_error / max(len(dataset), 1),
                "mean_spikes": cumulative_spikes / max(len(dataset), 1),
            }
        )
        print({"train": history[-1]})

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Noetic ↔ PyFolds bridge with an STDP-like loop.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--language", type=str, default="pt")
    parser.add_argument("--out", type=str, default="noetic_pyfolds_training.json")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    text_dataset = [
        "Noetic integra cognição simbólica.",
        "PyFolds converte entradas em atividade neural.",
        "A atenção adaptativa melhora a previsão.",
        "Sinais de surpresa ajustam embeddings cognitivos.",
    ]

    history = train(dataset=text_dataset, language=args.language, epochs=args.epochs, lr=args.lr)

    output_path = Path(args.out)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print({"saved": str(output_path), "epochs": args.epochs})


if __name__ == "__main__":
    main()
