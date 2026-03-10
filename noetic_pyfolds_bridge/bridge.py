from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List, Sequence

from noetic_pawp.tokenizer import PAWPTokenizer


@dataclass
class SymbolicState:
    concept_embeddings: List[float]
    phonetic_features: List[float]
    language_id: str


@dataclass
class NeuralState:
    spikes: List[float]
    membrane_potential: List[float]
    dendritic_states: List[float]
    surprise_trace: List[float]


class NoeticPyFoldsBridge:
    """Bridge from symbolic cognition (Noetic) to neural activity (PyFolds-like)."""

    def __init__(self, embedding_dim: int = 16, surprise_lr: float = 0.1) -> None:
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be >= 4")
        self.embedding_dim = embedding_dim
        self.surprise_lr = surprise_lr
        self.tokenizer = PAWPTokenizer()
        self.attention_weights = [1.0 for _ in range(embedding_dim)]
        self._spike_threshold = 0.5
        self._surprise_history: List[float] = []

    def encode_text(self, text: str, language: str = "pt") -> SymbolicState:
        tokens = self.tokenizer.encode(text, language=language, attach_cn=False)
        if not tokens:
            return SymbolicState([0.0] * self.embedding_dim, [0.0] * self.embedding_dim, language)

        concept_embeddings = [0.0] * self.embedding_dim
        phonetic_features = [0.0] * self.embedding_dim
        for token in tokens:
            for i, ch in enumerate(token.wp_piece):
                concept_embeddings[i % self.embedding_dim] += (ord(ch) % 97) / 97.0
            for i, phone in enumerate(token.ipa_units):
                phonetic_features[i % self.embedding_dim] += (ord(phone) % 64) / 64.0

        scale = float(len(tokens))
        concept_embeddings = [v / scale for v in concept_embeddings]
        phonetic_features = [v / scale for v in phonetic_features]
        return SymbolicState(concept_embeddings, phonetic_features, language)

    def convert_to_neural_input(self, symbolic_state: SymbolicState) -> List[float]:
        fused = []
        for i in range(self.embedding_dim):
            weighted = symbolic_state.concept_embeddings[i] * self.attention_weights[i]
            fused.append(weighted + symbolic_state.phonetic_features[i])
        return fused

    def run_pyfolds(self, neural_input: Sequence[float]) -> NeuralState:
        membrane_potential = [x / (1.0 + abs(x)) for x in neural_input]
        dendritic_states = [
            (membrane_potential[i] + membrane_potential[(i - 1) % len(membrane_potential)]) / 2.0
            for i in range(len(membrane_potential))
        ]
        spikes = [1.0 if value > self._spike_threshold else 0.0 for value in membrane_potential]
        surprise_trace = [abs(membrane_potential[i] - dendritic_states[i]) for i in range(len(spikes))]
        return NeuralState(
            spikes=spikes,
            membrane_potential=membrane_potential,
            dendritic_states=dendritic_states,
            surprise_trace=surprise_trace,
        )

    def decode_neural_state(self, predicted_state: Sequence[float], neural_state: NeuralState) -> Dict[str, List[float] | float]:
        if len(predicted_state) != len(neural_state.membrane_potential):
            raise ValueError("predicted_state and neural_state dimensions must match")

        prediction_error = sqrt(
            sum((predicted_state[i] - neural_state.membrane_potential[i]) ** 2 for i in range(len(predicted_state)))
        )
        surprise_signal = prediction_error
        self._surprise_history.append(surprise_signal)

        adaptation = 1.0 / (1.0 + surprise_signal)
        self.attention_weights = [
            (1.0 - self.surprise_lr) * w + self.surprise_lr * adaptation for w in self.attention_weights
        ]

        feedback_embeddings = [
            neural_state.membrane_potential[i] * self.attention_weights[i] for i in range(self.embedding_dim)
        ]

        return {
            "prediction_error": prediction_error,
            "surprise_signal": surprise_signal,
            "adapted_attention_weights": self.attention_weights,
            "updated_cognitive_embeddings": feedback_embeddings,
        }

    @staticmethod
    def torch_to_dlpack(tensor: Any) -> Any:
        """Exports a torch tensor as DLPack capsule for zero-copy hand-off."""
        import torch

        return torch.utils.dlpack.to_dlpack(tensor)

    @staticmethod
    def torch_from_dlpack(capsule: Any) -> Any:
        """Imports a DLPack capsule into torch without copying underlying storage."""
        import torch

        return torch.utils.dlpack.from_dlpack(capsule)
