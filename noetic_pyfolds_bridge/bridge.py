from __future__ import annotations

import time
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


@dataclass(frozen=True)
class BridgeProfile:
    transfer_time_ms: float
    copy_count: int
    bytes_transferred: int
    used_dlpack: bool
    serialization_fallback: bool


class BridgeProfiler:
    def __init__(self) -> None:
        self.last_profile = BridgeProfile(0.0, 0, 0, False, False)

    def record(self, *, started_at: float, bytes_transferred: int, used_dlpack: bool, copy_count: int = 0, serialization_fallback: bool = False) -> None:
        self.last_profile = BridgeProfile(
            transfer_time_ms=(time.perf_counter() - started_at) * 1000.0,
            copy_count=copy_count,
            bytes_transferred=bytes_transferred,
            used_dlpack=used_dlpack,
            serialization_fallback=serialization_fallback,
        )


class NoeticPyFoldsBridge:
    def __init__(self, embedding_dim: int = 16, surprise_lr: float = 0.1) -> None:
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be >= 4")
        self.embedding_dim = embedding_dim
        self.surprise_lr = surprise_lr
        self.tokenizer = PAWPTokenizer()
        self.attention_weights = [1.0 for _ in range(embedding_dim)]
        self._spike_threshold = 0.5
        self._surprise_history: List[float] = []
        self.profiler = BridgeProfiler()

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
    def validate_tensor_handoff(
        tensor: Any,
        *,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: Any | None = None,
        expected_device: str | None = None,
        require_contiguous: bool = True,
    ) -> None:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("handoff payload must be a torch.Tensor")
        if expected_shape is not None and tuple(tensor.shape) != expected_shape:
            raise ValueError(f"shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise ValueError(f"dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
        if expected_device is not None and str(tensor.device) != expected_device:
            raise ValueError(f"device mismatch: expected {expected_device}, got {tensor.device}")
        if require_contiguous and not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous for bridge handoff")

    @staticmethod
    def torch_to_dlpack(tensor: Any) -> Any:
        import torch

        NoeticPyFoldsBridge.validate_tensor_handoff(tensor)
        return torch.utils.dlpack.to_dlpack(tensor)

    @staticmethod
    def torch_from_dlpack(capsule: Any) -> Any:
        import torch

        return torch.utils.dlpack.from_dlpack(capsule)

    def transfer_torch_tensor(self, tensor: Any) -> Any:
        import torch

        self.validate_tensor_handoff(tensor)
        start = time.perf_counter()
        cap = torch.utils.dlpack.to_dlpack(tensor)
        out = torch.utils.dlpack.from_dlpack(cap)
        self.profiler.record(started_at=start, bytes_transferred=tensor.numel() * tensor.element_size(), used_dlpack=True)
        return out
