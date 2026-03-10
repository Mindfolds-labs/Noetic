"""
Contratos estáveis do Noetic — NUNCA quebre retrocompatibilidade
sem bump de versão major no pyproject.toml.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable
try:
    import torch
except ImportError:  # optional torch dependency
    class _TorchStub:
        Tensor = object

    torch = _TorchStub()


@dataclass
class CognitiveOutput:
    """Saída canônica do núcleo cognitivo.

    Todos os campos são tensores PyTorch com batch dimension na dim 0.
    """
    text_embeddings: torch.Tensor        # (B, T, text_dim)
    phonetic_features: torch.Tensor      # (B, T, phonetic_dim)
    concept_representation: torch.Tensor # (B, T, concept_dim)
    confidence: torch.Tensor             # (B, T)
    prediction_error: torch.Tensor       # (B, T)

    def validate(self) -> bool:
        return all([
            self.text_embeddings.dim() == 3,
            self.phonetic_features.dim() == 3,
            self.concept_representation.dim() == 3,
            self.confidence.dim() == 2,
            self.prediction_error.dim() == 2,
        ])


@runtime_checkable
class NoeticCore(Protocol):
    def process_text(self, text: str, language: str = "pt") -> CognitiveOutput: ...
    def process_batch(self, texts: List[str]) -> List[CognitiveOutput]: ...
