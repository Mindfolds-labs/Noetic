from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

try:
    import torch
except ImportError:  # optional torch dependency
    class _TorchStub:
        Tensor = object

        @staticmethod
        def zeros(*_args, **_kwargs):
            raise RuntimeError("torch is required")

        @staticmethod
        def tensor(*_args, **_kwargs):
            raise RuntimeError("torch is required")

        float32 = None

    torch = _TorchStub()

from noetic_pawp.ipa_encoder import ipa_to_ids
from noetic_pawp.tokenizer import PAWPTokenizer


@dataclass
class PAWPMultimodalPoint:
    token: str
    phonetic_vector: torch.Tensor
    semantic_vector: torch.Tensor
    visual_vector: torch.Tensor
    syntactic_features: torch.Tensor


class PAWPEncoder:
    """Maps PAWP tokens into multimodal points."""

    def __init__(self, tokenizer: PAWPTokenizer | None = None, semantic_dim: int = 32, visual_dim: int = 16) -> None:
        self.tokenizer = tokenizer or PAWPTokenizer()
        self.semantic_dim = semantic_dim
        self.visual_dim = visual_dim

    def _hash_vec(self, text: str, dim: int) -> torch.Tensor:
        vals = [(ord(ch) % 97) / 97.0 for ch in text] or [0.0]
        out = torch.zeros(dim, dtype=torch.float32)
        for i in range(dim):
            out[i] = vals[i % len(vals)]
        return out

    def encode_text(self, text: str, language: str = "pt") -> List[PAWPMultimodalPoint]:
        tokens = self.tokenizer.encode(text, language=language, attach_cn=False)
        points: List[PAWPMultimodalPoint] = []
        for token in tokens:
            ipa_ids = ipa_to_ids(token.ipa_sequence)
            points.append(
                PAWPMultimodalPoint(
                    token=token.wp_piece,
                    phonetic_vector=torch.tensor(ipa_ids, dtype=torch.float32),
                    semantic_vector=self._hash_vec(token.wp_piece, self.semantic_dim),
                    visual_vector=self._hash_vec(token.wp_piece[::-1], self.visual_dim),
                    syntactic_features=torch.tensor(
                        [
                            float(token.wp_id),
                            float(len(token.wp_piece)),
                            float(len(token.ipa_units)),
                            float(len(token.phoneme_spans)),
                        ],
                        dtype=torch.float32,
                    ),
                )
            )
        return points

    @staticmethod
    def validate_mapping(points: Sequence[PAWPMultimodalPoint]) -> bool:
        if not points:
            return True
        return all(
            p.phonetic_vector.ndim == 1
            and p.semantic_vector.ndim == 1
            and p.visual_vector.ndim == 1
            and p.syntactic_features.shape == (4,)
            for p in points
        )
