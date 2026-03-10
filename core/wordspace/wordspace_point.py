from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from noetic_pawp.wordspace_tokenizer import WordSpacePayload


Vector = tuple[float, ...]
TensorLike = "torch.Tensor | Sequence[float]"
Projector = Callable[[Any], Sequence[float]]


@dataclass(frozen=True)
class WordSpacePoint:
    text_vec: Vector
    ipa_vec: Vector
    context_vec: Vector
    assoc_vec: Vector
    concept_id: Optional[str] = None
    confidence: Optional[float] = None
    monitor_vec: Optional[Vector] = None
    integrity_code: Optional[Vector] = None
    metadata: Optional[dict[str, Any]] = None
    hyper_vec: Optional[Vector] = None
    phase: Optional[float] = None
    core_identity: Optional[Vector] = None
    context_state: Optional[Vector] = None
    is_anchor: bool = False

    @staticmethod
    def _coerce_vector(name: str, value: TensorLike | None) -> Optional[Vector]:
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            if value.dim() != 1:
                raise TypeError(f"{name} tensor must be rank-1")
            return tuple(float(v) for v in value.detach().cpu().tolist())
        if any(not isinstance(item, (int, float)) for item in value):
            raise TypeError(f"{name} must contain only numeric values")
        return tuple(float(item) for item in value)

    def __post_init__(self) -> None:
        for name in ("text_vec", "ipa_vec", "context_vec", "assoc_vec"):
            coerced = self._coerce_vector(name, getattr(self, name))
            assert coerced is not None
            object.__setattr__(self, name, coerced)
        for name in ("monitor_vec", "integrity_code", "hyper_vec", "core_identity", "context_state"):
            object.__setattr__(self, name, self._coerce_vector(name, getattr(self, name)))
        core_identity = self.core_identity if self.core_identity is not None else self.assoc_vec
        context_state = self.context_state if self.context_state is not None else self.context_vec
        object.__setattr__(self, "core_identity", core_identity)
        object.__setattr__(self, "context_state", context_state)
        if core_identity is not None and context_state is not None and len(core_identity) != len(context_state):
            if len(core_identity) == 0:
                core_identity = tuple(0.0 for _ in context_state)
                object.__setattr__(self, "core_identity", core_identity)
            elif len(context_state) == 0:
                context_state = tuple(0.0 for _ in core_identity)
                object.__setattr__(self, "context_state", context_state)
            else:
                raise ValueError("core_identity and context_state must have the same dimension")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            len(self.text_vec),
            len(self.ipa_vec),
            len(self.context_vec),
            len(self.assoc_vec),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_vec": list(self.text_vec),
            "ipa_vec": list(self.ipa_vec),
            "context_vec": list(self.context_vec),
            "assoc_vec": list(self.assoc_vec),
            "concept_id": self.concept_id,
            "confidence": self.confidence,
            "monitor_vec": list(self.monitor_vec) if self.monitor_vec is not None else None,
            "integrity_code": list(self.integrity_code) if self.integrity_code is not None else None,
            "metadata": self.metadata,
            "hyper_vec": list(self.hyper_vec) if self.hyper_vec is not None else None,
            "phase": self.phase,
            "core_identity": list(self.core_identity) if self.core_identity is not None else None,
            "context_state": list(self.context_state) if self.context_state is not None else None,
            "is_anchor": self.is_anchor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordSpacePoint":
        return cls(
            text_vec=tuple(data.get("text_vec", [])),
            ipa_vec=tuple(data.get("ipa_vec", [])),
            context_vec=tuple(data.get("context_vec", [])),
            assoc_vec=tuple(data.get("assoc_vec", [])),
            concept_id=data.get("concept_id"),
            confidence=data.get("confidence"),
            monitor_vec=tuple(data["monitor_vec"]) if data.get("monitor_vec") is not None else None,
            integrity_code=(
                tuple(data["integrity_code"]) if data.get("integrity_code") is not None else None
            ),
            metadata=data.get("metadata"),
            hyper_vec=tuple(data["hyper_vec"]) if data.get("hyper_vec") is not None else None,
            phase=data.get("phase"),
            core_identity=tuple(data["core_identity"]) if data.get("core_identity") is not None else None,
            context_state=tuple(data["context_state"]) if data.get("context_state") is not None else None,
            is_anchor=bool(data.get("is_anchor", False)),
        )

    @property
    def concept_representation(self) -> Vector:
        core = self.core_identity or ()
        ctx = self.context_state or ()
        if not core and not ctx:
            return ()
        if len(core) != len(ctx):
            raise ValueError("core_identity and context_state dimensions must match")
        return tuple(a + b for a, b in zip(core, ctx))

    @staticmethod
    def stack_vectors(points: Sequence["WordSpacePoint"], field: str):
        """Stacks a vector field into [batch, dim] tensor for efficient batch ops."""
        if torch is None:
            raise RuntimeError("PyTorch is required for stack_vectors")
        vectors = [getattr(point, field) for point in points]
        if not vectors:
            return torch.empty(0, 0)
        if any(v is None for v in vectors):
            raise ValueError(f"Field {field} contains None values")
        first_dim = len(vectors[0])
        if any(len(v) != first_dim for v in vectors):
            raise ValueError(f"Field {field} has inconsistent dimensions")
        return torch.tensor(vectors, dtype=torch.float32)


class QuaternionWordSpace:
    """Small quaternion helper for modality fusion math."""

    @staticmethod
    def multiply(q1: Sequence[float], q2: Sequence[float]) -> tuple[float, float, float, float]:
        a1, b1, c1, d1 = q1
        a2, b2, c2, d2 = q2
        return (
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        )

    @staticmethod
    def conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
        a, b, c, d = q
        return (a, -b, -c, -d)

    @staticmethod
    def normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
        norm = sqrt(sum(v * v for v in q))
        if norm == 0:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(float(v / norm) for v in q)  # type: ignore[return-value]

    @classmethod
    def rotate_vector(cls, vector: Sequence[float], rotor: Sequence[float]) -> tuple[float, float, float]:
        vx, vy, vz = vector
        rot = cls.normalize(rotor)
        vec_q = (0.0, vx, vy, vz)
        out = cls.multiply(cls.multiply(rot, vec_q), cls.conjugate(rot))
        return (out[1], out[2], out[3])


def _default_text_projector(token_id: int) -> Vector:
    return (float(token_id),)


def _default_ipa_projector(ipa_ids: Sequence[int]) -> Vector:
    return tuple(float(v) for v in ipa_ids)


def wordspace_points_from_payload(
    payload: "WordSpacePayload",
    text_projector: Optional[Projector] = None,
    ipa_projector: Optional[Projector] = None,
    context_projector: Optional[Projector] = None,
    assoc_projector: Optional[Projector] = None,
    default_confidence: Optional[float] = 1.0,
) -> list[WordSpacePoint]:
    text_projector = text_projector or _default_text_projector
    ipa_projector = ipa_projector or _default_ipa_projector
    context_projector = context_projector or (lambda _: ())
    assoc_projector = assoc_projector or (lambda _: ())

    points: list[WordSpacePoint] = []
    for idx, token_id in enumerate(payload.token_ids):
        concept_id = payload.concept_ids[idx] if payload.concept_ids is not None else None
        points.append(
            WordSpacePoint(
                text_vec=tuple(float(v) for v in text_projector(token_id)),
                ipa_vec=tuple(float(v) for v in ipa_projector(payload.token_ipa_ids[idx])),
                context_vec=tuple(float(v) for v in context_projector(payload.token_text[idx])),
                assoc_vec=tuple(float(v) for v in assoc_projector(concept_id)),
                concept_id=concept_id,
                confidence=default_confidence,
            )
        )
    return points


def payload_from_wordspace_points(
    points: Sequence[WordSpacePoint],
    token_text: Optional[Sequence[str]] = None,
    token_offsets: Optional[Sequence[tuple[int, int]]] = None,
):
    from noetic_pawp.wordspace_tokenizer import WordSpacePayload

    text = list(token_text) if token_text is not None else ["" for _ in points]
    offsets = list(token_offsets) if token_offsets is not None else [(0, 0) for _ in points]
    token_ids = [int(round(point.text_vec[0])) if point.text_vec else 0 for point in points]
    token_ipa_ids = [[int(round(v)) for v in point.ipa_vec] for point in points]
    concept_ids = [point.concept_id for point in points]

    return WordSpacePayload(
        token_ids=token_ids,
        token_text=text,
        token_offsets=offsets,
        token_ipa_ids=token_ipa_ids,
        concept_ids=concept_ids,
    )
