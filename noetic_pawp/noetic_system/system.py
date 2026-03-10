from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # optional torch dependency
    class _TorchStub:
        Tensor = object

        @staticmethod
        def randn(*_args, **_kwargs):
            raise RuntimeError("torch is required")

    torch = _TorchStub()


from noetic_pawp.feature_flags import FeatureFlags
from noetic_pawp.noetic_model import NoeticMMRNBridge

from .bridge.pyfolds_bridge import PyFoldsBridge
from .cognition.associative_attention import SparseAssociativeAttention
from .cognition.leibreg import LeibRegAdapter
from .cognition.wordspace import WordSpaceAdapter
from .memory.associative_memory import AssociativeMemoryStore
from .memory.concept_store import Concept, ConceptStore
from .ontology.ontology_loader import OntologyLoader
from .perception.ipa_channel import IPAChannel
from .perception.multimodal_fusion import MultimodalFusion
from .perception.pawp_encoder import PAWPEncoder


class PerceptionController:
    def __init__(self) -> None:
        self.pawp = PAWPEncoder()
        self.ipa = IPAChannel()
        self.fusion = MultimodalFusion()

    def encode(self, text: str, language: str = "pt") -> dict[str, Any]:
        points = self.pawp.encode_text(text, language=language)
        ipa_ids = self.ipa.encode(text, language=language)
        fused = [self.fusion.fuse(p.semantic_vector, p.visual_vector, p.phonetic_vector) for p in points]
        return {"points": points, "ipa_ids": ipa_ids, "fused": fused}


class CognitiveController:
    def __init__(self) -> None:
        self.wordspace = WordSpaceAdapter()
        self.leibreg = LeibRegAdapter()
        self.attention = SparseAssociativeAttention()


class MemoryController:
    def __init__(self) -> None:
        self.associative = AssociativeMemoryStore()
        self.concepts = ConceptStore()

    def remember_concepts(self, concepts: list[dict[str, str]]) -> None:
        for c in concepts:
            self.concepts.upsert(Concept(concept_id=c["concept_id"], label=c["label"], source=c["source"]))


class ActionController:
    def __init__(self) -> None:
        self.pyfolds = NoeticMMRNBridge()
        self.bridge = PyFoldsBridge()

    def act(self, cn: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.pyfolds(cn, reset_state=True)


@dataclass
class NoeticSystem:
    flags: FeatureFlags
    perception: PerceptionController
    cognition: CognitiveController
    memory: MemoryController
    action: ActionController
    ontology: OntologyLoader

    @classmethod
    def build(cls, flags: FeatureFlags | None = None) -> "NoeticSystem":
        return cls(
            flags=flags or FeatureFlags(),
            perception=PerceptionController(),
            cognition=CognitiveController(),
            memory=MemoryController(),
            action=ActionController(),
            ontology=OntologyLoader(),
        )
