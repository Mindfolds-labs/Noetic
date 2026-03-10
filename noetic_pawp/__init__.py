"""Noetic PAWP prototype package."""

from .concept_normalizer import ConceptNormalizer, resolve_aliases, resolve_concept
from .config import PAWPConfig, PAWPToken, TokenAnalysis, TokenizerMode
from .feature_flags import FeatureFlags, add_feature_flag_arguments, feature_flags_from_args
from .tokenizer import PAWPTokenizer, compare_wordpiece_vs_pawp, review_alignment
from .ipa_encoder import IPA_TOKEN_TO_ID, IPA_UNK_TOKEN, align_text_ipa, ipa_to_ids, text_to_ipa
from .wordspace_tokenizer import WordSpacePayload, WordSpaceTokenizer
from .gating import SprintGateStatus, can_enable_multimodal, gate_name_for_sprint
from .retrieval import RetrievalSample, rank_concepts, retrieval_at_k

__all__ = [
    "ConceptNormalizer",
    "resolve_concept",
    "resolve_aliases",
    "PAWPConfig",
    "TokenizerMode",
    "PAWPToken",
    "TokenAnalysis",
    "PAWPTokenizer",
    "compare_wordpiece_vs_pawp",
    "review_alignment",
    "feature_flags_from_args",
    "add_feature_flag_arguments",
    "FeatureFlags",
    "WordSpacePayload",
    "WordSpaceTokenizer",
    "IPA_UNK_TOKEN",
    "IPA_TOKEN_TO_ID",
    "text_to_ipa",
    "ipa_to_ids",
    "align_text_ipa",
    "RetrievalSample",
    "rank_concepts",
    "retrieval_at_k",
    "SprintGateStatus",
    "can_enable_multimodal",
    "gate_name_for_sprint",
    "PyFoldsConfig",
    "RIVEEncoder",
    "RadialExtractor",
    "TemporalBuffer",
    "DendriticFuser",
    "SurpriseField",
    "IntentionState",
    "IntentionCtrl",
    "GeoTokenizer",
    "MPJRDLayer",
    "UnifiedPyFoldsEncoder",
]

from .pyfolds_encoder import (
    DendriticFuser,
    GeoTokenizer,
    IntentionCtrl,
    IntentionState,
    MPJRDLayer,
    PyFoldsConfig,
    RIVEEncoder,
    RadialExtractor,
    SurpriseField,
    TemporalBuffer,
    UnifiedPyFoldsEncoder,
)
try:
    from .noetic_model import NoeticMMRNBridge, NoeticPyFoldsConfig, NoeticPyFoldsCore
except ImportError:  # optional torch dependency
    pass
else:
    __all__ += [
        "NoeticPyFoldsConfig",
        "NoeticPyFoldsCore",
        "NoeticMMRNBridge",
    ]

try:
    from .mmrn_prototype import MMRNPrototype, ProjectiveOCR, ProjectiveOCRConfig, ProjectiveOCRLoss
except ImportError:
    pass
else:
    __all__ += [
        "ProjectiveOCRConfig",
        "ProjectiveOCR",
        "ProjectiveOCRLoss",
        "MMRNPrototype",
    ]

try:
    from .rive_mpjrd import RIVEDepthLoss, RIVEDepthLossConfig, RIVEDepthNet
except ImportError:
    pass
else:
    __all__ += [
        "RIVEDepthNet",
        "RIVEDepthLossConfig",
        "RIVEDepthLoss",
    ]

try:
    from noetic_pawp.interfaces import CognitiveOutput, NoeticCore
except ImportError:  # optional torch dependency
    pass
else:
    __all__ += [
        "CognitiveOutput",
        "NoeticCore",
    ]

try:
    from noetic_pawp.noetic_system import (
        ActionController,
        CognitiveController,
        MemoryController,
        NoeticSystem,
        PerceptionController,
    )
except ImportError:
    pass
else:
    __all__ += [
        "NoeticSystem",
        "PerceptionController",
        "CognitiveController",
        "MemoryController",
        "ActionController",
    ]
