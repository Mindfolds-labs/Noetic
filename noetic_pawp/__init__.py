"""Noetic PAWP prototype package."""

from .config import PAWPConfig, PAWPToken, TokenAnalysis
from .tokenizer import PAWPTokenizer, compare_wordpiece_vs_pawp, review_alignment

__all__ = [
    "PAWPConfig",
    "PAWPToken",
    "TokenAnalysis",
    "PAWPTokenizer",
    "compare_wordpiece_vs_pawp",
    "review_alignment",
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
