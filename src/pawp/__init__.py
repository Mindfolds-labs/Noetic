"""PAWP package."""

from .config import CognitiveCoreConfig, FusionConfig, PAWPConfig
from .tokenizer import CognitiveToken, PAWPToken, PAWPTokenizer, TokenAnalysis, compare_wordpiece_vs_pawp, review_alignment

__all__ = [
    "CognitiveCoreConfig",
    "FusionConfig",
    "PAWPConfig",
    "CognitiveToken",
    "PAWPToken",
    "PAWPTokenizer",
    "TokenAnalysis",
    "compare_wordpiece_vs_pawp",
    "review_alignment",
]

try:  # optional torch-backed modules
    from .fusion import PAWPFusion
    from .model import NoeticCognitiveCore, PAWPEncoderModel, PyFoldsNeuralInterface

    __all__ += [
        "PAWPFusion",
        "NoeticCognitiveCore",
        "PAWPEncoderModel",
        "PyFoldsNeuralInterface",
    ]
except Exception:  # pragma: no cover
    pass
