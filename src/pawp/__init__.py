"""PAWP package."""

import warnings

from .config import CognitiveCoreConfig, FusionConfig, PAWPConfig, TokenizerMode
from .tokenizer import CognitiveToken, PAWPToken, PAWPTokenizer, TokenAnalysis, compare_wordpiece_vs_pawp, review_alignment

__all__ = [
    "CognitiveCoreConfig",
    "FusionConfig",
    "PAWPConfig",
    "TokenizerMode",
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
except ModuleNotFoundError as exc:  # pragma: no cover
    # Keep package importable when optional dependencies (e.g., torch) are absent.
    if exc.name == "torch":
        pass
    else:
        raise
except ImportError as exc:  # pragma: no cover
    warnings.warn(f"Optional PAWP torch-backed modules are unavailable: {exc}", RuntimeWarning, stacklevel=2)
    raise
