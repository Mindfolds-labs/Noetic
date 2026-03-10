from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .feature_flags import FeatureFlags


class TokenizerMode(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

    @classmethod
    def from_value(cls, value: "TokenizerMode | str") -> "TokenizerMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Invalid tokenizer mode '{value}'. Allowed: {allowed}") from exc


@dataclass
class PAWPConfig:
    lowercase: bool = True
    normalize_form: str = "NFC"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    wordpiece_prefix: str = "##"
    use_phonetic_hints: bool = True
    phonetic_weight: float = 0.2
    root_weight: float = 0.15
    max_vocab_size: int = 5000
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    default_tokenizer_mode: TokenizerMode = TokenizerMode.MULTIMODAL
    # G2P backend ordering controls numerical behavior of IPA side-channel features.
    # We keep a deterministic priority list so stochastic dependency availability
    # does not alter cache identities or training/inference comparability.
    g2p_backend_priority: List[str] = field(default_factory=lambda: ["epitran", "espeak", "fallback"])


@dataclass
class TokenAnalysis:
    original_word: str
    normalized_word: str
    pieces: List[str]
    ipa: str
    root_segments: List[str] = field(default_factory=list)
    used_phonetic_bias: bool = False


@dataclass
class PAWPToken:
    wp_piece: str
    wp_id: int
    ipa_units: List[str] = field(default_factory=list)
    ipa_sequence: str = ""
    phoneme_spans: List[Tuple[int, int]] = field(default_factory=list)
    root_tag: Optional[str] = None
    lang: Optional[str] = None
    script: str = "OTHER"
    unicode_meta: Dict[str, Any] = field(default_factory=dict)
    cn: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
