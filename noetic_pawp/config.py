from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .feature_flags import FeatureFlags


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
    phoneme_spans: List[Tuple[int, int]] = field(default_factory=list)
    root_tag: Optional[str] = None
    lang: Optional[str] = None
    cn: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
