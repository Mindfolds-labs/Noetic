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

    def __post_init__(self) -> None:
        # Mantém a representação IPA consistente em ambos os formatos
        # (sequência serial e unidades), preservando compatibilidade.
        if not self.ipa_sequence and self.ipa_units:
            self.ipa_sequence = "".join(self.ipa_units)
        elif self.ipa_sequence and not self.ipa_units:
            self.ipa_units = [ch for ch in self.ipa_sequence if not ch.isspace()]

    @property
    def token_id(self) -> int:
        """Alias semântico para wp_id (compatibilidade progressiva)."""
        return self.wp_id

    @token_id.setter
    def token_id(self, value: int) -> None:
        self.wp_id = value

    @property
    def text(self) -> str:
        """Alias legado usado por consumidores antigos."""
        return self.wp_piece

    @property
    def ipa_representation(self) -> str:
        """Alias legado para a serialização IPA."""
        return self.ipa_sequence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
