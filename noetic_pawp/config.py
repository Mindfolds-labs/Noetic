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


@dataclass(init=False)
class PAWPToken:
    wp_piece: str
    token_id: int
    ipa_sequence: str
    script: str
    unicode_meta: Dict[str, Any]
    phoneme_spans: List[Tuple[int, int]]
    root_tag: Optional[str]
    lang: Optional[str]
    cn: Optional[List[float]]
    _ipa_units: List[str]

    def __init__(
        self,
        wp_piece: str,
        token_id: int = 0,
        ipa_sequence: str = "",
        script: str = "OTHER",
        unicode_meta: Optional[Dict[str, Any]] = None,
        phoneme_spans: Optional[List[Tuple[int, int]]] = None,
        root_tag: Optional[str] = None,
        lang: Optional[str] = None,
        cn: Optional[List[float]] = None,
        # Compat aliases
        wp_id: Optional[int] = None,
        ipa_units: Optional[List[str]] = None,
    ) -> None:
        self.wp_piece = wp_piece
        self.token_id = int(wp_id if wp_id is not None else token_id)
        self.script = script
        self.unicode_meta = unicode_meta or {}
        self.phoneme_spans = list(phoneme_spans or [])
        self.root_tag = root_tag
        self.lang = lang
        self.cn = cn

        if not ipa_sequence and ipa_units:
            ipa_sequence = "".join(ipa_units)
        self.ipa_sequence = ipa_sequence
        if ipa_units is not None:
            self._ipa_units = [unit for unit in ipa_units if unit and not unit.isspace()]
        elif ipa_sequence:
            self._ipa_units = [ch for ch in ipa_sequence if not ch.isspace()]
        else:
            self._ipa_units = []

    @property
    def wp_id(self) -> int:
        """Alias legado para o identificador de WordPiece."""
        return self.token_id

    @wp_id.setter
    def wp_id(self, value: int) -> None:
        self.token_id = int(value)

    @property
    def text(self) -> str:
        """Alias legado usado por consumidores antigos."""
        return self.wp_piece

    @property
    def ipa_representation(self) -> str:
        """Alias legado para a serialização IPA."""
        return self.ipa_sequence

    @property
    def ipa_units(self) -> List[str]:
        """Alias legado para IPA em unidades discretas."""
        return list(self._ipa_units)

    @ipa_units.setter
    def ipa_units(self, value: List[str]) -> None:
        self._ipa_units = [unit for unit in value if unit and not unit.isspace()]
        self.ipa_sequence = "".join(self._ipa_units)


    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_ipa_units", None)
        data["wp_id"] = self.wp_id
        data["ipa_units"] = self.ipa_units
        return data
