from dataclasses import dataclass
from enum import Enum


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
    max_vocab_size: int = 5000
    min_frequency: int = 1
    default_language: str = "en"
    default_tokenizer_mode: TokenizerMode = TokenizerMode.MULTIMODAL


@dataclass
class FusionConfig:
    text_vocab_size: int = 8192
    phonetic_vocab_size: int = 1024
    root_vocab_size: int = 2048
    language_vocab_size: int = 64
    text_dim: int = 96
    phonetic_dim: int = 64
    root_dim: int = 48
    language_dim: int = 16
    model_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    pad_idx: int = 0


@dataclass
class CognitiveCoreConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 128
    memory_slots: int = 32
    episodic_decay: float = 0.95
    attention_heads: int = 4
