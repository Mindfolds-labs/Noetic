from dataclasses import dataclass


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


@dataclass
class ModelConfig:
    d_word: int = 128
    d_ipa: int = 64
    d_root: int = 32
    d_lang: int = 16
    d_cn: int = 16
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 10
    cn_dim: int = 8
