from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PAWPFusion requires PyTorch. Install with `pip install torch`.") from exc


class PAWPFusion(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        ipa_vocab_size: int,
        root_vocab_size: int,
        lang_vocab_size: int,
        d_word: int = 128,
        d_ipa: int = 64,
        d_root: int = 32,
        d_lang: int = 16,
        d_cn: int = 16,
        d_model: int = 256,
        cn_dim: int = 8,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(word_vocab_size, d_word, padding_idx=pad_idx)
        self.ipa_emb = nn.Embedding(ipa_vocab_size, d_ipa, padding_idx=pad_idx)
        self.root_emb = nn.Embedding(root_vocab_size, d_root, padding_idx=pad_idx)
        self.lang_emb = nn.Embedding(lang_vocab_size, d_lang, padding_idx=pad_idx)
        self.cn_proj = nn.Linear(cn_dim, d_cn)
        self.proj = nn.Linear(d_word + d_ipa + d_root + d_lang + d_cn, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, wp_ids, ipa_ids, root_ids, lang_ids, cn_feats):
        e_word = self.word_emb(wp_ids)
        e_ipa = self.ipa_emb(ipa_ids)
        e_root = self.root_emb(root_ids)
        e_lang = self.lang_emb(lang_ids)
        e_cn = self.cn_proj(cn_feats)
        fused = torch.cat([e_word, e_ipa, e_root, e_lang, e_cn], dim=-1)
        return self.norm(self.proj(fused))
