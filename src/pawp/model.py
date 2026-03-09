from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PAWPEncoderModel requires PyTorch. Install with `pip install torch`.") from exc

from pawp.fusion import PAWPFusion


class PAWPEncoderModel(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        ipa_vocab_size: int,
        root_vocab_size: int,
        lang_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.fusion = PAWPFusion(
            word_vocab_size=word_vocab_size,
            ipa_vocab_size=ipa_vocab_size,
            root_vocab_size=root_vocab_size,
            lang_vocab_size=lang_vocab_size,
            d_model=d_model,
            pad_idx=pad_idx,
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, wp_ids, ipa_ids, root_ids, lang_ids, cn_feats, attention_mask=None):
        x = self.fusion(wp_ids, ipa_ids, root_ids, lang_ids, cn_feats)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = h[:, 0]
        return self.cls_head(pooled)
