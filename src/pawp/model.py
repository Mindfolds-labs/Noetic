from __future__ import annotations

from collections import deque
from typing import Deque, Optional

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PAWP models requires PyTorch. Install with `pip install torch`.") from exc

from pawp.config import CognitiveCoreConfig
from pawp.fusion import PAWPFusion

try:  # optional integration path
    from core.attention.associative_attention import IsolatedAssociativeEncoderBlock
except Exception:  # pragma: no cover
    IsolatedAssociativeEncoderBlock = None


class NoeticCognitiveCore(nn.Module):
    """Cognitive layer with working memory, episodic buffer and prediction error."""

    def __init__(self, config: CognitiveCoreConfig) -> None:
        super().__init__()
        self.config = config
        self.working_memory = nn.GRU(config.input_dim, config.hidden_dim, batch_first=True)
        self.memory_attention = nn.MultiheadAttention(
            config.hidden_dim,
            num_heads=config.attention_heads,
            batch_first=True,
        )
        self.predictor = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.episodic_memory: Deque[Tensor] = deque(maxlen=config.memory_slots)

    def _get_memory_bank(self, current: Tensor) -> Tensor:
        if not self.episodic_memory:
            return current
        bank = torch.stack(list(self.episodic_memory), dim=1)  # [B, M, H]
        return bank * self.config.episodic_decay

    def forward(self, fused_embedding: Tensor) -> Tensor:
        wm, _ = self.working_memory(fused_embedding)
        current_state = wm[:, -1, :]
        self.episodic_memory.append(current_state.detach())

        memory_bank = self._get_memory_bank(current_state)
        query = current_state.unsqueeze(1)
        attended, _ = self.memory_attention(query, memory_bank, memory_bank, need_weights=False)
        attended = attended.squeeze(1)

        predicted = self.predictor(attended)
        prediction_error = current_state - predicted
        return self.output_proj(torch.cat([attended, prediction_error], dim=-1))


class PAWPEncoderModel(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        ipa_vocab_size: int,
        root_vocab_size: int,
        lang_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        pad_idx: int = 0,
        enable_associative_attention: bool = False,
        concept_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion = PAWPFusion(
            text_vocab_size=word_vocab_size,
            phonetic_vocab_size=ipa_vocab_size,
            root_vocab_size=root_vocab_size,
            language_vocab_size=lang_vocab_size,
            model_dim=d_model,
            num_heads=nhead,
            pad_idx=pad_idx,
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.cls_head = nn.Linear(d_model, num_classes)
        self.enable_associative_attention = enable_associative_attention and IsolatedAssociativeEncoderBlock is not None
        self.assoc_encoder_block: Optional[nn.Module]
        if self.enable_associative_attention:
            self.assoc_encoder_block = IsolatedAssociativeEncoderBlock(d_model=d_model, nhead=nhead, concept_bias=concept_bias)
        else:
            self.assoc_encoder_block = None

    def forward(
        self,
        wp_ids: Tensor,
        ipa_ids: Tensor,
        root_ids: Tensor,
        lang_ids: Tensor,
        cn_feats: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        concept_ids: Optional[Tensor] = None,
        ipa_affinity_bias: Optional[Tensor] = None,
        assoc_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.fusion(wp_ids, ipa_ids, root_ids, lang_ids)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None

        if self.assoc_encoder_block is not None:
            x = self.assoc_encoder_block(
                x,
                key_padding_mask=key_padding_mask,
                concept_ids=concept_ids,
                ipa_affinity_bias=ipa_affinity_bias,
                assoc_bias=assoc_bias,
            )
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.cls_head(h[:, 0])


class PyFoldsNeuralInterface(nn.Module):
    """Simple projection that keeps PAWP standalone while matching PyFolds tensor shape."""

    def __init__(self, input_dim: int, neuron_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, neuron_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
