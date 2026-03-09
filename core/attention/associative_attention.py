from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class AssociativeIncrementalAttention(nn.Module):
    """Scaled dot-product attention com vieses associativos opcionais."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, concept_bias: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model deve ser divisível por nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.concept_bias = float(concept_bias)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.nhead, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        concept_ids: Optional[Tensor] = None,
        ipa_affinity_bias: Optional[Tensor] = None,
        assoc_bias: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if concept_ids is not None and self.concept_bias != 0.0:
            same_concept = concept_ids.unsqueeze(-1) == concept_ids.unsqueeze(-2)
            valid_query = concept_ids.unsqueeze(-1) >= 0
            valid_key = concept_ids.unsqueeze(-2) >= 0
            diagonal = torch.eye(concept_ids.size(1), device=concept_ids.device, dtype=torch.bool).unsqueeze(0)
            concept_matrix = (same_concept & valid_query & valid_key & ~diagonal).unsqueeze(1).to(scores.dtype)
            scores = scores + self.concept_bias * concept_matrix

        if ipa_affinity_bias is not None:
            scores = scores + ipa_affinity_bias.unsqueeze(1)

        if assoc_bias is not None:
            scores = scores + assoc_bias.unsqueeze(1)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                scores = scores + attention_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        out = self.out_proj(context)
        if return_attention:
            return out, attn.mean(dim=1)
        return out


class IsolatedAssociativeEncoderBlock(nn.Module):
    """Bloco de encoder isolado: atenção associativa + FFN padrão."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        concept_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = AssociativeIncrementalAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            concept_bias=concept_bias,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
        concept_ids: Optional[Tensor] = None,
        ipa_affinity_bias: Optional[Tensor] = None,
        assoc_bias: Optional[Tensor] = None,
    ) -> Tensor:
        attn_out = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            concept_ids=concept_ids,
            ipa_affinity_bias=ipa_affinity_bias,
            assoc_bias=assoc_bias,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x
