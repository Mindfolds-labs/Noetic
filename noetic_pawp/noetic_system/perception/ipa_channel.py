from __future__ import annotations

import torch

from noetic_pawp.ipa_encoder import text_to_ipa, ipa_to_ids


class IPAChannel:
    def encode(self, text: str, language: str = "pt") -> torch.Tensor:
        ipa = text_to_ipa(text, language=language)
        return torch.tensor(ipa_to_ids(ipa), dtype=torch.long)
