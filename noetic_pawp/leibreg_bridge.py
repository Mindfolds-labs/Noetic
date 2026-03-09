"""Noetic-Leibreg integration bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from noetic_pawp.associative_memory import AssociativeMemory
from noetic_pawp.rive_encoder import RIVEEncoder
from pyfolds.leibreg.imagination import Imagination, ImaginationConfig
from pyfolds.leibreg.reg_core import REGCore, REGCoreConfig
from pyfolds.leibreg.wordspace import WordSpace, WordSpaceConfig


@dataclass(frozen=True)
class NoeticLeibregConfig:
    text_dim: int = 768
    image_dim: int = 72
    memory_dim: int = 256
    target_dim: int = 4
    wordspace_config: Optional[WordSpaceConfig] = None
    reg_config: Optional[REGCoreConfig] = None
    imagination_config: Optional[ImaginationConfig] = None
    fusion_mode: str = "mean"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    log_steps: int = 100


class NoeticLeibregBridge(nn.Module):
    def __init__(
        self,
        pawp_model: Optional[nn.Module] = None,
        rive_encoder: Optional[RIVEEncoder] = None,
        associative_memory: Optional[AssociativeMemory] = None,
        config: Optional[NoeticLeibregConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or NoeticLeibregConfig()
        self.pawp = pawp_model
        self.rive = rive_encoder
        self.assoc_mem = associative_memory

        self.text_proj = nn.Linear(self.config.text_dim, self.config.text_dim)
        self.image_proj = nn.Linear(self.config.image_dim, self.config.image_dim)

        self.wordspace = WordSpace(
            self.config.wordspace_config
            or WordSpaceConfig(
                text_input_dim=self.config.text_dim,
                image_input_dim=self.config.image_dim,
                memory_input_dim=self.config.memory_dim,
                target_dim=self.config.target_dim,
            )
        )
        self.reg = REGCore(self.config.reg_config or REGCoreConfig(dim=self.config.target_dim))
        self.imagination = Imagination(
            self.config.imagination_config or ImaginationConfig(dim=self.config.target_dim)
        )

        if self.config.fusion_mode == "gate":
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.config.target_dim * 3, self.config.target_dim * 2),
                nn.GELU(),
                nn.Linear(self.config.target_dim * 2, 3),
                nn.Sigmoid(),
            )

        self.register_buffer("train_steps", torch.tensor(0))
        self.train_losses: list[float] = []

    def _encode_text_fallback(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.config.text_dim, dtype=torch.float32)
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.config.text_dim] += float(ch) / 255.0
        return vec.unsqueeze(0)

    def _get_text_embedding(self, text: Union[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if isinstance(text, torch.Tensor):
            emb = text.float()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return self.text_proj(emb)
        if self.pawp is not None and hasattr(self.pawp, "embed"):
            emb = self.pawp.embed(text)
            return self.text_proj(emb)
        return self.text_proj(self._encode_text_fallback(text))

    def _get_image_embedding(self, image: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image is None:
            return None
        coeffs = (self.rive or RIVEEncoder(output_dim=self.config.image_dim)).encode(image)
        return self.image_proj(coeffs)

    def _get_memory_embedding(self, memory_keys: Optional[Union[str, list[str]]]) -> Optional[torch.Tensor]:
        if not memory_keys:
            return None
        key = memory_keys[0] if isinstance(memory_keys, list) else memory_keys
        if not key:
            return None
        return (self.assoc_mem or AssociativeMemory(self.config.memory_dim)).retrieve(key)

    def _fuse(
        self,
        q_text: Optional[torch.Tensor],
        q_image: Optional[torch.Tensor],
        q_memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tensors = [t for t in (q_text, q_image, q_memory) if t is not None]
        if not tensors:
            device = next(self.parameters()).device
            return torch.zeros(1, self.config.target_dim, device=device)
        if len(tensors) == 1:
            return tensors[0]
        if self.config.fusion_mode == "mean":
            return torch.stack(tensors).mean(dim=0)
        if self.config.fusion_mode == "attention":
            stacked = torch.stack(tensors)
            weights = torch.softmax(stacked.norm(dim=-1, keepdim=True), dim=0)
            return (weights * stacked).sum(dim=0)
        if self.config.fusion_mode == "gate":
            aligned = [t if t.shape[0] == tensors[0].shape[0] else t.expand(tensors[0].shape[0], -1) for t in tensors]
            while len(aligned) < 3:
                aligned.append(torch.zeros_like(aligned[0]))
            concat = torch.cat(aligned[:3], dim=-1)
            gates = self.fusion_gate(concat)
            return gates[:, :1] * aligned[0] + gates[:, 1:2] * aligned[1] + gates[:, 2:3] * aligned[2]
        raise ValueError(f"Modo de fusão desconhecido: {self.config.fusion_mode}")

    def forward(
        self,
        text: Optional[Union[str, torch.Tensor]] = None,
        image: Optional[torch.Tensor] = None,
        memory_keys: Optional[Union[str, list[str]]] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        q_text = self._get_text_embedding(text) if text is not None else None
        q_image = self._get_image_embedding(image)
        q_memory = self._get_memory_embedding(memory_keys)

        q_text_4d = self.wordspace.project_text(q_text) if q_text is not None else None
        q_image_4d = self.wordspace.project_image(q_image) if q_image is not None else None
        q_memory_4d = self.wordspace.project_memory(q_memory) if q_memory is not None else None

        fused = self._fuse(q_text_4d, q_image_4d, q_memory_4d)
        reg_out = self.reg(fused)
        imagination_out = self.imagination(reg_out)

        result: Dict[str, Any] = {
            "fused_point": fused,
            "reg_output": reg_out,
            "imagination": imagination_out["hypothesis"],
            "q_text_4d": q_text_4d,
            "q_image_4d": q_image_4d,
            "q_memory_4d": q_memory_4d,
        }
        if self.imagination.config.with_confidence:
            result["imagination_confidence"] = imagination_out["confidence"]
        if return_intermediate:
            result.update({"q_text_raw": q_text, "q_image_raw": q_image, "q_memory_raw": q_memory})
        return result

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.train()
        self.train_steps += 1
        output = self(
            text=batch.get("text"),
            image=batch.get("image"),
            memory_keys=batch.get("memory_keys"),
        )
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if output["q_text_4d"] is not None and output["q_image_4d"] is not None:
            loss = loss + F.mse_loss(output["q_text_4d"], output["q_image_4d"])
        if output["q_text_4d"] is not None and output["q_memory_4d"] is not None:
            loss = loss + F.mse_loss(output["q_text_4d"], output["q_memory_4d"])
        loss = loss + 0.1 * F.mse_loss(output["reg_output"], output["imagination"])

        if int(self.train_steps.item()) % self.config.log_steps == 0:
            self.train_losses.append(float(loss.item()))
        return {"loss": float(loss.item())}
