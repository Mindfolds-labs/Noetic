from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from noetic_pawp.controllers import (
    ActionController,
    BridgeController,
    CognitiveController,
    MemoryController,
    PerceptionController,
)


@dataclass
class NoeticPyFoldsConfig:
    input_dim: int = 72
    hidden_dim: int = 128
    output_dim: int = 64
    tau_mem: float = 0.9
    tau_trace: float = 0.95
    spike_threshold: float = 0.25
    surrogate_beta: float = 10.0

    def validate(self) -> None:
        if not (0.0 < self.tau_mem < 1.0):
            raise ValueError("tau_mem deve estar em (0,1) para estabilidade do estado de membrana")
        if not (0.0 < self.tau_trace < 1.0):
            raise ValueError("tau_trace deve estar em (0,1) para EMA estável")
        if self.hidden_dim <= 0 or self.output_dim <= 0 or self.input_dim <= 0:
            raise ValueError("dimensões devem ser positivas")
        if self.surrogate_beta <= 0:
            raise ValueError("surrogate_beta deve ser positivo")


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_theta: Tensor, beta: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(v_minus_theta)
        ctx.beta = beta
        return (v_minus_theta > 0.0).to(v_minus_theta.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (v_minus_theta,) = ctx.saved_tensors
        beta = ctx.beta
        s = torch.sigmoid(beta * v_minus_theta)
        grad = beta * s * (1.0 - s)
        return grad_output * grad, None


class NoeticPyFoldsCore(nn.Module):
    def __init__(self, cfg: Optional[NoeticPyFoldsConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or NoeticPyFoldsConfig()
        self.cfg.validate()

        self.in_proj = nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim)
        self.out_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim)
        self.norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.register_buffer("membrane", torch.zeros(1, self.cfg.hidden_dim))
        self.register_buffer("surprise_trace", torch.zeros(1, self.cfg.hidden_dim))

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.membrane = torch.zeros(batch_size, self.cfg.hidden_dim, device=device, dtype=dtype)
        self.surprise_trace = torch.zeros(batch_size, self.cfg.hidden_dim, device=device, dtype=dtype)

    def _needs_reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> bool:
        return (
            self.membrane.size(0) != batch_size
            or str(self.membrane.device) != str(device)
            or self.membrane.dtype != dtype
        )

    def forward(self, cn: Tensor, *, reset_state: bool = False) -> Dict[str, Tensor]:
        if cn.ndim != 2 or cn.size(-1) != self.cfg.input_dim:
            raise ValueError(f"cn deve ter shape [B, {self.cfg.input_dim}]")

        bsz = cn.size(0)
        if reset_state or self._needs_reset(bsz, cn.device, cn.dtype):
            self.reset_state(bsz, cn.device, cn.dtype)

        x = self.norm(self.in_proj(cn))
        v = torch.tanh(self.cfg.tau_mem * self.membrane + x)
        spikes = SurrogateSpike.apply(v - self.cfg.spike_threshold, self.cfg.surrogate_beta)
        baseline = v.mean(dim=0, keepdim=True)
        instant_surprise = (v - baseline).abs()
        surprise_trace = self.cfg.tau_trace * self.surprise_trace + (1.0 - self.cfg.tau_trace) * instant_surprise

        self.membrane = v.detach()
        self.surprise_trace = surprise_trace.detach()

        z_noetic = self.out_proj(spikes)
        return {
            "z_noetic": z_noetic,
            "spikes": spikes,
            "membrane": v,
            "surprise": instant_surprise,
            "surprise_trace": surprise_trace,
        }


class NoeticMMRNBridge(nn.Module):
    """Facade/orchestrator delegating responsibilities to modular controllers."""

    def __init__(self, prs_dim: int = 128, cfg: Optional[NoeticPyFoldsConfig] = None) -> None:
        super().__init__()
        self.core = NoeticPyFoldsCore(cfg=cfg)
        self.to_prs = nn.Sequential(
            nn.Linear(self.core.cfg.output_dim, prs_dim),
            nn.GELU(),
            nn.LayerNorm(prs_dim),
        )
        self.perception = PerceptionController()
        self.cognition = CognitiveController(self.core)
        self.memory = MemoryController()
        self.bridge = BridgeController(self.to_prs)
        self.action = ActionController()

    def forward(self, cn: Tensor, *, reset_state: bool = False) -> Dict[str, Tensor]:
        perception_out = self.perception.prepare(cn)
        cognition_state = self.cognition.run(perception_out, reset_state=reset_state)
        memory_query = self.memory.query(cognition_state)
        bridge_state = self.bridge.project(memory_query)
        action_out = self.action.build(cognition_state, bridge_state)
        return {
            "prs": action_out.prs,
            "z_noetic": action_out.z_noetic,
            "spikes": action_out.spikes,
            "membrane": action_out.membrane,
            "surprise": action_out.surprise,
            "surprise_trace": action_out.surprise_trace,
        }
