from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class NoeticPyFoldsConfig:
    """Configuração do núcleo noético baseado em dinâmica dendrítica.

    Parâmetros críticos:
    - tau_mem ∈ (0, 1): fator de decaimento da membrana (estabilidade BIBO em regime discreto).
    - tau_trace ∈ (0, 1): EMA para surpresa/intenção (evita oscilação em sinais ruidosos).
    - spike_threshold: limiar de disparo; controla sparsidade e regime de atividade.
    - surrogate_beta: inclinação da função surrogate; maior valor aproxima Heaviside com risco de gradiente instável.
    """

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
    """Heaviside no forward com gradiente surrogate sigmoidal no backward.

    Isso preserva interpretação de spike binário no estado direto,
    mantendo fluxo de gradiente durante treino supervisionado.
    """

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
    """Núcleo noético com estado recorrente inspirado no neurônio PyFolds.

    Equações discretas implementadas:
      v_t = tau_mem * v_{t-1} + W x_t
      s_t = H(v_t - theta)                (com surrogate no treino)
      r_t = tau_trace * r_{t-1} + (1-tau_trace) * |v_t - v̂_t|

    Onde v̂_t é média por batch do potencial (baseline local), usada como sinal de surpresa.
    """

    def __init__(self, cfg: Optional[NoeticPyFoldsConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or NoeticPyFoldsConfig()
        self.cfg.validate()

        self.in_proj = nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim)
        self.out_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim)
        self.norm = nn.LayerNorm(self.cfg.hidden_dim)

        # Estado interno não-paramétrico (registrado para persistência/estado do módulo).
        self.register_buffer("membrane", torch.zeros(1, self.cfg.hidden_dim))
        self.register_buffer("surprise_trace", torch.zeros(1, self.cfg.hidden_dim))

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.membrane = torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
        self.surprise_trace = torch.zeros(batch_size, self.cfg.hidden_dim, device=device)

    def forward(self, cn: Tensor, *, reset_state: bool = False) -> Dict[str, Tensor]:
        if cn.ndim != 2 or cn.size(-1) != self.cfg.input_dim:
            raise ValueError(f"cn deve ter shape [B, {self.cfg.input_dim}]")

        bsz = cn.size(0)
        if reset_state or self.membrane.size(0) != bsz or self.membrane.device != cn.device:
            self.reset_state(bsz, cn.device)

        x = self.in_proj(cn)
        x = self.norm(x)

        # Dinâmica linear estável em norma para tau_mem < 1.
        v = self.cfg.tau_mem * self.membrane + x

        # Clipping suave evita explosão de estado em entradas fora de escala durante treino inicial.
        v = torch.tanh(v)

        # Spike binário com surrogate gradient.
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
    """Ponte mínima para integrar cn(72) -> núcleo noético -> espaço latente PRS.

    Mantém compatibilidade com o pipeline atual: entrada geométrica (`cn`) e
    saída latente (`prs`) para módulos superiores (fusão, decoder, classificação).
    """

    def __init__(self, prs_dim: int = 128, cfg: Optional[NoeticPyFoldsConfig] = None) -> None:
        super().__init__()
        self.core = NoeticPyFoldsCore(cfg=cfg)
        self.to_prs = nn.Sequential(
            nn.Linear(self.core.cfg.output_dim, prs_dim),
            nn.GELU(),
            nn.LayerNorm(prs_dim),
        )

    def forward(self, cn: Tensor, *, reset_state: bool = False) -> Dict[str, Tensor]:
        out = self.core(cn, reset_state=reset_state)
        out["prs"] = self.to_prs(out["z_noetic"])
        return out
