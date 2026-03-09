from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WordSpaceConfig:
    text_input_dim: int = 768
    image_input_dim: int = 72
    memory_input_dim: int = 256
    target_dim: int = 4
    hyper_dim: int | None = 256
    monitor_dim: int = 4
    use_monitor_projection: bool = True
    similarity_metric: Literal["cosine", "euclidean"] = "cosine"
    use_layernorm: bool = True
    use_l2_normalization: bool = True
    enable_wave_projection: bool = False
    enable_integrity_head: bool = False
    integrity_dim: int = 8
    projector_bias: bool = True
    enable_telemetry: bool = False

    def __post_init__(self) -> None:
        if self.text_input_dim <= 0 or self.image_input_dim <= 0 or self.memory_input_dim <= 0:
            raise ValueError("input dims must be > 0")
        if self.target_dim <= 0:
            raise ValueError("target_dim must be > 0")
        if self.hyper_dim is None:
            # Legacy behavior: callers that explicitly pass hyper_dim=None keep the old
            # target_dim-sized projection path.
            object.__setattr__(self, "hyper_dim", self.target_dim)
        if int(self.hyper_dim) <= 0:
            raise ValueError("hyper_dim must be > 0")
        if self.monitor_dim <= 0:
            raise ValueError("monitor_dim must be > 0")
        if self.integrity_dim <= 0:
            raise ValueError("integrity_dim must be > 0")
        if self.similarity_metric not in {"cosine", "euclidean"}:
            raise ValueError("similarity_metric must be 'cosine' or 'euclidean'")


def verify_integrity(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Returns per-item L2 drift between integrity vectors."""
    if current.shape != reference.shape:
        raise ValueError("Integrity tensors must have the same shape")
    return (current - reference).norm(dim=-1)


class WordSpace(nn.Module):
    def __init__(self, config: WordSpaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or WordSpaceConfig()
        hyper_dim = int(self.config.hyper_dim or self.config.target_dim)
        self.text_head = nn.Linear(self.config.text_input_dim, hyper_dim, bias=self.config.projector_bias)
        self.image_head = nn.Linear(self.config.image_input_dim, hyper_dim, bias=self.config.projector_bias)
        self.memory_head = nn.Linear(self.config.memory_input_dim, hyper_dim, bias=self.config.projector_bias)
        self.layer_norm = nn.LayerNorm(hyper_dim) if self.config.use_layernorm else nn.Identity()
        self.monitor_head = nn.Linear(hyper_dim, self.config.monitor_dim, bias=self.config.projector_bias)
        self.integrity_head = (
            nn.Linear(hyper_dim, self.config.integrity_dim, bias=self.config.projector_bias)
            if self.config.enable_integrity_head
            else None
        )
        self._init_weights()

    @property
    def hyper_dim(self) -> int:
        return int(self.config.hyper_dim or self.config.target_dim)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected rank-2 tensor [batch, dim], got {tuple(x.shape)}")
        return x

    def _project_to_hyper(self, x: torch.Tensor, head: nn.Module) -> torch.Tensor:
        projected = head(self._normalize_shape(x))
        projected = self.layer_norm(projected)
        if self.config.use_l2_normalization:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def _project_monitor(self, hyper_point: torch.Tensor, phase: float | None = None) -> torch.Tensor:
        monitor = self.monitor_head(hyper_point)
        if self.config.enable_wave_projection and phase is not None and monitor.shape[-1] >= 4:
            monitor = self.apply_monitor_phase_rotation(monitor, phase)
        return monitor

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_to_hyper(x, self.text_head)

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_to_hyper(x, self.image_head)

    def project_memory(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_to_hyper(x, self.memory_head)

    def project_monitor(self, hyper_point: torch.Tensor, phase: float | None = None) -> torch.Tensor:
        return self._project_monitor(hyper_point, phase=phase)

    def project_text_monitor(self, x: torch.Tensor, phase: float | None = None) -> torch.Tensor:
        return self.project_monitor(self.project_text(x), phase=phase)

    def project_image_monitor(self, x: torch.Tensor, phase: float | None = None) -> torch.Tensor:
        return self._project_monitor(self.project_image(x), phase=phase)

    def project_memory_monitor(self, x: torch.Tensor, phase: float | None = None) -> torch.Tensor:
        return self._project_monitor(self.project_memory(x), phase=phase)

    @staticmethod
    def apply_monitor_phase_rotation(monitor: torch.Tensor, phase: float) -> torch.Tensor:
        """Applies two independent 2D rotations over a 4D monitor vector."""
        cos_p = torch.cos(torch.tensor(phase, device=monitor.device, dtype=monitor.dtype))
        sin_p = torch.sin(torch.tensor(phase, device=monitor.device, dtype=monitor.dtype))
        out = monitor.clone()
        x0 = out[..., 0] * cos_p - out[..., 1] * sin_p
        x1 = out[..., 0] * sin_p + out[..., 1] * cos_p
        x2 = out[..., 2] * cos_p - out[..., 3] * sin_p
        x3 = out[..., 2] * sin_p + out[..., 3] * cos_p
        out[..., 0], out[..., 1], out[..., 2], out[..., 3] = x0, x1, x2, x3
        return out

    def similarity(self, left: torch.Tensor, right: torch.Tensor, metric: str | None = None) -> torch.Tensor:
        metric_name = metric or self.config.similarity_metric
        if metric_name == "cosine":
            return F.cosine_similarity(left, right, dim=-1)
        if metric_name == "euclidean":
            return -torch.linalg.norm(left - right, dim=-1)
        raise ValueError(f"Unsupported metric: {metric_name}")

    def monitor_similarity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.similarity(left, right, metric=self.config.similarity_metric)

    def pairwise_similarity(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        metric: str | None = None,
    ) -> torch.Tensor:
        """Returns [batch_query, batch_candidates] similarity scores."""
        metric_name = metric or self.config.similarity_metric
        q = self._normalize_shape(query)
        c = self._normalize_shape(candidates)
        if metric_name == "cosine":
            qn = F.normalize(q, p=2, dim=-1)
            cn = F.normalize(c, p=2, dim=-1)
            return qn @ cn.transpose(0, 1)
        if metric_name == "euclidean":
            return -torch.cdist(q, c, p=2)
        raise ValueError(f"Unsupported metric: {metric_name}")

    def retrieve_topk(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        k: int = 5,
        metric: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k retrieval over candidate vectors; complexity O(Bq * Bc * D)."""
        if k <= 0:
            raise ValueError("k must be > 0")
        sims = self.pairwise_similarity(query, candidates, metric=metric)
        top_k = min(k, sims.shape[-1])
        values, indices = torch.topk(sims, k=top_k, dim=-1)
        return values, indices


    def _emit_telemetry(
        self,
        outputs: dict[str, Any],
        telemetry: Any,
        step: int,
        phase: float | None,
    ) -> None:
        if not hasattr(telemetry, "log_step"):
            return
        metrics: dict[str, float] = {"wave_enabled_flag": float(self.config.enable_wave_projection)}
        if phase is not None:
            metrics["wave_phase_mean"] = float(phase)

        hyper_norms: list[torch.Tensor] = []
        monitor_norms: list[torch.Tensor] = []
        projector_vars: list[torch.Tensor] = []
        integrity_means: list[torch.Tensor] = []
        hyper_points: list[torch.Tensor] = []
        for payload in outputs.values():
            hyper = payload["hyper_point"]
            hyper_points.append(hyper)
            hyper_norms.append(hyper.norm(dim=-1))
            projector_vars.append(hyper.var(dim=-1))
            monitor = payload.get("monitor_point")
            if monitor is not None:
                monitor_norms.append(monitor.norm(dim=-1))
            integrity = payload.get("integrity_code")
            if integrity is not None:
                integrity_means.append(integrity.mean(dim=-1))

        if hyper_norms:
            cat = torch.cat(hyper_norms)
            metrics["hyper_norm_mean"] = float(cat.mean().item())
            metrics["hyper_norm_std"] = float(cat.std(unbiased=False).item())
        if monitor_norms:
            metrics["monitor_norm_mean"] = float(torch.cat(monitor_norms).mean().item())
        if projector_vars:
            metrics["projector_output_variance"] = float(torch.cat(projector_vars).mean().item())
        if integrity_means:
            metrics["integrity_code_mean"] = float(torch.cat(integrity_means).mean().item())
        if len(hyper_points) >= 2:
            sims: list[torch.Tensor] = []
            for idx in range(len(hyper_points) - 1):
                sims.append(self.similarity(hyper_points[idx], hyper_points[idx + 1]))
            metrics["similarity_mean"] = float(torch.cat(sims).mean().item())

        telemetry.log_step(step, **metrics)

    def forward(
        self,
        text: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        phase: float | None = None,
        telemetry: Any | None = None,
        telemetry_step: int = 0,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for name, value, projector in (
            ("text", text, self.project_text),
            ("image", image, self.project_image),
            ("memory", memory, self.project_memory),
        ):
            if value is None:
                continue
            hyper = projector(value)
            monitor = self._project_monitor(hyper, phase=phase) if self.config.use_monitor_projection else None
            integrity = self.integrity_head(hyper) if self.integrity_head is not None else None
            outputs[name] = {
                "hyper_point": hyper,
                "monitor_point": monitor,
                "integrity_code": integrity,
                "norms": {
                    "hyper": hyper.norm(dim=-1),
                    "monitor": monitor.norm(dim=-1) if monitor is not None else None,
                },
            }
        if self.config.enable_telemetry and telemetry is not None:
            self._emit_telemetry(outputs=outputs, telemetry=telemetry, step=telemetry_step, phase=phase)
        return outputs
