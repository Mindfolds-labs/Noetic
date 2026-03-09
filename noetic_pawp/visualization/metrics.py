from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class MetricsTracker:
    """In-memory metric tracker with optional TensorBoard sink."""

    tensorboard_log_dir: str | None = None
    _records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._writer = None
        if self.tensorboard_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as exc:  # pragma: no cover
                raise ImportError("TensorBoard integration requires torch+tensorboard") from exc
            self._writer = SummaryWriter(self.tensorboard_log_dir)

    def log_step(self, step: int, **metrics: float) -> None:
        record = {"step": step, "timestamp": time(), **metrics}
        self._records.append(record)
        if self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, step)

    def log_epoch(self, epoch: int, **metrics: float) -> None:
        self.log_step(epoch, **{f"epoch/{k}": v for k, v in metrics.items()})

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
