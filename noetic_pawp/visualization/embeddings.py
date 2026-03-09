from __future__ import annotations

from pathlib import Path
from typing import Sequence


class EmbeddingVisualizer:
    """TensorBoard embedding/feature logger with minimal validation."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover
            raise ImportError("TensorBoard support requires `torch` and `tensorboard`.") from exc
        self._writer = SummaryWriter(log_dir=str(Path(log_dir).expanduser().resolve()))

    def log_pawp_embeddings(self, embeddings, metadata: Sequence[str] | None = None, step: int = 0) -> None:
        if metadata is not None and len(metadata) != len(embeddings):
            raise ValueError("metadata length must match number of embedding rows")
        self._writer.add_embedding(embeddings, metadata=metadata, global_step=step, tag="pawp_embeddings")

    def log_rive_features(self, features, step: int = 0) -> None:
        self._writer.add_histogram("rive_features", features, global_step=step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
