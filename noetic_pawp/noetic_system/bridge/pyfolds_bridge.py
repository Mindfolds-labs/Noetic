from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BridgeTensor:
    tensor: torch.Tensor


class PyFoldsBridge:
    """Bridge utilities optimized for zero-copy tensor exchange."""

    @staticmethod
    def to_dlpack(tensor: torch.Tensor):
        return torch.utils.dlpack.to_dlpack(tensor)

    @staticmethod
    def from_dlpack(capsule) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(capsule)

    @staticmethod
    def torch_to_tensorflow(tensor: torch.Tensor):
        import tensorflow as tf

        return tf.experimental.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

    @staticmethod
    def tensorflow_to_torch(tensor) -> torch.Tensor:
        import tensorflow as tf

        return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(tensor))
