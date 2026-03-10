from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerceptionOutput:
    cn: Tensor


@dataclass(frozen=True)
class CognitionState:
    z_noetic: Tensor
    spikes: Tensor
    membrane: Tensor
    surprise: Tensor
    surprise_trace: Tensor


@dataclass(frozen=True)
class MemoryQuery:
    z_noetic: Tensor


@dataclass(frozen=True)
class BridgeState:
    z_noetic: Tensor
    prs: Tensor


@dataclass(frozen=True)
class ActionOutput:
    prs: Tensor
    z_noetic: Tensor
    spikes: Tensor
    membrane: Tensor
    surprise: Tensor
    surprise_trace: Tensor
