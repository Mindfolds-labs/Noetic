import pytest

torch = pytest.importorskip("torch")

from pawp.config import CognitiveCoreConfig
from pawp.model import NoeticCognitiveCore


def test_cognitive_core_output_shape_and_memory() -> None:
    core = NoeticCognitiveCore(CognitiveCoreConfig(input_dim=32, hidden_dim=32, output_dim=16, memory_slots=4))
    fused = torch.randn(3, 6, 32)

    out1 = core(fused)
    out2 = core(fused)

    assert out1.shape == (3, 16)
    assert out2.shape == (3, 16)
    assert len(core.episodic_memory) >= 1
