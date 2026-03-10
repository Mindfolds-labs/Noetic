import pytest

torch = pytest.importorskip("torch")

from noetic_pyfolds_bridge import NoeticPyFoldsBridge


def test_dlpack_roundtrip_torch() -> None:
    x = torch.randn(4, 4)
    cap = NoeticPyFoldsBridge.torch_to_dlpack(x)
    y = NoeticPyFoldsBridge.torch_from_dlpack(cap)
    assert torch.allclose(x, y)
