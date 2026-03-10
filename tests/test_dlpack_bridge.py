import pytest

torch = pytest.importorskip("torch")

from noetic_pyfolds_bridge import NoeticPyFoldsBridge


def test_dlpack_roundtrip_torch() -> None:
    bridge = NoeticPyFoldsBridge()
    x = torch.randn(4, 4)
    cap = bridge.torch_to_dlpack(x)
    y = bridge.torch_from_dlpack(cap)
    assert torch.allclose(x, y)


def test_validate_tensor_handoff_rejects_non_contiguous() -> None:
    x = torch.randn(2, 3).transpose(0, 1)
    with pytest.raises(ValueError):
        NoeticPyFoldsBridge.validate_tensor_handoff(x, require_contiguous=True)


def test_transfer_profiles_zero_copy_path() -> None:
    bridge = NoeticPyFoldsBridge()
    x = torch.randn(8, 8)
    y = bridge.transfer_torch_tensor(x)
    assert torch.allclose(x, y)
    assert bridge.profiler.last_profile.used_dlpack is True
    assert bridge.profiler.last_profile.bytes_transferred == x.numel() * x.element_size()
