import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.noetic_model import NoeticMMRNBridge
from noetic_pyfolds_bridge import NoeticPyFoldsBridge


def test_bridge_equivalence_fixed_seed() -> None:
    torch.manual_seed(7)
    model = NoeticMMRNBridge(prs_dim=16)
    bridge = NoeticPyFoldsBridge(embedding_dim=16)

    cn = torch.randn(2, 72)
    out = model(cn, reset_state=True)
    exported = out["z_noetic"].detach().contiguous()

    NoeticPyFoldsBridge.validate_tensor_handoff(
        exported,
        expected_shape=tuple(exported.shape),
        expected_dtype=exported.dtype,
        expected_device=str(exported.device),
    )
    imported = bridge.transfer_torch_tensor(exported)
    assert torch.allclose(exported, imported, atol=1e-6, rtol=1e-6)
