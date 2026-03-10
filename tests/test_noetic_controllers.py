import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.noetic_model import NoeticMMRNBridge


def test_noetic_bridge_exposes_modular_controllers_and_runs() -> None:
    model = NoeticMMRNBridge(prs_dim=32)
    assert model.perception is not None
    assert model.cognition is not None
    assert model.memory is not None
    assert model.bridge is not None
    assert model.action is not None

    cn = torch.randn(3, 72)
    out = model(cn, reset_state=True)
    assert out["prs"].shape == (3, 32)
    assert out["z_noetic"].shape[0] == 3
