import pytest

torch = pytest.importorskip("torch")


from noetic_pawp.noetic_model import NoeticMMRNBridge, NoeticPyFoldsConfig, NoeticPyFoldsCore


def test_noetic_config_validation_rejects_unstable_tau() -> None:
    with pytest.raises(ValueError):
        NoeticPyFoldsConfig(tau_mem=1.0).validate()
    with pytest.raises(ValueError):
        NoeticPyFoldsConfig(tau_trace=0.0).validate()


def test_noetic_core_forward_shapes_and_state() -> None:
    cfg = NoeticPyFoldsConfig(input_dim=72, hidden_dim=32, output_dim=16)
    core = NoeticPyFoldsCore(cfg)

    cn = torch.randn(4, 72)
    out = core(cn, reset_state=True)

    assert out["z_noetic"].shape == (4, 16)
    assert out["spikes"].shape == (4, 32)
    assert out["membrane"].shape == (4, 32)
    assert out["surprise_trace"].shape == (4, 32)

    spike_values = torch.unique(out["spikes"])
    assert set(spike_values.tolist()).issubset({0.0, 1.0})


def test_noetic_bridge_outputs_prs() -> None:
    bridge = NoeticMMRNBridge(prs_dim=24)
    cn = torch.randn(2, 72)

    out = bridge(cn, reset_state=True)

    assert out["prs"].shape == (2, 24)
    assert torch.isfinite(out["prs"]).all()
