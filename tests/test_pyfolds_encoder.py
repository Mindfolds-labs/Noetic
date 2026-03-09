from noetic_pawp.pyfolds_encoder import (
    DendriticFuser,
    IntentionState,
    RIVEEncoder,
    TemporalBuffer,
    UnifiedPyFoldsEncoder,
)


def _image(h: int = 64, w: int = 64):
    return [[(i * w + j) / float(h * w) for j in range(w)] for i in range(h)]


def test_rive_encoder_returns_72() -> None:
    rive = RIVEEncoder(n_coeffs=72, p=8)
    cn = rive(_image())
    assert len(cn) == 72
    assert all(isinstance(v, float) for v in cn)


def test_temporal_buffer_produces_velocity_and_acceleration() -> None:
    tb = TemporalBuffer(T=8, n_coeffs=72)
    c0 = [0.0] * 72
    c1 = [1.0] * 72

    v0, a0 = tb.update(c0, dt=1.0)
    v1, a1 = tb.update(c1, dt=1.0)

    assert all(v == 0.0 for v in v0)
    assert all(v == 0.0 for v in a0)
    assert all(v == 1.0 for v in v1)
    assert all(v == 1.0 for v in a1)


def test_dendritic_fuser_shapes() -> None:
    cn = [float(i) for i in range(72)]
    x_rad = [[[0.0] * 64 for _ in range(4)] for _ in range(2)]
    fused = DendriticFuser()(cn, x_rad)
    assert len(fused) == 2
    assert len(fused[0]) == 4
    assert len(fused[0][0]) == 85


def test_unified_pipeline_step() -> None:
    enc = UnifiedPyFoldsEncoder(n_neurons=16)
    out = enc.step(_image(96, 96))

    assert len(out["cn"]) == 72
    assert len(out["cn_dot"]) == 72
    assert len(out["cn_ddot"]) == 72
    assert len(out["x_rad"]) == 1
    assert len(out["x_rad"][0]) == 4
    assert len(out["x_rad"][0][0]) == 64
    assert len(out["x_fused"][0][0]) == 85
    assert len(out["spikes"][0]) == 16
    assert len(out["tau_geo"]) == 216
    assert out["state"] in {state.value for state in IntentionState}
