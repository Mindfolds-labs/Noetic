from __future__ import annotations

import importlib.util
import math

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

if HAS_TORCH:
    import torch
    from pyfolds.leibreg.wordspace import WordSpace, WordSpaceConfig, verify_integrity


def test_config_backward_compatible_target_dim_default_hyper() -> None:
    cfg = WordSpaceConfig(target_dim=4)
    ws = WordSpace(cfg)
    out = ws.project_text(torch.randn(2, cfg.text_input_dim))
    assert cfg.hyper_dim == 4
    assert out.shape == (2, 4)


def test_hyper_and_monitor_projection_shapes() -> None:
    cfg = WordSpaceConfig(text_input_dim=16, hyper_dim=64, monitor_dim=4, target_dim=4)
    ws = WordSpace(cfg)

    text = torch.randn(8, 16)
    hyper = ws.project_text(text)
    monitor = ws.project_monitor(hyper)

    assert hyper.shape == (8, 64)
    assert monitor.shape == (8, 4)


def test_forward_contract_contains_dual_space() -> None:
    ws = WordSpace(WordSpaceConfig(text_input_dim=10, hyper_dim=32, monitor_dim=4, enable_integrity_head=True))
    out = ws(text=torch.randn(3, 10))

    assert "text" in out
    assert out["text"]["hyper_point"].shape == (3, 32)
    assert out["text"]["monitor_point"].shape == (3, 4)
    assert out["text"]["integrity_code"].shape[-1] == ws.config.integrity_dim


def test_normalization_and_similarity_sanity() -> None:
    ws = WordSpace(WordSpaceConfig(text_input_dim=6, hyper_dim=16, use_l2_normalization=True))
    x = ws.project_text(torch.randn(4, 6))
    norms = x.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    sim = ws.similarity(x, x)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-5)


def test_invalid_shape_raises() -> None:
    ws = WordSpace(WordSpaceConfig(text_input_dim=6))
    with pytest.raises(ValueError):
        ws.project_text(torch.randn(2, 3, 6))


def test_integrity_helper_and_disabled_mode() -> None:
    ws = WordSpace(WordSpaceConfig(text_input_dim=6, enable_integrity_head=False))
    out = ws(text=torch.randn(2, 6))
    assert out["text"]["integrity_code"] is None

    current = torch.tensor([[1.0, 2.0]])
    reference = torch.tensor([[1.0, 1.0]])
    drift = verify_integrity(current, reference)
    assert drift.shape == (1,)
    assert drift.item() == pytest.approx(1.0)


def test_wave_projection_norm_preservation_for_4d_rotation() -> None:
    ws = WordSpace(
        WordSpaceConfig(
            text_input_dim=8,
            hyper_dim=8,
            monitor_dim=4,
            enable_wave_projection=True,
            use_l2_normalization=False,
        )
    )
    hyper = ws.project_text(torch.randn(5, 8))
    monitor = ws.project_monitor(hyper, phase=math.pi / 3)
    unrotated = ws.project_monitor(hyper, phase=None)
    assert torch.allclose(monitor.norm(dim=-1), unrotated.norm(dim=-1), atol=1e-5)


def test_compile_smoke() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    ws = WordSpace(WordSpaceConfig(text_input_dim=8, hyper_dim=16))
    compiled = torch.compile(ws)
    out = compiled(text=torch.randn(2, 8))
    assert out["text"]["hyper_point"].shape == (2, 16)
