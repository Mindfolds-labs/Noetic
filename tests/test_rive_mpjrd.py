from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.rive_mpjrd import RIVEDepthLoss, RIVEDepthNet, compute_depth_metrics


def test_rive_mpjrd_shapes() -> None:
    model = RIVEDepthNet(output_h=16, output_w=16, hidden_dims=(32, 32))
    x = torch.rand(2, 3, 16, 16)
    depth, rive = model(x)
    assert depth.shape == (2, 1, 16, 16)
    assert rive.shape == (2, 84)


def test_rive_loss_and_metrics_finite() -> None:
    model = RIVEDepthNet(output_h=16, output_w=16, hidden_dims=(32, 32))
    x = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 1, 16, 16) * 10 + 0.1
    pred, rive = model(x)
    loss = RIVEDepthLoss()(pred, gt, rive)
    assert torch.isfinite(loss["total"]).item()
    metrics = compute_depth_metrics(pred, gt, max_depth=10.0)
    assert "AbsRel" in metrics
