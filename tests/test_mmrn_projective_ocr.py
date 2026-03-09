from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from noetic_pawp.mmrn_prototype import MMRNPrototype, ProjectiveOCRConfig, ProjectiveOCRLoss


def test_mmrn_forward_shapes() -> None:
    model = MMRNPrototype(ProjectiveOCRConfig(image_size=16, num_classes=10))
    x = torch.rand(4, 1, 16, 16)
    ipa = torch.randint(0, 8, (4, 3))
    out = model(x, ipa)
    assert out["y_hat"].shape == (4, 10)
    assert out["C_hat"].shape == (4, 1, 16, 16)
    assert out["B_hat"].shape[-2:] == (4, 2)
    assert out["prs"].shape[0] == 4


def test_projective_ocr_loss_is_finite() -> None:
    model = MMRNPrototype(ProjectiveOCRConfig(image_size=16, num_classes=10))
    x = torch.rand(2, 1, 16, 16)
    ipa = torch.randint(0, 8, (2, 3))
    y = torch.randint(0, 10, (2,))
    out = model(x, ipa)
    target_contour = torch.rand(2, 1, 16, 16)
    criterion = ProjectiveOCRLoss()
    ldict = criterion(out, y, target_contour)
    assert torch.isfinite(ldict["total"]).item()
