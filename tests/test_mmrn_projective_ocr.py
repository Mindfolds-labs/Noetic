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
    assert out["concept_logits"].shape == (4, 128)
    assert out["attribute_logits"].shape == (4, 64)
    assert out["relation_logits"].shape == (4, 64)
    assert out["context_logits"].shape == (4, 32)


def test_projective_ocr_loss_is_finite() -> None:
    model = MMRNPrototype(ProjectiveOCRConfig(image_size=16, num_classes=10))
    x = torch.rand(2, 1, 16, 16)
    ipa = torch.randint(0, 8, (2, 3))
    y = torch.randint(0, 10, (2,))
    out = model(x, ipa)
    target_contour = torch.rand(2, 1, 16, 16)
    criterion = ProjectiveOCRLoss()
    semantic_targets = {
        "concept_id": torch.randint(0, 128, (2,)),
        "context_id": torch.randint(0, 32, (2,)),
        "attributes": torch.randint(0, 2, (2, 64)).float(),
        "relations": torch.randint(0, 2, (2, 64)).float(),
    }
    ldict = criterion(out, y, target_contour, semantic_targets=semantic_targets)
    assert torch.isfinite(ldict["total"]).item()
    assert torch.isfinite(ldict["L_concept"]).item()


def test_projective_ocr_loss_validates_semantic_shapes() -> None:
    model = MMRNPrototype(ProjectiveOCRConfig(image_size=16, num_classes=10))
    x = torch.rand(2, 1, 16, 16)
    ipa = torch.randint(0, 8, (2, 3))
    y = torch.randint(0, 10, (2,))
    out = model(x, ipa)
    target_contour = torch.rand(2, 1, 16, 16)
    criterion = ProjectiveOCRLoss()

    bad_semantic_targets = {
        "concept_id": torch.randint(0, 128, (2,)),
        "attributes": torch.randint(0, 2, (2, 63)).float(),
    }
    with pytest.raises(ValueError, match="attributes"):
        criterion(out, y, target_contour, semantic_targets=bad_semantic_targets)
