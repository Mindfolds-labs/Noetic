from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch não instalado no ambiente")

if HAS_TORCH:
    import torch

if HAS_TORCH:
    from noetic_pawp.leibreg_bridge import NoeticLeibregBridge
    from noetic_pawp.training.train_multimodal import MultimodalDataset, MultimodalTrainer


def test_bridge_initialization() -> None:
    bridge = NoeticLeibregBridge()
    assert hasattr(bridge, "wordspace")
    assert hasattr(bridge, "reg")
    assert hasattr(bridge, "imagination")


def test_text_only_forward() -> None:
    bridge = NoeticLeibregBridge()
    out = bridge(text="gato mia")
    assert out["fused_point"].shape[-1] == 4


def test_image_only_forward() -> None:
    bridge = NoeticLeibregBridge()
    out = bridge(image=torch.randn(1, 3, 224, 224))
    assert out["fused_point"].shape[-1] == 4


def test_multimodal_forward() -> None:
    bridge = NoeticLeibregBridge()
    out = bridge(text="gato", image=torch.randn(1, 3, 224, 224), memory_keys="gato")
    assert out["reg_output"].shape[-1] == 4


def test_train_step() -> None:
    bridge = NoeticLeibregBridge()
    loss = bridge.train_step({"text": "gato", "image": torch.randn(1, 3, 224, 224), "memory_keys": "gato"})
    assert isinstance(loss["loss"], float)


def test_training_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.json"
        data = [
            {"text": "gato", "image_path": "", "concept": "gato"},
            {"text": "cachorro", "image_path": "", "concept": "cachorro"},
            {"text": "passaro", "image_path": "", "concept": "passaro"},
        ]
        data_path.write_text(json.dumps(data), encoding="utf-8")

        ds = MultimodalDataset(str(data_path))
        trainer = MultimodalTrainer(NoeticLeibregBridge(), ds, batch_size=2, device="cpu")
        hist = trainer.train(epochs=1)
        assert len(hist["train_loss"]) == 1
