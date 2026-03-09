#!/usr/bin/env python
from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch não instalado; execute `pip install torch` para rodar este smoke test.") from exc

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge


def test_smoke() -> bool:
    bridge = NoeticLeibregBridge()
    t = bridge(text="gato mia")
    assert t["fused_point"].shape[-1] == 4
    img = torch.randn(1, 3, 224, 224)
    i = bridge(image=img)
    assert i["fused_point"].shape[-1] == 4
    m = bridge(text="gato", image=img, memory_keys="gato")
    assert m["fused_point"].shape[-1] == 4
    loss = bridge.train_step({"text": "gato", "image": img, "memory_keys": "gato"})
    assert isinstance(loss["loss"], float)
    print("OK")
    return True


if __name__ == "__main__":
    test_smoke()
