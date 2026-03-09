from __future__ import annotations

import json
import tempfile
from pathlib import Path

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge
from noetic_pawp.training.train_multimodal import MultimodalDataset, MultimodalTrainer


def run_pipeline_smoke() -> dict[str, list[float]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.json"
        payload = [{"text": "gato", "image_path": "", "concept": "gato"}]
        path.write_text(json.dumps(payload), encoding="utf-8")
        ds = MultimodalDataset(str(path))
        trainer = MultimodalTrainer(NoeticLeibregBridge(), ds, batch_size=1, device="cpu")
        return trainer.train(epochs=1)


if __name__ == "__main__":
    print(run_pipeline_smoke())
