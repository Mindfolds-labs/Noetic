from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def generate_tfjs_demo(saved_model_dir: str | Path, output_dir: str | Path) -> Path:
    """Generate minimal TF.js demo assets from a TensorFlow SavedModel."""
    src = Path(saved_model_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"SavedModel path not found: {src}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)

    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        raise RuntimeError("tensorflowjs_converter not found. Install with `pip install tensorflowjs`.")

    cmd = [converter, "--input_format=tf_saved_model", str(src), str(model_dir)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    template = Path(__file__).parent / "templates" / "index.html"
    shutil.copyfile(template, out / "index.html")
    (out / "README.md").write_text(
        "# Noetic TF.js Demo\n\nRun `python -m http.server` in this folder and open index.html.\n",
        encoding="utf-8",
    )
    return out
