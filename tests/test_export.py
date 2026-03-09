from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from noetic_pawp.export.pawp_onnx import export_pawp_tokenizer_to_onnx
from noetic_pawp.mobile.android_app import generate_android_project
from noetic_pawp.web.tfjs_demo import generate_tfjs_demo


class Tiny(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_export_onnx(tmp_path):
    out = tmp_path / "tiny.onnx"
    model = Tiny()
    sample = torch.zeros(1, 4)
    path = export_pawp_tokenizer_to_onnx(model, sample, out, input_names=["x"], output_names=["y"])
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)


def test_generate_android_project_smoke(tmp_path):
    model = tmp_path / "m.tflite"
    model.write_bytes(b"demo")
    out = generate_android_project(model, tmp_path / "android")
    assert (out / "app/src/main/assets/m.tflite").exists()


def test_generate_tfjs_demo_missing_converter(tmp_path):
    saved = tmp_path / "saved_model"
    saved.mkdir()
    with pytest.raises(RuntimeError):
        generate_tfjs_demo(saved, tmp_path / "web")
