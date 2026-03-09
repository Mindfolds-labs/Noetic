from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for ONNX export. Install with `pip install torch`.") from exc
    return torch


def _require_onnx():
    try:
        import onnx
    except Exception as exc:  # pragma: no cover
        raise ImportError("ONNX package is required. Install with `pip install onnx`.") from exc
    return onnx


def _validate_model_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".onnx":
        raise ValueError("Output file must end with .onnx")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _export_to_onnx(
    model: Any,
    sample_input: Any,
    output_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> Path:
    torch = _require_torch()
    onnx = _require_onnx()

    path = _validate_model_path(output_path)
    model.eval()
    try:
        torch.onnx.export(
            model,
            sample_input,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"ONNX export failed, likely due to unsupported operator. Details: {exc}"
        ) from exc

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return path


def export_pawp_tokenizer_to_onnx(model: Any, sample_input: Any, output_path: str | Path, **kwargs: Any) -> Path:
    """Export a PAWP tokenizer-compatible module to ONNX and validate the graph."""
    return _export_to_onnx(model, sample_input, output_path, **kwargs)


def export_noetic_bridge_to_onnx(model: Any, sample_input: Any, output_path: str | Path, **kwargs: Any) -> Path:
    """Export a Noetic bridge module to ONNX and validate the graph."""
    return _export_to_onnx(model, sample_input, output_path, **kwargs)
