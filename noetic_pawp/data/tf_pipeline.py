from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetConfig:
    root_dir: Path
    split: Split
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 8
    shuffle: bool = True
    cache: bool = False
    prefetch: bool = True
    seed: int = 42


def _require_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for noetic_pawp.data.tf_pipeline. Install with `pip install tensorflow`."
        ) from exc
    return tf


def _validate_config(config: DatasetConfig) -> Path:
    if config.split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split '{config.split}'. Use one of train/val/test.")
    root = config.root_dir.expanduser().resolve()
    split_dir = root / config.split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    return split_dir


def _build_dataset(config: DatasetConfig, parser: Callable):
    tf = _require_tensorflow()
    split_dir = _validate_config(config)
    files = sorted(str(p) for p in split_dir.glob("*.jpg")) + sorted(str(p) for p in split_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No image files found in {split_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(files)
    if config.shuffle:
        dataset = dataset.shuffle(buffer_size=len(files), seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    if config.cache:
        dataset = dataset.cache()
    dataset = dataset.batch(config.batch_size, drop_remainder=False)
    if config.prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _parse_image_only(image_size: tuple[int, int]):
    tf = _require_tensorflow()

    def _parser(image_path):
        image_bytes = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return {"image": image, "path": image_path}

    return _parser


def create_nyuv2_dataset(
    root_dir: str | Path,
    split: Split = "train",
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
    shuffle: bool = True,
    cache: bool = False,
    prefetch: bool = True,
):
    """Create a tf.data pipeline for NYUv2-style folder structure.

    Expected layout: <root>/<split>/*.jpg|*.png
    """
    cfg = DatasetConfig(Path(root_dir), split, image_size, batch_size, shuffle, cache, prefetch)
    return _build_dataset(cfg, _parse_image_only(image_size))


def create_kitti_dataset(
    root_dir: str | Path,
    split: Split = "train",
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
    shuffle: bool = True,
    cache: bool = False,
    prefetch: bool = True,
):
    """Create a tf.data pipeline for KITTI-style image batches."""
    cfg = DatasetConfig(Path(root_dir), split, image_size, batch_size, shuffle, cache, prefetch)
    return _build_dataset(cfg, _parse_image_only(image_size))


def create_multimodal_dataset(
    root_dir: str | Path,
    split: Split = "train",
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
    shuffle: bool = True,
    cache: bool = False,
    prefetch: bool = True,
):
    """Create dataset returning image and placeholder modality fields for extension."""
    tf = _require_tensorflow()

    def _parser(image_path):
        payload = _parse_image_only(image_size)(image_path)
        payload["text"] = tf.constant("", dtype=tf.string)
        payload["sensor"] = tf.zeros([4], dtype=tf.float32)
        return payload

    cfg = DatasetConfig(Path(root_dir), split, image_size, batch_size, shuffle, cache, prefetch)
    return _build_dataset(cfg, _parser)
