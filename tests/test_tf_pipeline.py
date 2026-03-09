from __future__ import annotations

import pytest


tf = pytest.importorskip("tensorflow")

from noetic_pawp.data.rive_tf import RIVEPreprocessor
from noetic_pawp.data.tf_pipeline import create_kitti_dataset, create_nyuv2_dataset


def _build_fake_dataset(root, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    image = tf.zeros([16, 16, 3], dtype=tf.uint8)
    encoded = tf.io.encode_png(image)
    tf.io.write_file(str(split_dir / "sample.png"), encoded)


def test_create_nyuv2_dataset(tmp_path):
    _build_fake_dataset(tmp_path, "train")
    ds = create_nyuv2_dataset(tmp_path, split="train", batch_size=1, shuffle=False)
    batch = next(iter(ds))
    assert batch["image"].shape[0] == 1


def test_create_kitti_dataset(tmp_path):
    _build_fake_dataset(tmp_path, "train")
    ds = create_kitti_dataset(tmp_path, split="train", batch_size=1, shuffle=False)
    batch = next(iter(ds))
    assert batch["image"].shape[-1] == 3


def test_rive_preprocessor_shapes():
    layer = RIVEPreprocessor()
    x = tf.zeros([2, 32, 32, 3], dtype=tf.float32)
    out = layer(x)
    assert out["radial"].shape == (2, 32, 32, 1)
    assert out["frustum"].shape == (2, 32, 32, 1)
    assert out["rive_coeffs"].shape == (2, 9)
