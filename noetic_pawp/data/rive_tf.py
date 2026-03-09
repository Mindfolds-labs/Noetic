from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except Exception as _TF_IMPORT_ERROR:  # pragma: no cover
    tf = None  # type: ignore[assignment]
else:
    _TF_IMPORT_ERROR = None


class RIVEPreprocessor((tf.keras.layers.Layer if tf is not None else object)):  # type: ignore[misc]
    """TensorFlow implementation of a simplified RIVE preprocessor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required. Install with `pip install tensorflow`.") from _TF_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    def call(self, images):
        if tf is None:  # pragma: no cover
            raise RuntimeError("TensorFlow is unavailable.")
        if images.shape.rank != 4:
            raise ValueError("Expected images in NHWC format [batch, height, width, channels].")

        images = tf.cast(images, tf.float32)
        shape = tf.shape(images)
        batch, height, width = shape[0], shape[1], shape[2]

        y = tf.linspace(-1.0, 1.0, height)
        x = tf.linspace(-1.0, 1.0, width)
        yy, xx = tf.meshgrid(y, x, indexing="ij")
        radial = tf.sqrt(tf.square(xx) + tf.square(yy))
        radial = tf.expand_dims(tf.expand_dims(radial, axis=0), axis=-1)
        radial = tf.tile(radial, [batch, 1, 1, 1])

        frustum = tf.clip_by_value((yy + 1.0) / 2.0, 0.0, 1.0)
        frustum = tf.expand_dims(tf.expand_dims(frustum, axis=0), axis=-1)
        frustum = tf.tile(frustum, [batch, 1, 1, 1])

        mean = tf.reduce_mean(images, axis=[1, 2])
        std = tf.math.reduce_std(images, axis=[1, 2])
        maxv = tf.reduce_max(images, axis=[1, 2])
        rive_coeffs = tf.concat([mean, std, maxv], axis=-1)
        return {"radial": radial, "frustum": frustum, "rive_coeffs": rive_coeffs}

    def compute_output_shape(self, input_shape) -> dict[str, Any]:
        if tf is None:  # pragma: no cover
            raise RuntimeError("TensorFlow is unavailable.")
        batch, height, width, channels = input_shape
        return {
            "radial": tf.TensorShape((batch, height, width, 1)),
            "frustum": tf.TensorShape((batch, height, width, 1)),
            "rive_coeffs": tf.TensorShape((batch, channels * 3)),
        }
