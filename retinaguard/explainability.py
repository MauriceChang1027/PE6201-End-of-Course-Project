"""Class-specific Grad-CAM maps and overlays for accepted vision predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import GRADE_LABELS


@dataclass(frozen=True)
class GradCamResult:
    class_index: int
    layer_name: str
    heatmap: np.ndarray
    heatmap_image: Image.Image
    overlay: Image.Image


class GradCamError(RuntimeError):
    pass


class GradCamExplainer:
    def __init__(self, model):
        import keras

        if keras.backend.backend() != "tensorflow":
            raise GradCamError("Grad-CAM requires the TensorFlow Keras backend.")
        self.model = model
        self.feature_layer = find_feature_layer(model)
        input_shape = model.input_shape
        if len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
            raise GradCamError("The model must have a fixed channels-last image input shape.")
        self.input_size = (int(input_shape[2]), int(input_shape[1]))
        try:
            self.gradient_model = keras.Model(
                model.inputs,
                [connected_layer_output(self.feature_layer), model.output],
            )
        except ValueError as exc:
            raise GradCamError("The feature layer is not connected to the model input.") from exc

    def explain(
        self,
        image: Image.Image | str | Path,
        class_index: int | None = None,
        alpha: float = 0.40,
    ) -> GradCamResult:
        import tensorflow as tf

        original = load_rgb_image(image)
        prepared = np.expand_dims(
            np.asarray(original.resize(self.input_size), dtype=np.float32),
            axis=0,
        )
        tensor = tf.convert_to_tensor(prepared)
        with tf.GradientTape() as tape:
            feature_maps, predictions = self.gradient_model(tensor, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            if not 0 <= class_index < len(GRADE_LABELS):
                raise ValueError("Class index must be between 0 and 4.")
            score = predictions[:, class_index]
        gradients = tape.gradient(score, feature_maps)
        if gradients is None:
            raise GradCamError("The selected feature layer is not connected to the output.")
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0).numpy()
        maximum = float(np.max(heatmap))
        if maximum <= 0 or not np.isfinite(maximum):
            raise GradCamError("The model produced an empty Grad-CAM heatmap.")
        heatmap = heatmap / maximum
        heatmap_image = colourize_heatmap(heatmap, original.size)
        overlay = Image.blend(original, heatmap_image, alpha=alpha)
        return GradCamResult(
            class_index=class_index,
            layer_name=self.feature_layer.name,
            heatmap=heatmap,
            heatmap_image=heatmap_image,
            overlay=overlay,
        )


def find_feature_layer(model):
    for layer in reversed(model.layers):
        shape = getattr(layer.output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer
    raise GradCamError("No four-dimensional convolutional feature layer was found.")


def connected_layer_output(layer):
    nodes = getattr(layer, "_inbound_nodes", ())
    if nodes:
        outputs = nodes[-1].output_tensors
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return layer.output


def load_rgb_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def colourize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    colours = np.stack([red, green, blue], axis=-1)
    image = Image.fromarray(np.uint8(colours * 255))
    return image.resize(size, Image.Resampling.BILINEAR)
