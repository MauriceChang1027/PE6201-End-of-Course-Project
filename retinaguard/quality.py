from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


MIN_RESOLUTION = 224
MIN_BRIGHTNESS = 0.08
MAX_BRIGHTNESS = 0.90
MIN_CONTRAST = 0.035
MIN_SHARPNESS = 0.00035
MIN_FIELD_CONTRAST = 0.025
MIN_COLOURFULNESS = 0.035
MIN_RED_DOMINANCE = 0.015


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    issues: tuple[QualityIssue, ...]
    metrics: dict[str, float | int]


def assess_image_quality(image: Image.Image | str | Path) -> QualityAssessment:
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    else:
        image = image.convert("RGB")

    width, height = image.size
    sample = np.asarray(image.resize((256, 256)), dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(sample, -1, 0)
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
    coordinates = np.linspace(-1.0, 1.0, 256)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(horizontal**2 + vertical**2)
    centre = radius <= 0.45
    corners = radius >= 1.05
    centre_gray = grayscale[centre]
    corner_gray = grayscale[corners]
    laplacian = (
        -4 * grayscale[1:-1, 1:-1]
        + grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
    )
    sharpness = float(np.var(laplacian[centre[1:-1, 1:-1]]))
    centre_brightness = float(np.mean(centre_gray))
    contrast = float(np.std(centre_gray))
    field_contrast = centre_brightness - float(np.mean(corner_gray))
    colourfulness = float(np.mean(np.max(sample, axis=2) - np.min(sample, axis=2)))
    red_dominance = float(np.mean((red - (green + blue) / 2)[centre]))

    metrics = {
        "width": width,
        "height": height,
        "centre_brightness": centre_brightness,
        "centre_contrast": contrast,
        "sharpness": sharpness,
        "field_contrast": field_contrast,
        "colourfulness": colourfulness,
        "red_dominance": red_dominance,
    }
    issues = []
    if min(width, height) < MIN_RESOLUTION:
        issues.append(
            QualityIssue("resolution", "Image resolution is below 224 x 224 pixels.")
        )
    if centre_brightness < MIN_BRIGHTNESS:
        issues.append(QualityIssue("underexposed", "The retinal field appears too dark."))
    if centre_brightness > MAX_BRIGHTNESS:
        issues.append(QualityIssue("overexposed", "The retinal field appears too bright."))
    if contrast < MIN_CONTRAST:
        issues.append(
            QualityIssue("low_contrast", "The image has insufficient local contrast.")
        )
    if sharpness < MIN_SHARPNESS:
        issues.append(
            QualityIssue("blur", "The image appears too blurred for reliable analysis.")
        )
    if (
        field_contrast < MIN_FIELD_CONTRAST
        or colourfulness < MIN_COLOURFULNESS
        or red_dominance < MIN_RED_DOMINANCE
    ):
        issues.append(
            QualityIssue(
                "fundus_plausibility",
                "A clear reddish retinal field on a darker background was not detected.",
            )
        )
    return QualityAssessment(passed=not issues, issues=tuple(issues), metrics=metrics)
