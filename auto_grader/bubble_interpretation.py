from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


_CENTER_DARKNESS_THRESHOLD = 180.0
_CENTER_RADIUS_SCALE = 0.22


def read_marked_bubble_labels(
    image: np.ndarray,
    page: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return the marked bubble labels per question from a normalized page image."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if not isinstance(page, Mapping):
        raise TypeError("page must be a mapping")

    page_width = float(_require_number(page.get("width"), "page.width"))
    page_height = float(_require_number(page.get("height"), "page.height"))
    bubble_regions = page.get("bubble_regions")
    if not isinstance(bubble_regions, list) or len(bubble_regions) == 0:
        raise ValueError("page.bubble_regions must be a non-empty list")

    grayscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_per_point = min(grayscale.shape[1] / page_width, grayscale.shape[0] / page_height)

    grouped_regions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for region in bubble_regions:
        if not isinstance(region, Mapping):
            raise ValueError("page.bubble_regions entries must be mappings")
        question_id = _require_string(region.get("question_id"), "bubble_region.question_id")
        grouped_regions[question_id].append(dict(region))

    marked_labels_by_question: dict[str, list[str]] = {}
    for question_id, regions in grouped_regions.items():
        marked_labels: list[str] = []
        for region in sorted(regions, key=lambda item: (_require_number(item.get("x"), "bubble_region.x"), _require_string(item.get("bubble_label"), "bubble_region.bubble_label"))):
            crop = _crop_region(grayscale, region, pixels_per_point)
            if _bubble_center_mean_intensity(crop) <= _CENTER_DARKNESS_THRESHOLD:
                marked_labels.append(
                    _require_string(region.get("bubble_label"), "bubble_region.bubble_label")
                )
        marked_labels_by_question[question_id] = marked_labels

    return marked_labels_by_question


def _crop_region(
    grayscale: np.ndarray,
    region: Mapping[str, Any],
    pixels_per_point: float,
) -> np.ndarray:
    x = _require_number(region.get("x"), "bubble_region.x")
    y = _require_number(region.get("y"), "bubble_region.y")
    width = _require_number(region.get("width"), "bubble_region.width")
    height = _require_number(region.get("height"), "bubble_region.height")

    left = max(0, int(round(x * pixels_per_point)))
    top = max(0, int(round(y * pixels_per_point)))
    right = min(grayscale.shape[1], int(round((x + width) * pixels_per_point)))
    bottom = min(grayscale.shape[0], int(round((y + height) * pixels_per_point)))
    crop = grayscale[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("bubble region lies outside the normalized page image")
    return crop


def _bubble_center_mean_intensity(crop: np.ndarray) -> float:
    height, width = crop.shape[:2]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = min(width, height) * _CENTER_RADIUS_SCALE

    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius**2
    return float(crop[mask].mean())


def _require_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
