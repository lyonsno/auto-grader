"""Interpret marked MC bubble labels from normalized page images."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


_CENTER_DARKNESS_THRESHOLD = 180.0
_CENTER_RADIUS_SCALE = 0.22
_SURROUNDING_RING_RADIUS_SCALE = 0.42
_CENTER_TO_RING_CONTRAST_THRESHOLD = 24.0
_CENTER_MAX_INTENSITY_FOR_CONTRAST = 235.0
_DARK_PIXEL_THRESHOLD = 210
_MARKED_CENTER_DARK_FRACTION_THRESHOLD = 0.30
_AMBIGUOUS_CENTER_DARK_FRACTION_THRESHOLD = 0.10
_AMBIGUOUS_CENTER_TO_RING_CONTRAST_THRESHOLD = 10.0
_COMPACT_FILL_BBOX_FILL_RATIO_THRESHOLD = 0.34
_ILLEGIBLE_CENTER_DARK_FRACTION_THRESHOLD = 0.20
_ILLEGIBLE_RING_DARK_FRACTION_THRESHOLD = 0.18
_ILLEGIBLE_MIN_CENTER_MEAN_THRESHOLD = 120.0
_ILLEGIBLE_MAX_RING_MEAN_THRESHOLD = 215.0


def read_marked_bubble_labels(
    image: np.ndarray,
    page: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return the clearly marked bubble labels per question from a normalized page image."""
    observations = read_bubble_observations(image, page)
    return {
        question_id: question_observation["marked_bubble_labels"]
        for question_id, question_observation in observations.items()
    }


def read_bubble_observations(
    image: np.ndarray,
    page: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    """Return explicit bubble observations per question from a normalized page image."""
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

    observations_by_question: dict[str, dict[str, list[str]]] = {}
    for question_id, regions in grouped_regions.items():
        question_observation, _ = _analyze_question_regions(
            grayscale,
            regions,
            pixels_per_point=pixels_per_point,
        )
        observations_by_question[question_id] = question_observation

    return observations_by_question


def read_bubble_evidence(
    image: np.ndarray,
    page: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Return per-bubble evidence values for question-level arbitration."""
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

    evidence_by_question: dict[str, dict[str, dict[str, float | str]]] = {}
    for question_id, regions in grouped_regions.items():
        _, question_evidence = _analyze_question_regions(
            grayscale,
            regions,
            pixels_per_point=pixels_per_point,
        )
        evidence_by_question[question_id] = question_evidence
    return evidence_by_question


def _analyze_question_regions(
    grayscale: np.ndarray,
    regions: list[dict[str, Any]],
    *,
    pixels_per_point: float,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float | str]]]:
    question_observation = {
        "marked_bubble_labels": [],
        "ambiguous_bubble_labels": [],
        "illegible_bubble_labels": [],
    }
    question_evidence: dict[str, dict[str, float | str]] = {}
    for region in sorted(
        regions,
        key=lambda item: (
            _require_number(item.get("x"), "bubble_region.x"),
            _require_string(item.get("bubble_label"), "bubble_region.bubble_label"),
        ),
    ):
        crop = _crop_region(grayscale, region, pixels_per_point)
        bubble_label = _require_string(region.get("bubble_label"), "bubble_region.bubble_label")
        metrics = _bubble_metrics(crop)
        classification = _classify_bubble_from_metrics(metrics)
        if classification == "marked":
            question_observation["marked_bubble_labels"].append(bubble_label)
        elif classification == "ambiguous":
            question_observation["ambiguous_bubble_labels"].append(bubble_label)
        elif classification == "illegible":
            question_observation["illegible_bubble_labels"].append(bubble_label)
        question_evidence[bubble_label] = {
            "classification": classification,
            "fill_intent_score": float(
                float(metrics["center_dark_fraction"]) * float(metrics["center_dark_bbox_fill_ratio"])
            ),
            "center_dark_fraction": float(metrics["center_dark_fraction"]),
            "center_dark_bbox_fill_ratio": float(metrics["center_dark_bbox_fill_ratio"]),
            "center_mean": float(metrics["center_mean"]),
            "ring_mean": float(metrics["ring_mean"]),
        }
    return question_observation, question_evidence


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
    return _bubble_metrics(crop)["center_mean"]


def _bubble_center_looks_marked(crop: np.ndarray) -> bool:
    return _classify_bubble(crop) == "marked"


def _classify_bubble(crop: np.ndarray) -> str:
    metrics = _bubble_metrics(crop)
    return _classify_bubble_from_metrics(metrics)


def _classify_bubble_from_metrics(metrics: Mapping[str, float | int]) -> str:
    if _looks_illegible(metrics):
        return "illegible"
    if _looks_marked(metrics):
        return "marked"
    if _looks_ambiguous(metrics):
        return "ambiguous"
    return "blank"


def _looks_marked(metrics: Mapping[str, float | int]) -> bool:
    center_mean = float(metrics["center_mean"])
    ring_mean = float(metrics["ring_mean"])
    center_dark_fraction = float(metrics["center_dark_fraction"])
    center_dark_bbox_fill_ratio = float(metrics["center_dark_bbox_fill_ratio"])
    if center_dark_bbox_fill_ratio < _COMPACT_FILL_BBOX_FILL_RATIO_THRESHOLD:
        return False
    if center_mean <= _CENTER_DARKNESS_THRESHOLD:
        return True
    if center_dark_fraction >= _MARKED_CENTER_DARK_FRACTION_THRESHOLD:
        return True
    return (
        center_mean <= _CENTER_MAX_INTENSITY_FOR_CONTRAST
        and (ring_mean - center_mean) >= _CENTER_TO_RING_CONTRAST_THRESHOLD
    )


def _looks_ambiguous(metrics: Mapping[str, float | int]) -> bool:
    center_mean = float(metrics["center_mean"])
    ring_mean = float(metrics["ring_mean"])
    center_dark_fraction = float(metrics["center_dark_fraction"])
    if center_dark_fraction < _AMBIGUOUS_CENTER_DARK_FRACTION_THRESHOLD:
        return False
    if float(metrics["center_dark_bbox_fill_ratio"]) < _COMPACT_FILL_BBOX_FILL_RATIO_THRESHOLD:
        return False
    return (ring_mean - center_mean) >= _AMBIGUOUS_CENTER_TO_RING_CONTRAST_THRESHOLD


def _looks_illegible(metrics: Mapping[str, float | int]) -> bool:
    return (
        float(metrics["center_mean"]) >= _ILLEGIBLE_MIN_CENTER_MEAN_THRESHOLD
        and float(metrics["center_dark_fraction"]) >= _ILLEGIBLE_CENTER_DARK_FRACTION_THRESHOLD
        and float(metrics["ring_dark_fraction"]) >= _ILLEGIBLE_RING_DARK_FRACTION_THRESHOLD
        and float(metrics["ring_mean"]) <= _ILLEGIBLE_MAX_RING_MEAN_THRESHOLD
    )


def _bubble_metrics(crop: np.ndarray) -> dict[str, float | int]:
    center_mask, ring_mask = _bubble_masks(crop)
    dark_mask = crop <= _DARK_PIXEL_THRESHOLD
    center_pixels = crop[center_mask]
    ring_pixels = crop[ring_mask]
    center_dark_mask = dark_mask & center_mask
    center_dark_pixels = center_dark_mask[center_mask]
    ring_dark_pixels = dark_mask[ring_mask]

    center_mean = float(center_pixels.mean())
    ring_mean = float(ring_pixels.mean()) if ring_pixels.size else 255.0
    center_dark_fraction = float(center_dark_pixels.mean()) if center_dark_pixels.size else 0.0
    ring_dark_fraction = float(ring_dark_pixels.mean()) if ring_dark_pixels.size else 0.0
    center_dark_bbox_fill_ratio = _dark_bbox_fill_ratio(center_dark_mask)

    return {
        "center_mean": center_mean,
        "ring_mean": ring_mean,
        "center_dark_fraction": center_dark_fraction,
        "ring_dark_fraction": ring_dark_fraction,
        "center_dark_bbox_fill_ratio": center_dark_bbox_fill_ratio,
        "border_touch_sides": _dark_border_touch_sides(dark_mask),
    }


def _bubble_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = crop.shape[:2]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    min_dimension = min(width, height)
    center_radius = min_dimension * _CENTER_RADIUS_SCALE
    ring_radius = min_dimension * _SURROUNDING_RING_RADIUS_SCALE

    y_indices, x_indices = np.ogrid[:height, :width]
    distance_squared = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
    center_mask = distance_squared <= center_radius**2
    ring_mask = (distance_squared <= ring_radius**2) & ~center_mask
    return center_mask, ring_mask


def _dark_border_touch_sides(dark_mask: np.ndarray) -> int:
    touches = 0
    if dark_mask[0, :].any():
        touches += 1
    if dark_mask[-1, :].any():
        touches += 1
    if dark_mask[:, 0].any():
        touches += 1
    if dark_mask[:, -1].any():
        touches += 1
    return touches


def _dark_bbox_fill_ratio(center_dark_mask: np.ndarray) -> float:
    if not center_dark_mask.any():
        return 0.0
    ys, xs = np.where(center_dark_mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    bbox_area = width * height
    if bbox_area == 0:
        return 0.0
    return float(len(xs) / bbox_area)


def _require_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
