from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def normalize_page_image(image: np.ndarray, page: Mapping[str, Any]) -> np.ndarray:
    """Warp a skewed page image back into canonical page space using corner markers."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if not isinstance(page, Mapping):
        raise TypeError("page must be a mapping")

    page_width = float(_require_number(page.get("width"), "page.width"))
    page_height = float(_require_number(page.get("height"), "page.height"))
    registration_markers = page.get("registration_markers")
    if not isinstance(registration_markers, list) or len(registration_markers) != 4:
        raise ValueError("page.registration_markers must contain exactly four markers")

    detected_markers = _detect_registration_marker_boxes(image)
    image_height, image_width = image.shape[:2]
    pixels_per_point = min(image_width / page_width, image_height / page_height)
    output_width = max(int(round(page_width * pixels_per_point)), int(page_width))
    output_height = max(int(round(page_height * pixels_per_point)), int(page_height))
    source_points = np.float32(
        [
            point
            for marker_id in ("top_left", "top_right", "bottom_left", "bottom_right")
            for point in detected_markers[marker_id]
        ]
    )
    target_points = np.float32(
        [
            point
            for marker_id in ("top_left", "top_right", "bottom_left", "bottom_right")
            for point in _scaled_marker_box(
                _require_marker(registration_markers, marker_id),
                pixels_per_point,
            )
        ]
    )

    transform, _ = cv2.findHomography(source_points, target_points, method=0)
    if transform is None:
        raise ValueError("Could not compute registration homography from marker geometry")
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _detect_registration_marker_boxes(image: np.ndarray) -> dict[str, list[tuple[float, float]]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        (center_x, center_y), (width, height), _ = cv2.minAreaRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect_ratio = max(width, height) / min(width, height)
        box_area = width * height
        fill_ratio = area / box_area if box_area else 0.0
        if aspect_ratio > 1.35 or fill_ratio < 0.7:
            continue
        candidates.append(
            {
                "center": (float(center_x), float(center_y)),
                "box_points": _order_points(cv2.boxPoints(cv2.minAreaRect(contour))),
            }
        )

    if len(candidates) < 4:
        raise ValueError("Could not detect four registration marker candidates")

    image_height, image_width = gray.shape[:2]
    corner_targets = {
        "top_left": (0.0, 0.0),
        "top_right": (float(image_width), 0.0),
        "bottom_left": (0.0, float(image_height)),
        "bottom_right": (float(image_width), float(image_height)),
    }

    assignments: dict[str, list[tuple[float, float]]] = {}
    used_indices: set[int] = set()
    for marker_id, corner in corner_targets.items():
        distances = [
            (
                (candidate["center"][0] - corner[0]) ** 2 + (candidate["center"][1] - corner[1]) ** 2,
                index,
                candidate,
            )
            for index, candidate in enumerate(candidates)
            if index not in used_indices
        ]
        if not distances:
            raise ValueError("Could not assign all registration markers uniquely")
        _, index, candidate = min(distances, key=lambda item: item[0])
        used_indices.add(index)
        assignments[marker_id] = candidate["box_points"]

    return assignments


def _require_marker(
    registration_markers: list[Mapping[str, Any]],
    marker_id: str,
) -> Mapping[str, Any]:
    for marker in registration_markers:
        if marker.get("marker_id") == marker_id:
            return marker
    raise ValueError(f"Missing registration marker {marker_id!r}")


def _marker_center(marker: Mapping[str, Any]) -> tuple[float, float]:
    x = _require_number(marker.get("x"), "registration_marker.x")
    y = _require_number(marker.get("y"), "registration_marker.y")
    width = _require_number(marker.get("width"), "registration_marker.width")
    height = _require_number(marker.get("height"), "registration_marker.height")
    return (float(x + (width / 2)), float(y + (height / 2)))


def _scaled_marker_box(
    marker: Mapping[str, Any],
    pixels_per_point: float,
) -> list[tuple[float, float]]:
    x = _require_number(marker.get("x"), "registration_marker.x")
    y = _require_number(marker.get("y"), "registration_marker.y")
    width = _require_number(marker.get("width"), "registration_marker.width")
    height = _require_number(marker.get("height"), "registration_marker.height")
    return [
        (x * pixels_per_point, y * pixels_per_point),
        ((x + width) * pixels_per_point, y * pixels_per_point),
        (x * pixels_per_point, (y + height) * pixels_per_point),
        ((x + width) * pixels_per_point, (y + height) * pixels_per_point),
    ]


def _order_points(points: np.ndarray) -> list[tuple[float, float]]:
    raw_points = [(float(x), float(y)) for x, y in points]
    top_left = min(raw_points, key=lambda point: point[0] + point[1])
    bottom_right = max(raw_points, key=lambda point: point[0] + point[1])
    top_right = min(raw_points, key=lambda point: point[1] - point[0])
    bottom_left = max(raw_points, key=lambda point: point[1] - point[0])
    return [top_left, top_right, bottom_left, bottom_right]


def _require_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return value
