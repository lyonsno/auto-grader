"""Read duplicated page-identity QR payloads from scan images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def read_page_identity_qr_payload(image: np.ndarray) -> str:
    """Decode and resolve the duplicated page-identity QR payload from one scan image."""
    detections = decode_page_identity_qr_codes(image)
    if not detections:
        raise ValueError("No page-identity QR code detected")

    payloads = {detection["payload"] for detection in detections}
    if len(payloads) != 1:
        raise ValueError("Ambiguous page-identity QR payloads detected")
    return next(iter(payloads))


def decode_page_identity_qr_codes(image: np.ndarray) -> list[dict[str, Any]]:
    """Return QR payloads and corner points detected in a scan image."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    detector = cv2.QRCodeDetector()
    detections = _detect_multiple(detector, image)
    if detections:
        return detections

    detection = _detect_single(detector, image)
    if detection is not None:
        return [detection]

    return _detect_tiled_fallback(detector, image)


def _detect_multiple(detector: cv2.QRCodeDetector, image: np.ndarray) -> list[dict[str, Any]]:
    ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    if not ok or points is None:
        return []

    detections: list[dict[str, Any]] = []
    for payload, point_set in zip(decoded_info, points, strict=False):
        normalized_payload = str(payload).strip()
        if not normalized_payload:
            continue
        detections.append(
            {
                "payload": normalized_payload,
                "points": [[float(x), float(y)] for x, y in point_set],
            }
        )
    return detections


def _detect_single(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
) -> dict[str, Any] | None:
    payload, points, _ = detector.detectAndDecode(image)
    normalized_payload = str(payload).strip()
    if not normalized_payload or points is None:
        return None
    return {
        "payload": normalized_payload,
        "points": [[float(x), float(y)] for x, y in points],
    }


def _detect_tiled_fallback(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    window_width = min(width, max(220, int(round(width * 0.18))))
    window_height = min(height, max(220, int(round(height * 0.10))))
    top_limit = min(height, max(window_height, int(round(height * 0.45))))
    step_x = max(1, window_width // 3)
    step_y = max(1, window_height // 3)

    detections: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, int]] = set()

    for top in _sliding_positions(top_limit, window_height, step_y):
        for left in _sliding_positions(width, window_width, step_x):
            bottom = min(height, top + window_height)
            right = min(width, left + window_width)
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue

            crop_detections = _detect_multiple(detector, crop)
            if not crop_detections:
                single = _detect_single(detector, crop)
                crop_detections = [single] if single is not None else []

            for detection in crop_detections:
                translated = _translate_detection(detection, x_offset=left, y_offset=top)
                dedupe_key = _detection_key(translated)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                detections.append(translated)

    return detections


def _translate_detection(
    detection: dict[str, Any],
    *,
    x_offset: int,
    y_offset: int,
) -> dict[str, Any]:
    return {
        "payload": detection["payload"],
        "points": [
            [float(point[0] + x_offset), float(point[1] + y_offset)]
            for point in detection["points"]
        ],
    }


def _detection_key(detection: dict[str, Any]) -> tuple[str, int, int]:
    points = detection["points"]
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return (str(detection["payload"]), int(round(center_x / 12.0)), int(round(center_y / 12.0)))


def _sliding_positions(limit: int, window_size: int, step: int) -> list[int]:
    if limit <= window_size:
        return [0]

    positions = list(range(0, limit - window_size + 1, step))
    final_position = limit - window_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions
