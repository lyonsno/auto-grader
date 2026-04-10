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
    return [detection] if detection is not None else []


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
