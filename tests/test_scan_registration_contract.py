from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw
import qrcode


def _template() -> dict:
    return {
        "slug": "qr-registration",
        "title": "QR Registration",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Which species is elemental oxygen?",
                        "choices": {
                            "A": "CO2",
                            "B": "O2",
                            "C": "H2O",
                            "D": "NaCl",
                        },
                        "correct": "B",
                        "shuffle": True,
                    }
                ],
            }
        ],
    }


def _build_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    return build_mc_answer_sheet(
        _template(),
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=17,
    )


def _load_registration_module(test_case: unittest.TestCase):
    try:
        from auto_grader.scan_registration import normalize_page_image
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.scan_registration.normalize_page_image(...)` so "
            "the OpenCV lane can normalize skewed scans back into canonical page space."
        )
    return normalize_page_image


def _render_synthetic_page(page: dict, *, scale: int = 3) -> np.ndarray:
    width = int(page["width"] * scale)
    height = int(page["height"] * scale)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for marker in page["registration_markers"]:
        draw.rectangle(
            [
                marker["x"] * scale,
                marker["y"] * scale,
                (marker["x"] + marker["width"]) * scale,
                (marker["y"] + marker["height"]) * scale,
            ],
            fill="black",
        )

    correction_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    for qr_code in page["identity_qr_codes"]:
        qr = qrcode.QRCode(
            border=qr_code["border_modules"],
            error_correction=correction_levels[qr_code["error_correction"]],
            box_size=1,
        )
        qr.add_data(qr_code["payload"])
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_image = qr_image.resize(
            (int(qr_code["width"] * scale), int(qr_code["height"] * scale)),
            Image.Resampling.NEAREST,
        )
        canvas.paste(qr_image, (int(qr_code["x"] * scale), int(qr_code["y"] * scale)))

    return np.array(canvas)


def _perspective_distort(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]
    )
    destination = np.float32(
        [
            [70, 40],
            [width - 90, 25],
            [45, height - 65],
            [width - 20, height - 35],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


class ScanRegistrationContractTests(unittest.TestCase):
    def test_normalize_page_image_restores_canonical_page_space_from_skewed_scan(self) -> None:
        normalize_page_image = _load_registration_module(self)

        artifact = _build_artifact()
        page = artifact["pages"][0]
        distorted = _perspective_distort(_render_synthetic_page(page))

        normalized = normalize_page_image(distorted, page)
        pixels_per_point = normalized.shape[1] / page["width"]

        self.assertGreaterEqual(
            normalized.shape[1],
            page["width"],
            "Registration should preserve enough raster resolution for later QR and "
            "bubble readback instead of collapsing the page down below 1 pixel per point.",
        )
        self.assertAlmostEqual(
            normalized.shape[1] / normalized.shape[0],
            page["width"] / page["height"],
            places=2,
            msg=(
                "Registration should return a canonical-aspect raster even when it keeps "
                "the scan's practical resolution."
            ),
        )
        for marker in page["registration_markers"]:
            marker_region = normalized[
                int(marker["y"] * pixels_per_point) : int((marker["y"] + marker["height"]) * pixels_per_point),
                int(marker["x"] * pixels_per_point) : int((marker["x"] + marker["width"]) * pixels_per_point),
            ]
            self.assertLess(
                float(marker_region.mean()),
                80.0,
                "Registration markers should land back near their scaled canonical page-space "
                "positions after normalization.",
            )
        for qr_code in page["identity_qr_codes"]:
            qr_region = normalized[
                int(qr_code["y"] * pixels_per_point) : int((qr_code["y"] + qr_code["height"]) * pixels_per_point),
                int(qr_code["x"] * pixels_per_point) : int((qr_code["x"] + qr_code["width"]) * pixels_per_point),
            ]
            self.assertLess(
                float(qr_region.mean()),
                245.0,
                "The normalized page should bring the identity QR markers back into their "
                "expected canonical neighborhood even before downstream decode-hardening work.",
            )

    def test_normalize_page_image_fails_when_registration_markers_are_missing(self) -> None:
        normalize_page_image = _load_registration_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        with self.assertRaisesRegex(ValueError, "registration marker"):
            normalize_page_image(np.full((600, 900, 3), 255, dtype=np.uint8), page)


if __name__ == "__main__":
    unittest.main()
