from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw
import qrcode


def _template() -> dict:
    return {
        "slug": "mc-extraction",
        "title": "MC Extraction",
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


def _multi_question_template() -> dict:
    return {
        "slug": "mc-degraded-page-smoke",
        "title": "MC Degraded Page Smoke",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Q1",
                        "choices": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
                        "correct": "B",
                        "shuffle": True,
                    },
                    {
                        "id": "mc-2",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Q2",
                        "choices": {"A": "A2", "B": "B2", "C": "C2", "D": "D2"},
                        "correct": "C",
                        "shuffle": True,
                    },
                    {
                        "id": "mc-3",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Q3",
                        "choices": {"A": "A3", "B": "B3", "C": "C3", "D": "D3"},
                        "correct": "D",
                        "shuffle": True,
                    },
                    {
                        "id": "mc-4",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Q4",
                        "choices": {"A": "A4", "B": "B4", "C": "C4", "D": "D4"},
                        "correct": "A",
                        "shuffle": True,
                    },
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


def _build_multi_question_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    return build_mc_answer_sheet(
        _multi_question_template(),
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=17,
    )


def _build_paper_calibration_artifact() -> dict:
    from auto_grader.paper_calibration_packet import build_mc_paper_calibration_packet

    return build_mc_paper_calibration_packet()["artifact"]


def _load_extraction_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_page_extraction import extract_scored_mc_page
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_page_extraction.extract_scored_mc_page(...)` so the "
            "matched-page OpenCV path can return one honest extraction bundle instead "
            "of forcing callers to stitch registration, bubble readback, and scoring together."
        )
    return extract_scored_mc_page


def _load_readback_module(test_case: unittest.TestCase):
    try:
        from auto_grader.scan_readback import read_page_identity_qr_payload
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.scan_readback.read_page_identity_qr_payload(...)` so "
            "the degraded-page extraction contract can prove QR identity survives the "
            "same scan damage the scoring path is asked to tolerate."
        )
    return read_page_identity_qr_payload


def _render_marked_page(
    page: dict,
    *,
    marked_labels: dict[str, list[str]],
    light_marked_labels: dict[str, list[str]] | None = None,
    scale: int = 4,
) -> np.ndarray:
    width = int(page["width"] * scale)
    height = int(page["height"] * scale)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    light_marked_labels = light_marked_labels or {}

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
    for qr_code in page.get("identity_qr_codes", []):
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

    for bubble in page["bubble_regions"]:
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        draw.ellipse([left, top, right, bottom], outline="black", width=max(2, scale))

        if bubble["bubble_label"] in marked_labels.get(bubble["question_id"], []):
            inset = 0.22 * bubble["width"] * scale
            draw.ellipse(
                [left + inset, top + inset, right - inset, bottom - inset],
                fill="black",
            )
        elif bubble["bubble_label"] in light_marked_labels.get(bubble["question_id"], []):
            inset = 0.33 * bubble["width"] * scale
            draw.ellipse(
                [left + inset, top + inset, right - inset, bottom - inset],
                fill=(188, 188, 188),
            )

    return np.array(canvas)


def _render_dominant_plus_secondary_page(
    page: dict,
    *,
    question_id: str,
    dominant_bubble_label: str,
    secondary_bubble_label: str,
    secondary_gray: int = 198,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(
        _render_marked_page(page, marked_labels={question_id: [dominant_bubble_label]}, scale=scale)
    )
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != secondary_bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        inset = 0.33 * bubble["width"] * scale
        draw.ellipse(
            [left + inset, top + inset, right - inset, bottom - inset],
            fill=(secondary_gray, secondary_gray, secondary_gray),
        )
        break

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


def _heavily_degrade_scan(image: np.ndarray) -> np.ndarray:
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
            [90, 45],
            [width - 120, 35],
            [55, height - 95],
            [width - 25, height - 55],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    distorted = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    blurred = cv2.GaussianBlur(distorted, (7, 7), 0)
    noisy = blurred.astype(np.int16)
    noisy += np.random.default_rng(7).normal(0, 16, size=noisy.shape).astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


class McPageExtractionContractTests(unittest.TestCase):
    def test_extract_scored_mc_page_returns_page_identity_readback_and_scores(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        question_id = "mc-1"
        correct_bubble = artifact["answer_key"][question_id]["correct_bubble_label"]

        distorted = _perspective_distort(
            _render_marked_page(page, marked_labels={question_id: [correct_bubble]})
        )

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(
            extracted["page_number"],
            page["page_number"],
            "The matched-page extraction bundle should preserve the page identity callers already have.",
        )
        self.assertEqual(extracted["fallback_page_code"], page["fallback_page_code"])
        self.assertIn("bubble_evidence", extracted)
        self.assertIn(question_id, extracted["bubble_evidence"])
        self.assertIn(correct_bubble, extracted["bubble_evidence"][question_id])
        self.assertEqual(extracted["marked_bubble_labels"], {question_id: [correct_bubble]})
        self.assertEqual(extracted["scored_questions"][question_id]["status"], "correct")
        self.assertFalse(extracted["scored_questions"][question_id]["review_required"])
        self.assertEqual(
            extracted["normalized_image"].shape[1] / extracted["normalized_image"].shape[0],
            page["width"] / page["height"],
            "The extraction bundle should expose the normalized page image for downstream consumers.",
        )

    def test_extract_scored_mc_page_keeps_multiple_marks_visible(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": ["A", "D"]})
        )

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(extracted["marked_bubble_labels"], {"mc-1": ["A", "D"]})
        self.assertEqual(
            extracted["scored_questions"]["mc-1"]["status"],
            "multiple_marked",
            "The page-level extraction surface should preserve ambiguous marks all the way "
            "through the packaged scoring result.",
        )
        self.assertTrue(extracted["scored_questions"]["mc-1"]["review_required"])

    def test_extract_scored_mc_page_prefers_one_dominant_mark_over_weaker_secondary_trace(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]
        weaker_bubble = next(
            choice["bubble_label"]
            for choice in artifact["answer_key"]["mc-1"]["choices"]
            if choice["bubble_label"] != correct_bubble
        )

        distorted = _perspective_distort(
            _render_dominant_plus_secondary_page(
                page,
                question_id="mc-1",
                dominant_bubble_label=correct_bubble,
                secondary_bubble_label=weaker_bubble,
            )
        )

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(
            extracted["scored_questions"]["mc-1"]["status"],
            "correct",
            "One clearly stronger filled bubble should remain machine-gradable even if "
            "a weaker secondary trace survives readback on another bubble.",
        )
        self.assertEqual(extracted["scored_questions"]["mc-1"]["resolved_bubble_labels"], [correct_bubble])
        self.assertEqual(
            extracted["scored_questions"]["mc-1"]["ignored_incidental_bubble_labels"],
            [weaker_bubble],
        )
        self.assertFalse(extracted["scored_questions"]["mc-1"]["review_required"])

    def test_extract_scored_mc_page_preserves_ambiguous_observation_surface(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        from tests.test_bubble_interpretation_contract import _render_patchy_center_mark

        distorted = _perspective_distort(
            _render_patchy_center_mark(page, question_id="mc-1", bubble_label="B")
        )

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(
            extracted["bubble_observations"],
            {
                "mc-1": {
                    "marked_bubble_labels": [],
                    "ambiguous_bubble_labels": ["B"],
                    "illegible_bubble_labels": [],
                }
            },
            "Matched-page extraction should expose explicit bubble observations so "
            "borderline fills do not disappear into a flat blank surface.",
        )
        self.assertEqual(extracted["marked_bubble_labels"], {"mc-1": []})
        self.assertEqual(extracted["scored_questions"]["mc-1"]["status"], "ambiguous_mark")
        self.assertTrue(extracted["scored_questions"]["mc-1"]["review_required"])

    def test_extract_scored_mc_page_keeps_blank_question_explicit(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(_render_marked_page(page, marked_labels={}))

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(extracted["marked_bubble_labels"], {"mc-1": []})
        self.assertEqual(extracted["scored_questions"]["mc-1"]["status"], "blank")
        self.assertFalse(extracted["scored_questions"]["mc-1"]["review_required"])

    def test_extract_scored_mc_page_scores_only_questions_present_on_that_page(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        artifact = _build_paper_calibration_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_marked_page(page, marked_labels={"cal-01": ["B"]})
        )

        extracted = extract_scored_mc_page(distorted, page, artifact["answer_key"])

        self.assertEqual(
            sorted(extracted["scored_questions"]),
            ["cal-01", "cal-02", "cal-03", "cal-04", "cal-05", "cal-06"],
            "Matched-page extraction should score only the question ids whose bubble "
            "regions actually live on that page, not inflate blanks for the rest of "
            "the exam answer key.",
        )
        self.assertEqual(len(extracted["marked_bubble_labels"]), 6)

    def test_extract_scored_mc_page_handles_mixed_outcomes_on_one_degraded_page(self) -> None:
        extract_scored_mc_page = _load_extraction_module(self)
        read_page_identity_qr_payload = _load_readback_module(self)
        artifact = _build_multi_question_artifact()
        page = artifact["pages"][0]

        incorrect_for_mc3 = next(
            choice["bubble_label"]
            for choice in artifact["answer_key"]["mc-3"]["choices"]
            if choice["bubble_label"] != artifact["answer_key"]["mc-3"]["correct_bubble_label"]
        )

        rendered = _render_marked_page(
            page,
            marked_labels={
                "mc-2": ["A", "B"],
                "mc-3": [incorrect_for_mc3],
            },
            light_marked_labels={
                "mc-1": [artifact["answer_key"]["mc-1"]["correct_bubble_label"]],
            },
        )
        degraded = _heavily_degrade_scan(rendered)

        page_payload = read_page_identity_qr_payload(degraded)
        extracted = extract_scored_mc_page(degraded, page, artifact["answer_key"])

        self.assertEqual(
            page_payload,
            page["fallback_page_code"],
            "The degraded-page smoke should prove QR identity survives the same blur/noise "
            "stack as the matched-page extraction path.",
        )
        self.assertEqual(
            extracted["fallback_page_code"],
            page["fallback_page_code"],
        )
        self.assertEqual(
            extracted["marked_bubble_labels"],
            {
                "mc-1": [artifact["answer_key"]["mc-1"]["correct_bubble_label"]],
                "mc-2": ["A", "B"],
                "mc-3": [incorrect_for_mc3],
                "mc-4": [],
            },
            "The degraded-page extraction bundle should preserve mixed MC outcomes "
            "on one page instead of only working when every question follows the same path.",
        )
        self.assertEqual(extracted["scored_questions"]["mc-1"]["status"], "correct")
        self.assertEqual(extracted["scored_questions"]["mc-2"]["status"], "multiple_marked")
        self.assertEqual(extracted["scored_questions"]["mc-3"]["status"], "incorrect")
        self.assertEqual(extracted["scored_questions"]["mc-4"]["status"], "blank")
        self.assertTrue(extracted["scored_questions"]["mc-2"]["review_required"])
        self.assertFalse(extracted["scored_questions"]["mc-1"]["review_required"])
        self.assertFalse(extracted["scored_questions"]["mc-3"]["review_required"])
        self.assertFalse(extracted["scored_questions"]["mc-4"]["review_required"])


if __name__ == "__main__":
    unittest.main()
