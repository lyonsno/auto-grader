from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw


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


def _build_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    return build_mc_answer_sheet(
        _template(),
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=17,
    )


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


def _render_marked_page(
    page: dict,
    *,
    marked_labels: dict[str, list[str]],
    scale: int = 4,
) -> np.ndarray:
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


if __name__ == "__main__":
    unittest.main()
