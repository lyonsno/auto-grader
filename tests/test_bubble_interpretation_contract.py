from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _template() -> dict:
    return {
        "slug": "bubble-readback",
        "title": "Bubble Readback",
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
                        "shuffle": False,
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


def _load_modules(test_case: unittest.TestCase):
    try:
        from auto_grader.bubble_interpretation import read_marked_bubble_labels
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.bubble_interpretation.read_marked_bubble_labels(...)` "
            "so the OpenCV lane can interpret filled MC bubbles from a normalized page image."
        )
    from auto_grader.scan_registration import normalize_page_image

    return read_marked_bubble_labels, normalize_page_image


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


class BubbleInterpretationContractTests(unittest.TestCase):
    def test_read_marked_bubble_labels_returns_empty_list_for_blank_question(self) -> None:
        read_marked_bubble_labels, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(_render_marked_page(page, marked_labels={}))
        normalized = normalize_page_image(distorted, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": []},
            "Blank questions should stay visible as empty selections instead of disappearing "
            "from the interpretation surface.",
        )

    def test_read_marked_bubble_labels_returns_filled_labels_from_normalized_page(self) -> None:
        read_marked_bubble_labels, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": ["B"]})
        )
        normalized = normalize_page_image(distorted, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": ["B"]},
            "Bubble readback should recover the filled bubble label after registration "
            "returns the scan to canonical page space.",
        )

    def test_read_marked_bubble_labels_preserves_multiple_marks_as_ambiguous_surface(self) -> None:
        read_marked_bubble_labels, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": ["A", "C"]})
        )
        normalized = normalize_page_image(distorted, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": ["A", "C"]},
            "The first bubble slice should not collapse multiple filled bubbles into "
            "a fake single answer; ambiguity needs to remain visible to later scoring.",
        )


if __name__ == "__main__":
    unittest.main()
