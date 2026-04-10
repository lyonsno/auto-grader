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
    try:
        from auto_grader.bubble_interpretation import read_bubble_observations
    except ImportError:
        test_case.fail(
            "Add `auto_grader.bubble_interpretation.read_bubble_observations(...)` so "
            "the OpenCV lane can preserve hostile and borderline bubble observations "
            "instead of silently flattening everything into marked-or-blank."
        )
    from auto_grader.scan_registration import normalize_page_image

    return read_marked_bubble_labels, read_bubble_observations, normalize_page_image


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


def _render_pencil_like_mark(
    page: dict,
    *,
    question_id: str,
    bubble_label: str,
    scale: int = 4,
    gray: int = 150,
) -> np.ndarray:
    canvas = Image.fromarray(_render_marked_page(page, marked_labels={}, scale=scale))
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        inset = 0.28 * bubble["width"] * scale
        draw.ellipse(
            [left + inset, top + inset, right - inset, bottom - inset],
            fill=(gray, gray, gray),
        )
        break

    return np.array(canvas)


def _render_light_pencil_like_mark(
    page: dict,
    *,
    question_id: str,
    bubble_label: str,
    scale: int = 4,
    gray: int = 188,
) -> np.ndarray:
    canvas = Image.fromarray(_render_marked_page(page, marked_labels={}, scale=scale))
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        inset = 0.33 * bubble["width"] * scale
        draw.ellipse(
            [left + inset, top + inset, right - inset, bottom - inset],
            fill=(gray, gray, gray),
        )
        break

    return np.array(canvas)


def _render_edge_smudge(
    page: dict,
    *,
    question_id: str,
    bubble_label: str,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(_render_marked_page(page, marked_labels={}, scale=scale))
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        width = bubble["width"] * scale
        height = bubble["height"] * scale
        draw.ellipse(
            [
                left + 1,
                top + (0.35 * height),
                left + (0.28 * width),
                top + (0.7 * height),
            ],
            fill="black",
        )
        break

    return np.array(canvas)


def _render_patchy_center_mark(
    page: dict,
    *,
    question_id: str,
    bubble_label: str,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(_render_marked_page(page, marked_labels={}, scale=scale))
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        width = bubble["width"] * scale
        height = bubble["height"] * scale
        draw.ellipse(
            [
                left + (0.37 * width),
                top + (0.35 * height),
                left + (0.53 * width),
                top + (0.51 * height),
            ],
            fill=(198, 198, 198),
        )
        draw.ellipse(
            [
                left + (0.49 * width),
                top + (0.47 * height),
                left + (0.62 * width),
                top + (0.60 * height),
            ],
            fill=(206, 206, 206),
        )
        break

    return np.array(canvas)


def _render_illegible_scratchout(
    page: dict,
    *,
    question_id: str,
    bubble_label: str,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(_render_marked_page(page, marked_labels={}, scale=scale))
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        width = max(3, scale + 1)
        draw.line(
            [
                left - (0.08 * (right - left)),
                top + (0.18 * (bottom - top)),
                right + (0.08 * (right - left)),
                bottom - (0.18 * (bottom - top)),
            ],
            fill="black",
            width=width,
        )
        draw.line(
            [
                left + (0.12 * (right - left)),
                bottom + (0.04 * (bottom - top)),
                right - (0.08 * (right - left)),
                top - (0.05 * (bottom - top)),
            ],
            fill="black",
            width=width,
        )
        draw.line(
            [
                left - (0.04 * (right - left)),
                top + (0.58 * (bottom - top)),
                right + (0.06 * (right - left)),
                top + (0.38 * (bottom - top)),
            ],
            fill="black",
            width=width,
        )
        break

    return np.array(canvas)


def _render_correct_plus_stray_glance(
    page: dict,
    *,
    question_id: str,
    correct_bubble_label: str,
    stray_bubble_label: str,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(
        _render_marked_page(page, marked_labels={question_id: [correct_bubble_label]}, scale=scale)
    )
    draw = ImageDraw.Draw(canvas)

    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != question_id or bubble["bubble_label"] != stray_bubble_label:
            continue
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        width = bubble["width"] * scale
        height = bubble["height"] * scale
        draw.line(
            [
                left + (0.03 * width),
                top + (0.56 * height),
                left + (0.31 * width),
                top + (0.28 * height),
            ],
            fill="black",
            width=max(2, scale),
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


class BubbleInterpretationContractTests(unittest.TestCase):
    def test_read_marked_bubble_labels_returns_empty_list_for_blank_question(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
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
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
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

    def test_read_marked_bubble_labels_accepts_pencil_like_center_fill(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_pencil_like_mark(page, question_id="mc-1", bubble_label="B")
        )
        normalized = normalize_page_image(distorted, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": ["B"]},
            "A reasonably dark pencil-like center fill should still count as a mark, "
            "not require an unrealistically solid black bubble.",
        )

    def test_read_marked_bubble_labels_accepts_light_center_fill_after_blur(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_light_pencil_like_mark(page, question_id="mc-1", bubble_label="B")
        )
        blurred = cv2.GaussianBlur(distorted, (7, 7), 0)
        normalized = normalize_page_image(blurred, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": ["B"]},
            "A centered but lighter pencil fill should still count after mild scan blur "
            "instead of disappearing just because the graphite is not very dark.",
        )

    def test_read_marked_bubble_labels_ignores_edge_smudge_without_center_fill(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_edge_smudge(page, question_id="mc-1", bubble_label="B")
        )
        normalized = normalize_page_image(distorted, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": []},
            "A dark edge smudge without center fill should stay blank so the readback "
            "surface does not overreact to incidental scanner noise.",
        )

    def test_read_marked_bubble_labels_keeps_blank_page_blank_after_blur_and_noise(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(_render_marked_page(page, marked_labels={}))
        blurred = cv2.GaussianBlur(distorted, (11, 11), 0)
        noisy = blurred.copy().astype(np.int16)
        noisy += np.random.default_rng(1).normal(0, 22, size=noisy.shape).astype(np.int16)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        normalized = normalize_page_image(noisy, page)

        marked = read_marked_bubble_labels(normalized, page)

        self.assertEqual(
            marked,
            {"mc-1": []},
            "A blank page should remain blank under moderate blur and scanner-like noise "
            "so hardening for lighter marks does not turn into false positives.",
        )

    def test_read_marked_bubble_labels_preserves_multiple_marks_as_ambiguous_surface(self) -> None:
        read_marked_bubble_labels, _, normalize_page_image = _load_modules(self)
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

    def test_read_bubble_observations_keeps_correct_fill_with_stray_glance_corrigible(self) -> None:
        _, read_bubble_observations, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_correct_plus_stray_glance(
                page,
                question_id="mc-1",
                correct_bubble_label="B",
                stray_bubble_label="C",
            )
        )
        normalized = normalize_page_image(distorted, page)

        observations = read_bubble_observations(normalized, page)

        self.assertEqual(
            observations,
            {
                "mc-1": {
                    "marked_bubble_labels": ["B"],
                    "ambiguous_bubble_labels": [],
                    "illegible_bubble_labels": [],
                }
            },
            "A clear intended mark plus a glancing stray on another bubble should stay "
            "corrigible instead of escalating to fake ambiguity.",
        )

    def test_read_bubble_observations_surfaces_patchy_center_fill_as_ambiguous(self) -> None:
        _, read_bubble_observations, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_patchy_center_mark(page, question_id="mc-1", bubble_label="B")
        )
        normalized = normalize_page_image(distorted, page)

        observations = read_bubble_observations(normalized, page)

        self.assertEqual(
            observations,
            {
                "mc-1": {
                    "marked_bubble_labels": [],
                    "ambiguous_bubble_labels": ["B"],
                    "illegible_bubble_labels": [],
                }
            },
            "A patchy center fill near the current boundary should surface as explicit "
            "ambiguity instead of quietly disappearing into blank.",
        )

    def test_read_bubble_observations_surfaces_scratchout_as_illegible(self) -> None:
        _, read_bubble_observations, normalize_page_image = _load_modules(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        distorted = _perspective_distort(
            _render_illegible_scratchout(page, question_id="mc-1", bubble_label="B")
        )
        normalized = normalize_page_image(distorted, page)

        observations = read_bubble_observations(normalized, page)

        self.assertEqual(
            observations,
            {
                "mc-1": {
                    "marked_bubble_labels": [],
                    "ambiguous_bubble_labels": [],
                    "illegible_bubble_labels": ["B"],
                }
            },
            "A scratchy unreadable fill that drags across the bubble should surface as "
            "illegible review work, not a quiet blank or a fake confident mark.",
        )


if __name__ == "__main__":
    unittest.main()
