from __future__ import annotations

import unittest


def _template() -> dict:
    return {
        "slug": "quiz-1",
        "title": "Quiz 1",
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
                    },
                    {
                        "id": "mc-2",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Which gas is monatomic at STP?",
                        "choices": {
                            "A": "He",
                            "B": "N2",
                            "C": "Cl2",
                            "D": "CO2",
                        },
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


def _load_pdf_renderer(test_case: unittest.TestCase):
    try:
        from auto_grader.pdf_rendering import render_mc_answer_sheet_pdf
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.pdf_rendering.render_mc_answer_sheet_pdf(...)` so "
            "the generation artifact can be rendered into a concrete answer-sheet PDF."
        )
    return render_mc_answer_sheet_pdf


def _pdf_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


class PdfRenderingContractTests(unittest.TestCase):
    def test_renderer_emits_pdf_bytes_with_contract_visible_on_page(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(
            pdf_bytes.startswith(b"%PDF-1."),
            "Rendered answer sheets should be returned as real PDF bytes.",
        )
        self.assertIn(
            b"/MediaBox [0 0 612 792]",
            pdf_bytes,
            "The rendered page size should match the Letter-size layout contract.",
        )
        self.assertIn(
            artifact["opaque_instance_code"].encode("utf-8"),
            pdf_bytes,
            "The page should visibly carry the opaque instance identifier for manual recovery.",
        )
        self.assertIn(
            artifact["pages"][0]["fallback_page_code"].encode("utf-8"),
            pdf_bytes,
            "The page should visibly carry the fallback page code for manual recovery.",
        )
        self.assertIn(
            b"Which species is elemental oxygen?",
            pdf_bytes,
            "The rendered page should include question prompt text from the artifact.",
        )
        self.assertIn(
            b"O2",
            pdf_bytes,
            "The rendered page should include rendered choice text from the artifact.",
        )

    def test_renderer_draws_bubble_circles_at_artifact_coordinates(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        for region in page["bubble_regions"]:
            pdf_y = page["height"] - region["y"] - region["height"]
            circle_command = _pdf_circle_command(
                region["x"],
                pdf_y,
                region["width"],
                region["height"],
            ).encode("utf-8")
            self.assertIn(
                circle_command,
                pdf_bytes,
                "Every rendered bubble should be drawn as a circle at the page-space "
                "coordinates from the generation artifact exactly.",
            )

    def test_renderer_positions_question_prompt_above_bubble_row(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        first_question = artifact["mc_questions"][0]
        first_question_regions = [
            region
            for region in page["bubble_regions"]
            if region["question_id"] == first_question["question_id"]
        ]

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        prompt_x = max(36, min(region["x"] for region in first_question_regions) - 96)
        prompt_y = _pdf_number(page["height"] - (min(region["y"] for region in first_question_regions) - 18) - 10)
        prompt_command = (
            f"BT\n/F1 10 Tf\n{_pdf_number(prompt_x)} {prompt_y} Td\n"
            f"({_escape_pdf_text('1. ' + first_question['prompt'])}) Tj\nET"
        ).encode("utf-8")

        self.assertIn(
            prompt_command,
            pdf_bytes,
            "Question prompt text should render on its own line above the bubble row "
            "instead of sharing the same baseline as the choices.",
        )

    def test_renderer_offsets_header_clear_of_top_left_registration_marker(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        top_left_marker = next(
            marker for marker in page["registration_markers"] if marker["marker_id"] == "top_left"
        )

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        expected_x = top_left_marker["x"] + top_left_marker["width"] + 18
        title_command = (
            f"BT\n/F1 11 Tf\n{_pdf_number(expected_x)} 750 Td\n"
            f"({_escape_pdf_text('MC Answer Sheet')}) Tj\nET"
        ).encode("utf-8")
        self.assertIn(
            title_command,
            pdf_bytes,
            "Header text should clear the top-left registration marker instead of colliding with it.",
        )

    def test_renderer_positions_choice_legend_below_bubble_row(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        first_question = artifact["mc_questions"][0]
        first_question_regions = [
            region
            for region in page["bubble_regions"]
            if region["question_id"] == first_question["question_id"]
        ]

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        legend_text = "   ".join(
            f"{choice['bubble_label']}. {choice['text']}" for choice in first_question["choices"]
        )
        legend_command = (
            f"BT\n/F1 8 Tf\n156 "
            f"{_pdf_number(page['height'] - (min(region['y'] for region in first_question_regions) + 24) - 10)} Td\n"
            f"({_escape_pdf_text(legend_text)}) Tj\nET"
        ).encode("utf-8")
        self.assertIn(
            legend_command,
            pdf_bytes,
            "Choice labels should render on their own legend line below the bubble row "
            "instead of being crammed into the marks themselves.",
        )

    def test_renderer_draws_registration_markers_at_artifact_coordinates(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        for marker in page["registration_markers"]:
            pdf_y = page["height"] - marker["y"] - marker["height"]
            marker_command = (
                f"{_pdf_number(marker['x'])} "
                f"{_pdf_number(pdf_y)} "
                f"{_pdf_number(marker['width'])} "
                f"{_pdf_number(marker['height'])} re f"
            ).encode("utf-8")
            self.assertIn(
                marker_command,
                pdf_bytes,
                "Registration markers should be drawn at the exact page-space "
                "coordinates recorded in the artifact.",
            )


def _escape_pdf_text(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_circle_command(x: int | float, y: int | float, width: int | float, height: int | float) -> str:
    radius_x = width / 2
    radius_y = height / 2
    center_x = x + radius_x
    center_y = y + radius_y
    control = 0.552284749831 * radius_x
    return " ".join(
        [
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y)} m",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y + control)}",
            f"{_pdf_number(center_x + control)} {_pdf_number(center_y + radius_y)}",
            f"{_pdf_number(center_x)} {_pdf_number(center_y + radius_y)} c",
            f"{_pdf_number(center_x - control)} {_pdf_number(center_y + radius_y)}",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y + control)}",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y)} c",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y - control)}",
            f"{_pdf_number(center_x - control)} {_pdf_number(center_y - radius_y)}",
            f"{_pdf_number(center_x)} {_pdf_number(center_y - radius_y)} c",
            f"{_pdf_number(center_x + control)} {_pdf_number(center_y - radius_y)}",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y - control)}",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y)} c S",
        ]
    )
