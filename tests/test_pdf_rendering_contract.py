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


def _build_dense_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    questions = []
    for index in range(1, 21):
        questions.append(
            {
                "id": f"mc-{index}",
                "points": 2,
                "answer_type": "multiple_choice",
                "prompt": (
                    f"Question {index}: Which statement best describes why increasing "
                    "surface area speeds up a heterogeneous reaction in a powder sample?"
                ),
                "choices": {
                    "A": "More exposed particles create more collision opportunities",
                    "B": "The activation energy always drops to zero",
                    "C": "The equilibrium constant becomes larger",
                    "D": "The sample gains additional electrons",
                },
                "correct": "A",
                "shuffle": True,
            }
        )

    template = {
        "slug": "quiz-dense",
        "title": "Dense Quiz",
        "sections": [{"id": "mc", "title": "Multiple Choice", "questions": questions}],
    }
    return build_mc_answer_sheet(
        template,
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=17,
    )


def _build_wrapped_prompt_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    template = {
        "slug": "quiz-wrap",
        "title": "Wrap Quiz",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-wrap-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": (
                            "Which statement best describes why increasing surface area "
                            "speeds up a heterogeneous reaction in a powder sample "
                            "during a busy laboratory period?"
                        ),
                        "choices": {
                            "A": "More exposed particles create more collision opportunities",
                            "B": "The activation energy always drops to zero",
                            "C": "The equilibrium constant becomes larger",
                            "D": "The sample gains additional electrons",
                        },
                        "correct": "A",
                        "shuffle": False,
                    }
                ],
            }
        ],
    }
    return build_mc_answer_sheet(
        template,
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=17,
    )


def _build_wrapped_choice_artifact() -> dict:
    from auto_grader.generation import build_mc_answer_sheet

    template = {
        "slug": "quiz-choice-wrap",
        "title": "Choice Wrap Quiz",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-choice-wrap-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": (
                            "Which statement best describes why increasing surface area "
                            "speeds up a heterogeneous reaction in a powder sample?"
                        ),
                        "choices": {
                            "A": "More exposed particles create more collision opportunities",
                            "B": "The equilibrium constant becomes larger",
                            "C": "The sample gains additional electrons",
                            "D": "The activation energy always drops to zero",
                        },
                        "correct": "A",
                        "shuffle": False,
                    }
                ],
            }
        ],
    }
    return build_mc_answer_sheet(
        template,
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

        prompt_x = 72
        prompt_y = _pdf_number(page["height"] - (min(region["y"] for region in first_question_regions) - 28) - 10)
        prompt_command = (
            f"BT\n/F1 13 Tf\n{_pdf_number(prompt_x)} {prompt_y} Td\n"
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
            f"BT\n/F1 18 Tf\n{_pdf_number(expected_x)} 746 Td\n"
            f"({_escape_pdf_text('MC Answer Sheet')}) Tj\nET"
        ).encode("utf-8")
        self.assertIn(
            title_command,
            pdf_bytes,
            "Header text should clear the top-left registration marker instead of colliding with it.",
        )

    def test_renderer_uses_larger_question_and_choice_typography(self) -> None:
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

        prompt_x = 72
        prompt_y = _pdf_number(page["height"] - (min(region["y"] for region in first_question_regions) - 28) - 10)
        prompt_command = (
            f"BT\n/F1 13 Tf\n{_pdf_number(prompt_x)} {prompt_y} Td\n"
            f"({_escape_pdf_text('1. ' + first_question['prompt'])}) Tj\nET"
        ).encode("utf-8")
        choice_command = (
            f"BT\n/F1 10 Tf\n84 "
            f"{_pdf_number(page['height'] - (min(region['y'] for region in first_question_regions) + 4) - 10)} Td\n"
            f"({_escape_pdf_text(first_question['choices'][0]['bubble_label'] + '. ' + first_question['choices'][0]['text'])}) Tj\nET"
        ).encode("utf-8")

        self.assertIn(
            prompt_command,
            pdf_bytes,
            "Question prompts should use a more readable font size once the page structure is settled.",
        )
        self.assertIn(
            choice_command,
            pdf_bytes,
            "Choice text should scale up with the question prompt instead of staying tiny.",
        )

    def test_renderer_draws_bubble_labels_below_circles(self) -> None:
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

        for region in first_question_regions:
            center_x = region["x"] + (region["width"] / 2)
            label_command = (
                f"BT\n/F1 10 Tf\n{_pdf_number(center_x - 2)} "
                f"{_pdf_number(page['height'] - (region['y'] + region['height'] + 6) - 10)} Td\n"
                f"({_escape_pdf_text(region['bubble_label'])}) Tj\nET"
            ).encode("utf-8")
            self.assertIn(
                label_command,
                pdf_bytes,
                "Each bubble should be labeled directly underneath so the response row "
                "is readable without forcing the option text onto the same line.",
            )

    def test_renderer_stacks_choice_text_below_question_and_left_of_bubbles(self) -> None:
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

        prompt_x = 72
        legend_x = 84
        line_spacing = 14
        first_row_top = min(region["y"] for region in first_question_regions)
        first_bubble_x = min(region["x"] for region in first_question_regions)
        prompt_y = page["height"] - (first_row_top - 28) - 10
        for index, choice in enumerate(first_question["choices"]):
            legend_y = page["height"] - (first_row_top + 4 + index * line_spacing) - 10
            legend_command = (
                f"BT\n/F1 10 Tf\n{legend_x} "
                f"{_pdf_number(legend_y)} Td\n"
                f"({_escape_pdf_text(choice['bubble_label'] + '. ' + choice['text'])}) Tj\nET"
            ).encode("utf-8")
            self.assertIn(
                legend_command,
                pdf_bytes,
                "Choice text should stack under the question prompt and sit to the "
                "left of the bubble row instead of being shoved off to the side.",
            )
            self.assertLess(legend_x, first_bubble_x)
            self.assertLess(legend_y, prompt_y)
            self.assertGreaterEqual(
                first_bubble_x - legend_x,
                180,
                "The choice list needs a real horizontal gutter before the bubble row "
                "so longer distractors do not crash into the circles.",
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

    def test_renderer_emits_multiple_pages_for_dense_mc_sheets(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_dense_artifact()

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        self.assertIn(
            b"<< /Type /Pages /Count 5 /Kids [4 0 R 6 0 R 8 0 R 10 0 R 12 0 R] >>",
            pdf_bytes,
            "A dense 20-question sheet should render as a multi-page PDF rather than a single overflowing page.",
        )
        self.assertIn(artifact["pages"][1]["fallback_page_code"].encode("utf-8"), pdf_bytes)
        self.assertIn(b"(9. ", pdf_bytes)

    def test_renderer_wraps_long_prompts_within_the_question_block(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_wrapped_prompt_artifact()

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        self.assertIn(
            b"(1. Which statement best describes why increasing)",
            pdf_bytes,
            "Long prompts should wrap onto multiple renderer lines instead of staying as one unbounded text run.",
        )
        self.assertIn(
            b"(surface area speeds up a heterogeneous reaction in a)",
            pdf_bytes,
        )
        self.assertIn(
            b"(powder sample during a busy laboratory period?)",
            pdf_bytes,
        )
        self.assertNotIn(
            b"(1. Which statement best describes why increasing surface area speeds up a heterogeneous reaction in a powder sample during a busy laboratory period?)",
            pdf_bytes,
            "The renderer should not emit the full long prompt as one unwrapped line once prompt wrapping is supported.",
        )

    def test_renderer_avoids_duplicate_question_label_when_prompt_is_already_numbered(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_dense_artifact()

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        self.assertIn(
            b"8. Which statement best describes why increasing",
            pdf_bytes,
            "Dense-sheet prompts should keep the renderer's own numeric prefix without repeating "
            "a second 'Question N:' label from the prompt text itself.",
        )
        self.assertNotIn(
            b"Question 8:",
            pdf_bytes,
            "Rendered prompt text should not repeat 'Question N:' once the renderer already "
            "numbers each item on the page.",
        )

    def test_renderer_wraps_long_choice_lines_before_the_bubble_lane(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_wrapped_choice_artifact()

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        self.assertIn(
            b"(A. More exposed particles create more)",
            pdf_bytes,
            "Long choice text should wrap inside the left-hand text block instead of overrunning the bubble lane.",
        )
        self.assertIn(
            b"(   collision opportunities)",
            pdf_bytes,
        )
        self.assertNotIn(
            b"(A. More exposed particles create more collision opportunities)",
            pdf_bytes,
            "Long choice text should not remain one unbounded line once choice wrapping is supported.",
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
