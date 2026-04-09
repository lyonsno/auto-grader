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

    def test_renderer_draws_bubble_rectangles_at_artifact_coordinates(self) -> None:
        render_mc_answer_sheet_pdf = _load_pdf_renderer(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        pdf_bytes = render_mc_answer_sheet_pdf(artifact)

        for region in page["bubble_regions"]:
            pdf_y = page["height"] - region["y"] - region["height"]
            rectangle_command = (
                f"{_pdf_number(region['x'])} "
                f"{_pdf_number(pdf_y)} "
                f"{_pdf_number(region['width'])} "
                f"{_pdf_number(region['height'])} re S"
            ).encode("utf-8")
            self.assertIn(
                rectangle_command,
                pdf_bytes,
                "Every rendered bubble rectangle should use the page-space coordinates "
                "from the generation artifact exactly.",
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
