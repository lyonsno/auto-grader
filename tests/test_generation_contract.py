"""Contract tests for the initial MC answer-sheet generation slice.

This suite defines the smallest generation artifact that the deterministic
OpenCV grading lane can honestly depend on: per-student MC answer-sheet data
with stable identity codes, shuffled answer-key mapping, and canonical bubble
regions in page coordinates.
"""

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


def _variableized_template(variable_order: tuple[str, ...]) -> dict:
    variable_specs = {
        "a": {"type": "int", "min": 1, "max": 5, "step": 1},
        "b": {"type": "int", "min": 10, "max": 90, "step": 10},
        "c": {"type": "int", "min": 100, "max": 500, "step": 100},
    }
    return {
        "slug": "quiz-variables",
        "title": "Variable Quiz",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-var-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Compute {{a}} + {{b}} + {{c}}.",
                        "variables": {
                            name: variable_specs[name] for name in variable_order
                        },
                        "choices": {
                            "A": "{{a}}",
                            "B": "{{b}}",
                            "C": "{{c}}",
                            "D": "{{a}} + {{b}} + {{c}}",
                        },
                        "correct": "D",
                        "shuffle": True,
                    }
                ],
            }
        ],
    }


def _dense_template(question_count: int = 20) -> dict:
    questions = []
    for index in range(1, question_count + 1):
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
    return {
        "slug": "quiz-dense",
        "title": "Dense Quiz",
        "sections": [{"id": "mc", "title": "Multiple Choice", "questions": questions}],
    }


class TestMcAnswerSheetGeneration(unittest.TestCase):
    def _build_one(self, student_id: str = "s-001", seed: int = 17):
        from auto_grader.generation import build_mc_answer_sheet

        return build_mc_answer_sheet(
            _template(),
            {"student_id": student_id, "student_name": "Ada Lovelace"},
            attempt_number=1,
            seed=seed,
        )

    def test_module_and_entrypoint_exist(self):
        from auto_grader.generation import build_mc_answer_sheet

        self.assertTrue(callable(build_mc_answer_sheet))

    def test_generation_is_deterministic_for_same_student_and_seed(self):
        first = self._build_one()
        second = self._build_one()

        self.assertEqual(first, second)

    def test_generation_changes_identity_for_different_students(self):
        first = self._build_one(student_id="s-001")
        second = self._build_one(student_id="s-002")

        self.assertNotEqual(first["opaque_instance_code"], second["opaque_instance_code"])
        self.assertNotEqual(first["pages"][0]["fallback_page_code"], second["pages"][0]["fallback_page_code"])

    def test_opaque_instance_code_does_not_expose_student_or_template_identifiers(self):
        artifact = self._build_one(student_id="Student 007")

        opaque_code = artifact["opaque_instance_code"].lower()
        self.assertNotIn("quiz-1", opaque_code)
        self.assertNotIn("student", opaque_code)
        self.assertNotIn("007", opaque_code)

    def test_answer_key_matches_rendered_bubble_layout(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        regions_by_question = {}
        for region in page["bubble_regions"]:
            regions_by_question.setdefault(region["question_id"], []).append(region)

        for question in artifact["mc_questions"]:
            answer_key = artifact["answer_key"][question["question_id"]]
            rendered_labels = [choice["bubble_label"] for choice in question["choices"]]
            self.assertEqual(rendered_labels, [choice["bubble_label"] for choice in answer_key["choices"]])
            self.assertIn(answer_key["correct_bubble_label"], rendered_labels)

            region_labels = sorted(region["bubble_label"] for region in regions_by_question[question["question_id"]])
            self.assertEqual(region_labels, sorted(rendered_labels))

    def test_bubble_regions_are_positive_rectangles_in_page_space(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        self.assertEqual(page["units"], "pt")
        self.assertEqual(page["origin"], "top_left")
        self.assertEqual(page["y_axis"], "down")
        self.assertEqual(page["layout_version"], "mc_answer_sheet_v1")
        self.assertGreater(page["width"], 0)
        self.assertGreater(page["height"], 0)
        self.assertGreater(len(page["bubble_regions"]), 0)

        for region in page["bubble_regions"]:
            self.assertEqual(region["shape"], "circle")
            self.assertGreater(region["x"], 0)
            self.assertGreater(region["y"], 0)
            self.assertGreater(region["width"], 0)
            self.assertGreater(region["height"], 0)
            self.assertLessEqual(region["x"] + region["width"], page["width"])
            self.assertLessEqual(region["y"] + region["height"], page["height"])

    def test_bubble_regions_are_large_enough_for_human_marks(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        for region in page["bubble_regions"]:
            self.assertGreaterEqual(
                region["width"],
                22,
                "Bubble regions should be large enough for a clean pencil mark, not tiny placeholders.",
            )
            self.assertGreaterEqual(
                region["height"],
                22,
                "Bubble regions should be large enough for a clean pencil mark, not tiny placeholders.",
            )

    def test_page_exposes_registration_markers_for_scan_normalization(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        self.assertEqual(
            [marker["marker_id"] for marker in page["registration_markers"]],
            ["top_left", "top_right", "bottom_left", "bottom_right"],
        )

        for marker in page["registration_markers"]:
            self.assertEqual(marker["kind"], "square")
            self.assertGreaterEqual(marker["x"], 0)
            self.assertGreaterEqual(marker["y"], 0)
            self.assertGreater(marker["width"], 0)
            self.assertGreater(marker["height"], 0)
            self.assertLessEqual(marker["x"] + marker["width"], page["width"])
            self.assertLessEqual(marker["y"] + marker["height"], page["height"])

    def test_page_exposes_duplicated_identity_qr_codes_for_machine_readback(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        qr_codes = page["identity_qr_codes"]
        self.assertEqual(
            len(qr_codes),
            2,
            "Each page should carry two duplicated QR placements so scan identification "
            "can survive one damaged or unreadable symbol.",
        )

        for qr_code in qr_codes:
            self.assertEqual(qr_code["kind"], "page_identity_qr")
            self.assertEqual(qr_code["encoding"], "qr")
            self.assertEqual(qr_code["payload"], page["fallback_page_code"])
            self.assertEqual(qr_code["error_correction"], "M")
            self.assertEqual(qr_code["border_modules"], 4)
            self.assertGreater(qr_code["width"], 0)
            self.assertEqual(qr_code["width"], qr_code["height"])
            self.assertLessEqual(qr_code["x"] + qr_code["width"], page["width"])
            self.assertLessEqual(qr_code["y"] + qr_code["height"], page["height"])

    def test_public_page_builder_supports_compact_layout_without_forking_page_substrate(self):
        try:
            from auto_grader.generation import (
                McAnswerSheetLayout,
                build_mc_answer_sheet_pages,
            )
        except ImportError:
            self.fail(
                "Export a public `McAnswerSheetLayout` plus "
                "`build_mc_answer_sheet_pages(...)` so alternate printable packet "
                "layouts can reuse the same QR, marker, and page-geometry substrate "
                "instead of rebuilding pages through private generation internals."
            )

        artifact = self._build_one()
        compact_layout = McAnswerSheetLayout(
            rows_per_page=6,
            layout_top=150,
            row_height=94,
            bubble_row_left=372,
        )

        pages = build_mc_answer_sheet_pages(
            artifact["opaque_instance_code"],
            artifact["mc_questions"],
            layout=compact_layout,
        )

        self.assertEqual(len(pages), 1)
        first_page = pages[0]
        self.assertEqual(first_page["fallback_page_code"], f"{artifact['opaque_instance_code']}-p1")
        self.assertEqual(
            [marker["marker_id"] for marker in first_page["registration_markers"]],
            ["top_left", "top_right", "bottom_left", "bottom_right"],
        )
        self.assertEqual(
            [qr_code["payload"] for qr_code in first_page["identity_qr_codes"]],
            [first_page["fallback_page_code"], first_page["fallback_page_code"]],
        )

        first_row_regions = [
            region for region in first_page["bubble_regions"] if region["question_id"] == "mc-1"
        ]
        self.assertEqual(min(region["x"] for region in first_row_regions), 372)
        self.assertEqual(min(region["y"] for region in first_row_regions), 150)

    def test_bubble_regions_leave_readable_horizontal_gaps_between_choices(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        first_question_regions = sorted(
            [
                region
                for region in page["bubble_regions"]
                if region["question_id"] == artifact["mc_questions"][0]["question_id"]
            ],
            key=lambda region: region["x"],
        )
        gaps = [
            later["x"] - (earlier["x"] + earlier["width"])
            for earlier, later in zip(first_question_regions, first_question_regions[1:], strict=False)
        ]
        self.assertTrue(gaps)
        for gap in gaps:
            self.assertGreaterEqual(
                gap,
                20,
                "Bubble boxes should not be packed edge-to-edge; the page contract "
                "needs enough horizontal breathing room for humans and later scan review.",
            )

    def test_question_rows_leave_vertical_room_for_prompt_and_stacked_choices(self):
        artifact = self._build_one()

        page = artifact["pages"][0]
        question_tops = []
        for question in artifact["mc_questions"]:
            question_regions = [
                region
                for region in page["bubble_regions"]
                if region["question_id"] == question["question_id"]
            ]
            question_tops.append(min(region["y"] for region in question_regions))

        row_gaps = [
            later - earlier
            for earlier, later in zip(question_tops, question_tops[1:], strict=False)
        ]
        self.assertTrue(row_gaps)
        for gap in row_gaps:
            self.assertGreaterEqual(
                gap,
                88,
                "Question rows need enough vertical room for a prompt line plus a "
                "stacked choice list under it without the next question crowding in.",
            )

    def test_blank_student_identifier_is_rejected(self):
        from auto_grader.generation import build_mc_answer_sheet

        with self.assertRaises(ValueError):
            build_mc_answer_sheet(
                _template(),
                {"student_id": "   ", "student_name": "Ada Lovelace"},
                attempt_number=1,
                seed=17,
            )

    def test_variableized_generation_is_stable_across_variable_declaration_order(self):
        from auto_grader.generation import build_mc_answer_sheet

        student = {"student_id": "s-001", "student_name": "Ada Lovelace"}
        first = build_mc_answer_sheet(
            _variableized_template(("a", "b", "c")),
            student,
            attempt_number=1,
            seed=0,
        )
        second = build_mc_answer_sheet(
            _variableized_template(("c", "b", "a")),
            student,
            attempt_number=1,
            seed=0,
        )

        self.assertEqual(first, second)

    def test_dense_mc_sheets_paginate_instead_of_cramming_twenty_questions_onto_one_page(self):
        from auto_grader.generation import build_mc_answer_sheet

        artifact = build_mc_answer_sheet(
            _dense_template(),
            {"student_id": "s-001", "student_name": "Ada Lovelace"},
            attempt_number=1,
            seed=17,
        )

        self.assertEqual(
            len(artifact["pages"]),
            5,
            "A dense 20-question MC sheet should paginate instead of forcing two busy "
            "columns onto one page.",
        )
        self.assertEqual(
            [page["fallback_page_code"] for page in artifact["pages"]],
            [
                f"{artifact['opaque_instance_code']}-p1",
                f"{artifact['opaque_instance_code']}-p2",
                f"{artifact['opaque_instance_code']}-p3",
                f"{artifact['opaque_instance_code']}-p4",
                f"{artifact['opaque_instance_code']}-p5",
            ],
        )
        page_question_ids = [
            {region["question_id"] for region in page["bubble_regions"]}
            for page in artifact["pages"]
        ]
        self.assertEqual(page_question_ids[0], {f"mc-{index}" for index in range(1, 5)})
        self.assertEqual(page_question_ids[1], {f"mc-{index}" for index in range(5, 9)})
        self.assertEqual(page_question_ids[2], {f"mc-{index}" for index in range(9, 13)})
        self.assertEqual(page_question_ids[3], {f"mc-{index}" for index in range(13, 17)})
        self.assertEqual(page_question_ids[4], {f"mc-{index}" for index in range(17, 21)})
