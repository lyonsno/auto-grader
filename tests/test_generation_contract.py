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
        self.assertGreater(page["width"], 0)
        self.assertGreater(page["height"], 0)
        self.assertGreater(len(page["bubble_regions"]), 0)

        for region in page["bubble_regions"]:
            self.assertGreater(region["x"], 0)
            self.assertGreater(region["y"], 0)
            self.assertGreater(region["width"], 0)
            self.assertGreater(region["height"], 0)
            self.assertLessEqual(region["x"] + region["width"], page["width"])
            self.assertLessEqual(region["y"] + region["height"], page["height"])

    def test_blank_student_identifier_is_rejected(self):
        from auto_grader.generation import build_mc_answer_sheet

        with self.assertRaises(ValueError):
            build_mc_answer_sheet(
                _template(),
                {"student_id": "   ", "student_name": "Ada Lovelace"},
                attempt_number=1,
                seed=17,
            )
