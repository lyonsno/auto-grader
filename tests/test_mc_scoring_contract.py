from __future__ import annotations

import unittest


def _answer_key() -> dict[str, dict]:
    return {
        "mc-1": {
            "question_id": "mc-1",
            "correct_choice_key": "A",
            "correct_bubble_label": "C",
            "choices": [
                {"bubble_label": "A", "choice_key": "D"},
                {"bubble_label": "B", "choice_key": "B"},
                {"bubble_label": "C", "choice_key": "A"},
                {"bubble_label": "D", "choice_key": "C"},
            ],
        }
    }


def _load_scoring_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_scoring import score_marked_mc_bubbles
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_scoring.score_marked_mc_bubbles(...)` so the "
            "OpenCV lane can turn bubble readback plus the per-student answer key "
            "into explicit grading decisions."
        )
    return score_marked_mc_bubbles


class McScoringContractTests(unittest.TestCase):
    def test_scoring_marks_correct_single_selection(self) -> None:
        score_marked_mc_bubbles = _load_scoring_module(self)

        results = score_marked_mc_bubbles({"mc-1": ["C"]}, _answer_key())

        self.assertEqual(
            results,
            {
                "mc-1": {
                    "question_id": "mc-1",
                    "status": "correct",
                    "marked_bubble_labels": ["C"],
                    "marked_choice_keys": ["A"],
                    "correct_bubble_label": "C",
                    "correct_choice_key": "A",
                    "is_correct": True,
                    "review_required": False,
                }
            },
            "A single marked bubble that maps to the keyed correct answer should score "
            "as correct without requiring review.",
        )

    def test_scoring_marks_incorrect_single_selection(self) -> None:
        score_marked_mc_bubbles = _load_scoring_module(self)

        results = score_marked_mc_bubbles({"mc-1": ["B"]}, _answer_key())

        self.assertEqual(
            results["mc-1"]["status"],
            "incorrect",
            "A single marked bubble that maps to the wrong logical choice should score "
            "as incorrect.",
        )
        self.assertEqual(results["mc-1"]["marked_choice_keys"], ["B"])
        self.assertFalse(results["mc-1"]["is_correct"])
        self.assertFalse(results["mc-1"]["review_required"])

    def test_scoring_keeps_blank_question_explicit(self) -> None:
        score_marked_mc_bubbles = _load_scoring_module(self)

        results = score_marked_mc_bubbles({"mc-1": []}, _answer_key())

        self.assertEqual(results["mc-1"]["status"], "blank")
        self.assertEqual(results["mc-1"]["marked_choice_keys"], [])
        self.assertFalse(results["mc-1"]["is_correct"])
        self.assertFalse(results["mc-1"]["review_required"])

    def test_scoring_preserves_multiple_marks_as_review_required(self) -> None:
        score_marked_mc_bubbles = _load_scoring_module(self)

        results = score_marked_mc_bubbles({"mc-1": ["A", "C"]}, _answer_key())

        self.assertEqual(results["mc-1"]["status"], "multiple_marked")
        self.assertEqual(results["mc-1"]["marked_choice_keys"], ["D", "A"])
        self.assertFalse(results["mc-1"]["is_correct"])
        self.assertTrue(results["mc-1"]["review_required"])

    def test_scoring_rejects_unknown_bubble_labels_in_readback_surface(self) -> None:
        score_marked_mc_bubbles = _load_scoring_module(self)

        with self.assertRaisesRegex(ValueError, "Unknown bubble label"):
            score_marked_mc_bubbles({"mc-1": ["Z"]}, _answer_key())


if __name__ == "__main__":
    unittest.main()
