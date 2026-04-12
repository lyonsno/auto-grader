"""Contract tests for the MC review-override surface.

The scoring pipeline flags certain questions as review_required (multiple_marked,
ambiguous_mark, illegible_mark). This module tests the surface that lets a human
reviewer supply override decisions on those flagged questions and get back corrected
scoring results with provenance.
"""

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
        },
        "mc-2": {
            "question_id": "mc-2",
            "correct_choice_key": "B",
            "correct_bubble_label": "B",
            "choices": [
                {"bubble_label": "A", "choice_key": "D"},
                {"bubble_label": "B", "choice_key": "B"},
                {"bubble_label": "C", "choice_key": "A"},
                {"bubble_label": "D", "choice_key": "C"},
            ],
        },
    }


def _load_override_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_review_override import apply_mc_review_overrides
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_review_override.apply_mc_review_overrides(...)` so "
            "a human reviewer can override flagged MC question decisions and get back "
            "corrected scoring with provenance."
        )
    return apply_mc_review_overrides


class McReviewOverrideContractTests(unittest.TestCase):
    def test_override_resolves_multiple_marked_to_single_correct_answer(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "multiple_marked",
                "marked_bubble_labels": ["A", "C"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["A", "C"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["D", "A"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": True,
            },
        }

        overrides = {
            "mc-1": {"resolved_bubble_label": "C"},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "correct")
        self.assertTrue(result["mc-1"]["is_correct"])
        self.assertFalse(result["mc-1"]["review_required"])
        self.assertEqual(result["mc-1"]["resolved_bubble_labels"], ["C"])
        self.assertEqual(result["mc-1"]["marked_choice_keys"], ["A"])
        self.assertEqual(
            result["mc-1"]["override"]["original_status"],
            "multiple_marked",
            "The override result must preserve the original machine-scored status "
            "so the professor can see what the system originally thought.",
        )
        self.assertEqual(result["mc-1"]["override"]["resolved_bubble_label"], "C")

    def test_override_resolves_ambiguous_mark_to_single_answer(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "ambiguous_mark",
                "marked_bubble_labels": [],
                "ambiguous_bubble_labels": ["C"],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": [],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": [],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": True,
            },
        }

        overrides = {
            "mc-1": {"resolved_bubble_label": "C"},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "correct")
        self.assertTrue(result["mc-1"]["is_correct"])
        self.assertFalse(result["mc-1"]["review_required"])
        self.assertEqual(result["mc-1"]["override"]["original_status"], "ambiguous_mark")

    def test_override_resolves_illegible_mark_to_blank(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "illegible_mark",
                "marked_bubble_labels": [],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": ["C"],
                "resolved_bubble_labels": [],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": [],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": True,
            },
        }

        overrides = {
            "mc-1": {"resolved_bubble_label": None},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "blank")
        self.assertFalse(result["mc-1"]["is_correct"])
        self.assertFalse(result["mc-1"]["review_required"])
        self.assertEqual(result["mc-1"]["resolved_bubble_labels"], [])
        self.assertEqual(result["mc-1"]["marked_choice_keys"], [])
        self.assertEqual(result["mc-1"]["override"]["original_status"], "illegible_mark")
        self.assertIsNone(result["mc-1"]["override"]["resolved_bubble_label"])

    def test_override_passes_through_non_overridden_questions_unchanged(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "correct",
                "marked_bubble_labels": ["C"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["C"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["A"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": True,
                "review_required": False,
            },
            "mc-2": {
                "question_id": "mc-2",
                "status": "multiple_marked",
                "marked_bubble_labels": ["A", "B"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["A", "B"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["D", "B"],
                "correct_bubble_label": "B",
                "correct_choice_key": "B",
                "is_correct": False,
                "review_required": True,
            },
        }

        overrides = {
            "mc-2": {"resolved_bubble_label": "B"},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "correct")
        self.assertTrue(result["mc-1"]["is_correct"])
        self.assertNotIn("override", result["mc-1"])

        self.assertEqual(result["mc-2"]["status"], "correct")
        self.assertTrue(result["mc-2"]["is_correct"])
        self.assertIn("override", result["mc-2"])

    def test_override_rejects_unknown_question_id(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "correct",
                "marked_bubble_labels": ["C"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["C"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["A"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": True,
                "review_required": False,
            },
        }

        with self.assertRaisesRegex(ValueError, r"(?i)unknown.*question"):
            apply_mc_review_overrides(
                scored_questions, answer_key, {"mc-99": {"resolved_bubble_label": "A"}}
            )

    def test_override_rejects_unknown_bubble_label(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "multiple_marked",
                "marked_bubble_labels": ["A", "C"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["A", "C"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["D", "A"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": True,
            },
        }

        with self.assertRaisesRegex(ValueError, r"(?i)unknown.*bubble"):
            apply_mc_review_overrides(
                scored_questions, answer_key, {"mc-1": {"resolved_bubble_label": "Z"}}
            )

    def test_override_on_non_review_required_question_still_applies(self) -> None:
        """A professor should be able to override even machine-confident decisions."""
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "incorrect",
                "marked_bubble_labels": ["A"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["A"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["D"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": False,
            },
        }

        overrides = {
            "mc-1": {"resolved_bubble_label": "C"},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "correct")
        self.assertTrue(result["mc-1"]["is_correct"])
        self.assertEqual(result["mc-1"]["override"]["original_status"], "incorrect")

    def test_override_produces_incorrect_when_professor_resolves_to_wrong_answer(self) -> None:
        apply_mc_review_overrides = _load_override_module(self)
        answer_key = _answer_key()

        scored_questions = {
            "mc-1": {
                "question_id": "mc-1",
                "status": "multiple_marked",
                "marked_bubble_labels": ["A", "C"],
                "ambiguous_bubble_labels": [],
                "illegible_bubble_labels": [],
                "resolved_bubble_labels": ["A", "C"],
                "ignored_incidental_bubble_labels": [],
                "marked_choice_keys": ["D", "A"],
                "correct_bubble_label": "C",
                "correct_choice_key": "A",
                "is_correct": False,
                "review_required": True,
            },
        }

        overrides = {
            "mc-1": {"resolved_bubble_label": "A"},
        }

        result = apply_mc_review_overrides(scored_questions, answer_key, overrides)

        self.assertEqual(result["mc-1"]["status"], "incorrect")
        self.assertFalse(result["mc-1"]["is_correct"])
        self.assertFalse(result["mc-1"]["review_required"])
        self.assertEqual(result["mc-1"]["resolved_bubble_labels"], ["A"])
        self.assertEqual(result["mc-1"]["marked_choice_keys"], ["D"])
        self.assertEqual(result["mc-1"]["override"]["original_status"], "multiple_marked")


if __name__ == "__main__":
    unittest.main()
