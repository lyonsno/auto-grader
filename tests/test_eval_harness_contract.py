"""Contract tests for the eval harness.

These tests define the contract for auto_grader.eval_harness — the module
that loads ground truth data, accepts model predictions, and produces
accuracy/calibration reports.  Tests are written before the implementation
(fail-first discipline).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
_GROUND_TRUTH_PATH = _EVAL_DIR / "ground_truth.yaml"


def _make_perfect_predictions(ground_truth: list) -> list:
    """Return predictions that exactly match the scoring truth."""
    from auto_grader.eval_harness import Prediction

    preds = []
    for item in ground_truth:
        preds.append(
            Prediction(
                exam_id=item.exam_id,
                question_id=item.question_id,
                model_score=item.truth_score,
                model_confidence=1.0,
                model_reasoning="matches scoring truth",
                model_read=item.student_answer,
            )
        )
    return preds


def _make_wrong_predictions(ground_truth: list) -> list:
    """Return predictions that always give max_points (wrong for x/partial)."""
    from auto_grader.eval_harness import Prediction

    preds = []
    for item in ground_truth:
        preds.append(
            Prediction(
                exam_id=item.exam_id,
                question_id=item.question_id,
                model_score=item.max_points,  # always max — wrong for x/partial
                model_confidence=0.5,
                model_reasoning="always max",
                model_read=item.student_answer,
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Contract: Module loads and exports expected names
# ---------------------------------------------------------------------------


class TestEvalHarnessImports(unittest.TestCase):
    """The module must export the documented public API."""

    def test_load_ground_truth_is_callable(self):
        from auto_grader.eval_harness import load_ground_truth

        self.assertTrue(callable(load_ground_truth))

    def test_score_predictions_is_callable(self):
        from auto_grader.eval_harness import score_predictions

        self.assertTrue(callable(score_predictions))

    def test_eval_item_is_a_class(self):
        from auto_grader.eval_harness import EvalItem

        self.assertTrue(isinstance(EvalItem, type))

    def test_prediction_is_a_class(self):
        from auto_grader.eval_harness import Prediction

        self.assertTrue(isinstance(Prediction, type))

    def test_eval_report_is_a_class(self):
        from auto_grader.eval_harness import EvalReport

        self.assertTrue(isinstance(EvalReport, type))


# ---------------------------------------------------------------------------
# Contract: load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth(unittest.TestCase):
    """load_ground_truth must parse the YAML into a flat list of EvalItem."""

    def test_returns_list(self):
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        self.assertIsInstance(result, list)

    def test_returns_152_items(self):
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        self.assertEqual(len(result), 152)

    def test_items_are_eval_item_instances(self):
        from auto_grader.eval_harness import EvalItem, load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        for item in result:
            self.assertIsInstance(item, EvalItem)

    def test_eval_item_has_required_fields(self):
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        item = result[0]
        for field in (
            "exam_id",
            "question_id",
            "answer_type",
            "page",
            "professor_score",
            "max_points",
            "professor_mark",
            "student_answer",
            "notes",
        ):
            self.assertTrue(
                hasattr(item, field), f"EvalItem missing field: {field}"
            )

    def test_first_item_values(self):
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        item = result[0]
        self.assertEqual(item.exam_id, "15-blue")
        self.assertEqual(item.question_id, "fr-1")
        self.assertEqual(item.answer_type, "numeric")
        self.assertEqual(item.professor_score, 2)
        self.assertEqual(item.max_points, 2)
        self.assertEqual(item.professor_mark, "check")

    def test_covers_all_four_exams(self):
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        exam_ids = {item.exam_id for item in result}
        self.assertEqual(len(exam_ids), 4)

    def test_professor_mark_values(self):
        """All marks must be one of the four documented values."""
        from auto_grader.eval_harness import load_ground_truth

        result = load_ground_truth(_GROUND_TRUTH_PATH)
        valid_marks = {"check", "x", "partial", "unclear"}
        for item in result:
            self.assertIn(
                item.professor_mark,
                valid_marks,
                f"Invalid mark {item.professor_mark!r} for {item.exam_id}/{item.question_id}",
            )

    def test_nonexistent_file_raises(self):
        from auto_grader.eval_harness import load_ground_truth

        with self.assertRaises(FileNotFoundError):
            load_ground_truth(Path("/nonexistent/path.yaml"))


# ---------------------------------------------------------------------------
# Contract: score_predictions — perfect predictions
# ---------------------------------------------------------------------------


class TestScorePerfectPredictions(unittest.TestCase):
    """When model scores exactly match scoring truth, accuracy must be 1.0."""

    @classmethod
    def setUpClass(cls):
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        cls.ground_truth = load_ground_truth(_GROUND_TRUTH_PATH)
        cls.predictions = _make_perfect_predictions(cls.ground_truth)
        cls.report = score_predictions(cls.ground_truth, cls.predictions)

    def test_report_is_eval_report(self):
        from auto_grader.eval_harness import EvalReport

        self.assertIsInstance(self.report, EvalReport)

    def test_overall_exact_accuracy_is_1(self):
        self.assertEqual(self.report.overall_exact_accuracy, 1.0)

    def test_overall_tolerance_accuracy_is_1(self):
        self.assertEqual(self.report.overall_tolerance_accuracy, 1.0)

    def test_false_positives_is_0(self):
        self.assertEqual(self.report.false_positives, 0)

    def test_false_negatives_is_0(self):
        self.assertEqual(self.report.false_negatives, 0)

    def test_per_answer_type_exact_accuracy_all_1(self):
        for atype, acc in self.report.per_answer_type_exact.items():
            self.assertEqual(
                acc, 1.0, f"answer_type {atype} exact accuracy != 1.0"
            )

    def test_per_answer_type_tolerance_accuracy_all_1(self):
        for atype, acc in self.report.per_answer_type_tolerance.items():
            self.assertEqual(
                acc, 1.0, f"answer_type {atype} tolerance accuracy != 1.0"
            )

    def test_total_items_counted(self):
        """Report should track how many items were scored."""
        self.assertGreater(self.report.total_scored, 0)

    def test_unclear_excluded_from_accuracy(self):
        """Items with professor_mark=unclear should be excluded from scoring."""
        from auto_grader.eval_harness import load_ground_truth

        gt = load_ground_truth(_GROUND_TRUTH_PATH)
        unclear_count = sum(1 for i in gt if i.professor_mark == "unclear")
        # total_scored should be 152 minus unclear items
        self.assertEqual(self.report.total_scored, 152 - unclear_count)

    def test_unclear_count_tracked(self):
        """Report must separately track how many unclear items were excluded."""
        self.assertGreater(self.report.unclear_excluded, 0)

    def test_corrected_score_item_uses_truth_score(self):
        """Corrected items must score against corrected truth, not history."""
        corrected_items = [
            item for item in self.ground_truth if item.corrected_score is not None
        ]
        self.assertGreater(
            len(corrected_items), 0, "Sanity: expected at least one corrected item"
        )
        item = corrected_items[0]
        self.assertNotEqual(item.professor_score, item.truth_score)

        matched = next(
            pred
            for pred in self.predictions
            if pred.exam_id == item.exam_id and pred.question_id == item.question_id
        )
        self.assertEqual(matched.model_score, item.truth_score)


# ---------------------------------------------------------------------------
# Contract: score_predictions — deliberately wrong predictions
# ---------------------------------------------------------------------------


class TestScoreWrongPredictions(unittest.TestCase):
    """When model always awards max_points, accuracy must be < 1.0."""

    @classmethod
    def setUpClass(cls):
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        cls.ground_truth = load_ground_truth(_GROUND_TRUTH_PATH)
        cls.predictions = _make_wrong_predictions(cls.ground_truth)
        cls.report = score_predictions(cls.ground_truth, cls.predictions)

    def test_overall_exact_accuracy_below_1(self):
        self.assertLess(self.report.overall_exact_accuracy, 1.0)

    def test_has_false_positives(self):
        """Awarding max to x-marked items should produce false positives."""
        self.assertGreater(self.report.false_positives, 0)

    def test_no_false_negatives(self):
        """Awarding max never docks points, so no false negatives."""
        self.assertEqual(self.report.false_negatives, 0)

    def test_numeric_type_accuracy_below_1(self):
        """Not all numeric items got full marks from professor."""
        self.assertLess(self.report.per_answer_type_exact["numeric"], 1.0)


# ---------------------------------------------------------------------------
# Contract: score_predictions — partial credit items
# ---------------------------------------------------------------------------


class TestPartialCreditScoring(unittest.TestCase):
    """Partial credit items (professor_mark=partial) must be included."""

    @classmethod
    def setUpClass(cls):
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        cls.ground_truth = load_ground_truth(_GROUND_TRUTH_PATH)
        # Make predictions that give 0 for everything — partial items should
        # count as wrong (professor awarded partial credit, model gave 0).
        from auto_grader.eval_harness import Prediction

        cls.predictions = [
            Prediction(
                exam_id=item.exam_id,
                question_id=item.question_id,
                model_score=0,
                model_confidence=0.1,
                model_reasoning="always zero",
                model_read="",
            )
            for item in cls.ground_truth
        ]
        cls.report = score_predictions(cls.ground_truth, cls.predictions)

    def test_partial_items_are_scored(self):
        """Partial credit items should be included in total_scored."""
        from auto_grader.eval_harness import load_ground_truth

        gt = load_ground_truth(_GROUND_TRUTH_PATH)
        partial_count = sum(1 for i in gt if i.professor_mark == "partial")
        self.assertGreater(partial_count, 0, "Sanity: should have partial items")
        # All partial items should be scored (not excluded)
        unclear_count = sum(1 for i in gt if i.professor_mark == "unclear")
        self.assertEqual(self.report.total_scored, 152 - unclear_count)

    def test_has_false_negatives(self):
        """Giving 0 when professor gave partial credit = false negative."""
        self.assertGreater(self.report.false_negatives, 0)


# ---------------------------------------------------------------------------
# Contract: score_predictions — calibration data
# ---------------------------------------------------------------------------


class TestCalibrationData(unittest.TestCase):
    """Report should include confidence calibration data when available."""

    @classmethod
    def setUpClass(cls):
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        cls.ground_truth = load_ground_truth(_GROUND_TRUTH_PATH)
        cls.predictions = _make_perfect_predictions(cls.ground_truth)
        cls.report = score_predictions(cls.ground_truth, cls.predictions)

    def test_calibration_data_exists(self):
        """Report must have a calibration_bins attribute."""
        self.assertTrue(hasattr(self.report, "calibration_bins"))

    def test_calibration_bins_are_list(self):
        self.assertIsInstance(self.report.calibration_bins, list)

    def test_calibration_bins_have_expected_shape(self):
        """Each bin should have bin_start, bin_end, count, avg_confidence,
        and accuracy fields."""
        if not self.report.calibration_bins:
            self.skipTest("No calibration bins produced")
        b = self.report.calibration_bins[0]
        for field in ("bin_start", "bin_end", "count", "avg_confidence", "accuracy"):
            self.assertTrue(
                hasattr(b, field), f"CalibrationBin missing field: {field}"
            )


# ---------------------------------------------------------------------------
# Contract: score_predictions — prediction matching
# ---------------------------------------------------------------------------


class TestPredictionMatching(unittest.TestCase):
    """Predictions are matched to ground truth by (exam_id, question_id)."""

    def test_missing_prediction_raises(self):
        """If a prediction is missing for a scored item, raise ValueError."""
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        gt = load_ground_truth(_GROUND_TRUTH_PATH)
        # Empty predictions list — should raise
        with self.assertRaises(ValueError):
            score_predictions(gt, [])

    def test_extra_prediction_is_ignored(self):
        """Extra predictions (no matching ground truth) should be ignored."""
        from auto_grader.eval_harness import (
            Prediction,
            load_ground_truth,
            score_predictions,
        )

        gt = load_ground_truth(_GROUND_TRUTH_PATH)
        preds = _make_perfect_predictions(gt)
        # Add an extra prediction with no matching ground truth
        preds.append(
            Prediction(
                exam_id="nonexistent-exam",
                question_id="fr-999",
                model_score=0,
                model_confidence=0.5,
                model_reasoning="extra",
                model_read="",
            )
        )
        report = score_predictions(gt, preds)
        # Should still work and produce valid results
        self.assertEqual(report.overall_exact_accuracy, 1.0)


# ---------------------------------------------------------------------------
# Contract: EvalReport summary statistics
# ---------------------------------------------------------------------------


class TestEvalReportSummary(unittest.TestCase):
    """EvalReport must provide summary statistics."""

    @classmethod
    def setUpClass(cls):
        from auto_grader.eval_harness import load_ground_truth, score_predictions

        cls.ground_truth = load_ground_truth(_GROUND_TRUTH_PATH)
        cls.predictions = _make_perfect_predictions(cls.ground_truth)
        cls.report = score_predictions(cls.ground_truth, cls.predictions)

    def test_has_per_answer_type_exact(self):
        self.assertIsInstance(self.report.per_answer_type_exact, dict)
        self.assertGreater(len(self.report.per_answer_type_exact), 0)

    def test_has_per_answer_type_tolerance(self):
        self.assertIsInstance(self.report.per_answer_type_tolerance, dict)
        self.assertGreater(len(self.report.per_answer_type_tolerance), 0)

    def test_answer_types_match_ground_truth(self):
        """The answer types in the report should match those in the data."""
        gt_types = {item.answer_type for item in self.ground_truth
                    if item.professor_mark != "unclear"}
        self.assertEqual(
            set(self.report.per_answer_type_exact.keys()), gt_types
        )

    def test_total_points_possible(self):
        """Report should track total points possible across scored items."""
        self.assertGreater(self.report.total_points_possible, 0)

    def test_total_points_awarded_by_professor(self):
        """Report should track total points the professor actually awarded."""
        self.assertGreater(self.report.total_points_professor, 0)
        self.assertLessEqual(
            self.report.total_points_professor,
            self.report.total_points_possible,
        )


if __name__ == "__main__":
    unittest.main()
