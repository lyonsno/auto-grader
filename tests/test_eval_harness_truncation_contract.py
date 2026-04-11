"""Fail-first contract tests for Operation Zilch Reaper (forward lane).

Companion to ``test_vlm_inference_truncation_contract.py``. That file
pins the grader-side contract (truncated output must produce a Prediction
with ``model_score=None`` / ``model_confidence=None`` / ``truncated=True``
instead of the old ``0.0`` + free-text lie). This file pins the
scoring-side contract: once those truncated rows exist in the eval run,
``score_predictions`` must exclude them from every accuracy / MAE /
calibration denominator and surface the truncation count as a separate
first-class field on ``EvalReport``.

The load-bearing failure mode this guards against: on an exam where
ground truth for a given item happens to be 0 (e.g. ``fr-11c``,
``fr-5b#2``, ``fr-3#2``), a truncation row currently scores as "correct
by accident" and improves the aggregate. On items where ground truth is
non-zero (e.g. ``fr-10b``, ``fr-10a#2``), truncation rows score as
"maximally wrong" and degrade the aggregate. A prompt change that
shifts the truncation rate will move aggregate metrics *independently
of any change in actual grading quality*. The fix: treat truncated rows
as "the model made no prediction," drop them from the denominator,
and report the truncation rate separately so operators can still see
completion rate as a first-class number.

These tests must fail against the pre-fix ``score_predictions`` because
it computes ``pred.model_score == truth`` on every prediction unconditionally,
so a truncated row with ``model_score=0.0`` is folded into accuracy exactly
the same way a real zero would be.
"""

from __future__ import annotations

import unittest

from auto_grader.eval_harness import (
    EvalItem,
    Prediction,
    score_predictions,
)


def _item(
    question_id: str,
    professor_score: float,
    max_points: float = 2.0,
) -> EvalItem:
    return EvalItem(
        exam_id="15-blue",
        question_id=question_id,
        answer_type="numeric",
        page=1,
        professor_score=professor_score,
        max_points=max_points,
        professor_mark="check",
        student_answer="placeholder",
        notes="",
    )


def _complete(
    question_id: str,
    model_score: float,
    *,
    model_confidence: float = 0.9,
) -> Prediction:
    return Prediction(
        exam_id="15-blue",
        question_id=question_id,
        model_score=model_score,
        model_confidence=model_confidence,
        model_reasoning="ok",
        model_read="ok",
    )


def _truncated(question_id: str) -> Prediction:
    # The forward-fix contract: truncated rows carry null sentinels for
    # score and confidence plus an explicit truncated=True flag. The
    # test constructs the post-fix shape directly — if the Prediction
    # dataclass does not yet accept these kwargs, that itself is a
    # valid fail-first signal that the contract has not landed.
    return Prediction(  # type: ignore[call-arg]
        exam_id="15-blue",
        question_id=question_id,
        model_score=None,
        model_confidence=None,
        model_reasoning="",
        model_read="",
        truncated=True,
    )


class TruncatedRowsExcludedFromEvalReport(unittest.TestCase):
    """score_predictions must treat truncated rows as non-predictions:
    not counted in the accuracy numerator, not counted in the accuracy
    denominator, not folded into calibration, and surfaced as a
    first-class truncation count on EvalReport so the operator can see
    completion rate at a glance."""

    def test_truncated_row_does_not_contaminate_exact_accuracy_when_truth_is_zero(self):
        # This is the load-bearing "looks correct by accident" case.
        # Ground truth is 0.0, the legacy contract records the
        # truncation row as model_score=0.0, and the pre-fix scorer
        # folds that into exact_matches — "100% accuracy!" when in
        # reality the model committed to nothing on that item.
        gt = [
            _item("fr-11c", professor_score=0.0),
            _item("fr-1", professor_score=2.0),
        ]
        preds = [
            _truncated("fr-11c"),
            _complete("fr-1", model_score=2.0),
        ]

        report = score_predictions(gt, preds)

        # Only the one complete row should count toward accuracy.
        self.assertEqual(
            report.total_scored,
            1,
            "truncated rows must not count in total_scored — the model "
            "made no prediction for them",
        )
        self.assertEqual(
            report.overall_exact_accuracy,
            1.0,
            "complete row matched truth exactly; accuracy is 1/1, not "
            "2/2 (which would be the pre-fix lie where a null-on-zero "
            "truncation counts as 'correct')",
        )

    def test_truncated_row_does_not_contaminate_exact_accuracy_when_truth_is_nonzero(self):
        # The symmetric case: ground truth is nonzero, the legacy
        # contract records the truncation as 0.0, and the pre-fix
        # scorer counts that as a wrong answer. That's also a lie —
        # the model didn't answer wrong, it didn't answer at all.
        gt = [
            _item("fr-10b", professor_score=1.0),
            _item("fr-1", professor_score=2.0),
        ]
        preds = [
            _truncated("fr-10b"),
            _complete("fr-1", model_score=2.0),
        ]

        report = score_predictions(gt, preds)

        self.assertEqual(report.total_scored, 1)
        self.assertEqual(
            report.overall_exact_accuracy,
            1.0,
            "only the complete row contributes; the truncated row is "
            "not a wrong answer, it is a non-answer, and must not drag "
            "the denominator",
        )

    def test_truncated_count_is_surfaced_on_eval_report(self):
        gt = [
            _item("fr-10b", professor_score=1.0),
            _item("fr-10a", professor_score=1.5),
            _item("fr-1", professor_score=2.0),
        ]
        preds = [
            _truncated("fr-10b"),
            _truncated("fr-10a"),
            _complete("fr-1", model_score=2.0),
        ]

        report = score_predictions(gt, preds)

        # The exact attribute name is part of the contract this lane
        # is landing. Using `truncated_excluded` to mirror the existing
        # `unclear_excluded` field — both are "items the scorer saw
        # but deliberately did not include in the accuracy denominator."
        self.assertEqual(
            getattr(report, "truncated_excluded", None),
            2,
            "EvalReport must surface truncated_excluded as a first-class "
            "field so operators can see completion rate without having "
            "to grep predictions.jsonl for a magic substring",
        )
        self.assertEqual(report.total_scored, 1)

    def test_truncated_rows_do_not_inflate_false_positives_or_negatives(self):
        gt = [
            _item("fr-10b", professor_score=1.0),
            _item("fr-11c", professor_score=0.0),
        ]
        preds = [
            _truncated("fr-10b"),   # would be false_negative under legacy
            _truncated("fr-11c"),   # would be "exact match" under legacy
        ]

        report = score_predictions(gt, preds)

        self.assertEqual(
            report.false_positives,
            0,
            "truncated rows contribute no predictions, so they cannot "
            "be false positives",
        )
        self.assertEqual(
            report.false_negatives,
            0,
            "truncated rows contribute no predictions, so they cannot "
            "be false negatives either — the pre-fix scorer counted "
            "the fr-10b row as a false negative because 0.0 < 1.0",
        )
        self.assertEqual(report.total_scored, 0)

    def test_truncated_rows_excluded_from_calibration_bins(self):
        # Calibration binning uses pred.model_confidence, which is None
        # on truncated rows. Feeding None into the bin comparator is
        # both semantically wrong ("confidence 0.0" is a claim the
        # model did not make) and a concrete runtime hazard (TypeError
        # on None < float in strict typing modes). The scorer must
        # drop truncated rows before calibration.
        gt = [
            _item("fr-1", professor_score=2.0),
            _item("fr-10b", professor_score=1.0),
        ]
        preds = [
            _complete("fr-1", model_score=2.0, model_confidence=0.9),
            _truncated("fr-10b"),
        ]

        # Pre-fix: this either raises on `None < 1.0` in calibration
        # binning, or silently bins a null-confidence row, depending
        # on implementation details. Both are wrong. Post-fix: the
        # call returns cleanly and the single complete row bins alone.
        report = score_predictions(gt, preds)

        total_binned = sum(b.count for b in report.calibration_bins)
        self.assertEqual(
            total_binned,
            1,
            "only the one complete row should appear in calibration "
            "bins; truncated rows are non-predictions and have no "
            "confidence to bin",
        )


if __name__ == "__main__":
    unittest.main()
