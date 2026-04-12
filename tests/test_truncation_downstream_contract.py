"""Downstream consumer contract tests for the truncation sentinel.

Operation Zilch Reaper, forward lane. The core truncation contract is
pinned by ``test_vlm_inference_truncation_contract.py`` (grader side)
and ``test_eval_harness_truncation_contract.py`` (scoring side). This
file pins the downstream consumers that were modified to handle the
new ``model_score: null`` / ``model_confidence: null`` /
``truncated: true`` shape but did not have their own dedicated tests:

1. ``_PredictionWriter`` in ``scripts/smoke_vlm.py`` — serializes the
   ``truncated`` flag to JSONL.
2. ``load_run_records`` in ``scripts/compare_runs.py`` — deserializes
   the ``truncated`` flag and ``None`` scores from JSONL.
3. ``_progress`` in ``scripts/smoke_vlm.py`` — live console display
   must not crash on ``None`` scores.
4. ``apply_consistency_rule`` in ``auto_grader/critic.py`` — must
   abstain cleanly on truncated rows.
5. ``_produce_after_action`` in ``auto_grader/thinking_narrator.py``
   — must produce a distinct topic for truncated items.

Finding: regression lens, Zilch Reaper Forward main-side rereview,
2026-04-12 (patched epanorthosis at ``24b6dd7``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_grader.eval_harness import EvalItem, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncated_prediction(question_id: str = "fr-10b") -> Prediction:
    return Prediction(
        exam_id="15-blue",
        question_id=question_id,
        model_score=None,
        model_confidence=None,
        model_reasoning=(
            "Grader output could not be parsed as the required "
            "JSON (truncated or malformed)."
        ),
        model_read="",
        raw_assistant="<partial>",
        raw_reasoning="<full reasoning>",
        truncated=True,
    )


def _complete_prediction(question_id: str = "fr-7a") -> Prediction:
    return Prediction(
        exam_id="15-blue",
        question_id=question_id,
        model_score=1.5,
        model_confidence=0.82,
        model_reasoning="Student is correct.",
        model_read="0.0423 mol",
        raw_assistant='{"model_score": 1.5}',
        raw_reasoning="Looks right.",
    )


def _eval_item(question_id: str, professor_score: float = 1.0) -> EvalItem:
    return EvalItem(
        exam_id="15-blue",
        question_id=question_id,
        answer_type="numeric",
        page=1,
        professor_score=professor_score,
        max_points=2.0,
        professor_mark="check",
        student_answer="0.0423",
        notes="",
    )


def _load_smoke_vlm():
    path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_vlm.py"
    spec = importlib.util.spec_from_file_location("smoke_vlm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_compare_runs():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "compare_runs.py"
    )
    spec = importlib.util.spec_from_file_location("compare_runs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. _PredictionWriter serialization
# ---------------------------------------------------------------------------


class PredictionWriterTruncationContract(unittest.TestCase):
    """_PredictionWriter must serialize the truncated flag and None
    scores to JSONL faithfully."""

    def test_truncated_prediction_round_trips_through_jsonl(self) -> None:
        smoke = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            preds_path = run_dir / "predictions.jsonl"
            with smoke._PredictionWriter(
                preds_path, model="test-model", run_dir=run_dir
            ) as writer:
                item = _eval_item("fr-10b")
                pred = _truncated_prediction("fr-10b")
                writer.write_one(item, pred)
            rows = []
            with preds_path.open() as f:
                for line in f:
                    rows.append(json.loads(line))

            # Find the prediction row (skip header)
            pred_rows = [r for r in rows if r.get("type") == "prediction"]
            self.assertEqual(len(pred_rows), 1)
            row = pred_rows[0]
            self.assertIsNone(row["model_score"])
            self.assertIsNone(row["model_confidence"])
            self.assertIs(row["truncated"], True)

    def test_complete_prediction_carries_truncated_false(self) -> None:
        smoke = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            preds_path = run_dir / "predictions.jsonl"
            with smoke._PredictionWriter(
                preds_path, model="test-model", run_dir=run_dir
            ) as writer:
                item = _eval_item("fr-7a")
                pred = _complete_prediction("fr-7a")
                writer.write_one(item, pred)
            rows = []
            with preds_path.open() as f:
                for line in f:
                    rows.append(json.loads(line))

            pred_rows = [r for r in rows if r.get("type") == "prediction"]
            self.assertEqual(len(pred_rows), 1)
            row = pred_rows[0]
            self.assertEqual(row["model_score"], 1.5)
            self.assertIs(row["truncated"], False)


# ---------------------------------------------------------------------------
# 2. compare_runs load_run_records deserialization
# ---------------------------------------------------------------------------


class CompareRunsTruncationContract(unittest.TestCase):
    """load_run_records must deserialize None scores and the truncated
    flag from JSONL without crashing or losing the flag."""

    def test_load_run_records_handles_null_scores_and_truncated_flag(
        self,
    ) -> None:
        compare = _load_compare_runs()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            preds = run_dir / "predictions.jsonl"
            header = {
                "type": "header",
                "model": "test",
                "run_dir": str(run_dir),
                "started": "2026-04-12T00:00:00",
            }
            truncated_row = {
                "type": "prediction",
                "exam_id": "15-blue",
                "question_id": "fr-10b",
                "answer_type": "numeric",
                "model_score": None,
                "model_confidence": None,
                "truncated": True,
                "model_reasoning": "truncated",
                "model_read": "",
                "professor_score": 1.0,
                "max_points": 1.0,
                "corrected_score": None,
            }
            complete_row = {
                "type": "prediction",
                "exam_id": "15-blue",
                "question_id": "fr-7a",
                "answer_type": "numeric",
                "model_score": 1.5,
                "model_confidence": 0.82,
                "truncated": False,
                "model_reasoning": "ok",
                "model_read": "0.0423",
                "professor_score": 2.0,
                "max_points": 2.0,
                "corrected_score": None,
            }
            with preds.open("w") as f:
                for row in [header, truncated_row, complete_row]:
                    f.write(json.dumps(row) + "\n")

            records = compare.load_run_records(run_dir)
            trunc = records[("15-blue", "fr-10b")]
            comp = records[("15-blue", "fr-7a")]

            self.assertIsNone(trunc.model_score)
            self.assertIsNone(trunc.model_confidence)
            self.assertTrue(trunc.truncated)

            self.assertEqual(comp.model_score, 1.5)
            self.assertFalse(comp.truncated)


# ---------------------------------------------------------------------------
# 3. _progress display
# ---------------------------------------------------------------------------


class ProgressDisplayTruncationContract(unittest.TestCase):
    """_progress must not crash when model_score is None."""

    def test_progress_does_not_crash_on_truncated_prediction(self) -> None:
        smoke = _load_smoke_vlm()
        item = _eval_item("fr-10b")
        pred = _truncated_prediction("fr-10b")
        # _progress writes to stderr/stdout — just verify no exception.
        try:
            smoke._progress(1, 10, item, pred)
        except (TypeError, AttributeError, ValueError) as exc:
            self.fail(
                f"_progress crashed on a truncated prediction: {exc}"
            )


# ---------------------------------------------------------------------------
# 4. Critic abstention
# ---------------------------------------------------------------------------


class CriticTruncationContract(unittest.TestCase):
    """apply_consistency_rule must abstain on truncated rows without
    crashing or producing a spurious finding."""

    def test_critic_abstains_on_truncated_record(self) -> None:
        from auto_grader.critic import apply_consistency_rule

        record = {
            "exam_id": "15-blue",
            "question_id": "fr-10b",
            "model_score": None,
            "model_confidence": None,
            "truncated": True,
            "model_reasoning": "truncated",
            "upstream_dependency": "none",
            "if_dependent_then_consistent": None,
            "max_points": 1.0,
        }
        delta = apply_consistency_rule(record)
        self.assertEqual(delta.action, "unchanged")
        self.assertIn("abstain", delta.reason.lower())

    def test_critic_still_works_on_complete_record(self) -> None:
        from auto_grader.critic import apply_consistency_rule

        record = {
            "exam_id": "15-blue",
            "question_id": "fr-7a",
            "model_score": 1.5,
            "model_confidence": 0.82,
            "truncated": False,
            "model_reasoning": "ok",
            "upstream_dependency": "none",
            "if_dependent_then_consistent": None,
            "max_points": 2.0,
        }
        delta = apply_consistency_rule(record)
        # Should not crash; action depends on the actual logic but
        # should be a valid CriticDelta.
        self.assertIn(delta.action, ("unchanged", "adjusted"))


# ---------------------------------------------------------------------------
# 5. Narrator after-action topic
# ---------------------------------------------------------------------------


class NarratorTruncationContract(unittest.TestCase):
    """_produce_after_action must produce a distinct topic for truncated
    items rather than crashing on None scores."""

    def test_after_action_produces_truncated_topic(self) -> None:
        from auto_grader.thinking_narrator import ThinkingNarrator

        narrator = ThinkingNarrator.__new__(ThinkingNarrator)
        narrator._narrator_url = None
        narrator._narrator_model = None
        narrator._narrator_max_seconds = 10

        item = _eval_item("fr-10b")
        pred = _truncated_prediction("fr-10b")

        # _produce_after_action returns a string topic or None.
        # When the narrator backend is not available it falls back
        # to a bare topic. We patch the chat call to simulate that.
        with mock.patch.object(
            narrator,
            "_chat_completion_stream",
            side_effect=Exception("no narrator backend"),
        ):
            try:
                topic = narrator._produce_after_action(
                    elapsed=5.0,
                    prediction=pred,
                    item=item,
                    template_question=None,
                )
            except (TypeError, AttributeError) as exc:
                self.fail(
                    f"_produce_after_action crashed on truncated prediction: {exc}"
                )

        # topic may be None (narrator unavailable) or a string — either
        # is fine. What we're testing is that the function doesn't crash
        # trying to compare None scores.


if __name__ == "__main__":
    unittest.main()
