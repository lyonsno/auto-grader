"""Fail-first contract tests for Operation Zilch Reaper (forward lane).

Pins the new data-contract shape for truncated / unparseable grader output:
when the VLM runs out of its token budget (or otherwise emits output that
cannot be parsed into a valid ``model_score``), the resulting ``Prediction``
must record ``model_score is None``, ``model_confidence is None``, and
``truncated is True``. The old contract — ``model_score: 0.0`` +
``model_confidence: 0.0`` + a free-text "Grader output was truncated..."
``model_reasoning`` string — conflated "model said 0" with "model said
nothing," silently contaminating every aggregate metric downstream. See
``attractors/auto-grader_zilch-reaper-forward_stop-recording-truncated-
grader-output-as-model-score-zero_2026-04-11.md`` for the full framing.

These assertions target the integration smoke branch
(``cc/solipsism-integration-break-0410``), where the truncation-as-zero
behavior is live today (commit ``86fc0a0`` "grader: degrade truncated
items instead of crashing"). The failure prediction currently returned by
``grade_single_item`` on a length-truncated response carries
``model_score=0.0`` and ``model_confidence=0.0``, so these tests must
fail against the pre-fix implementation for the right reason (contract
assertion on the sentinel, not import / typo / unrelated error).

The companion test covering the scoring path — that ``score_predictions``
excludes truncated rows from MAE/accuracy denominators and reports
truncation rate as a separate first-class metric — lives in
``test_eval_harness_truncation_contract.py``.
"""

from __future__ import annotations

import unittest
from unittest import mock

from auto_grader.eval_harness import EvalItem
from auto_grader.vlm_inference import ServerConfig, grade_single_item


class _DummyResponse:
    def close(self) -> None:
        return None


def _item() -> EvalItem:
    return EvalItem(
        exam_id="15-blue",
        question_id="fr-10b",
        answer_type="numeric",
        page=1,
        professor_score=1.0,
        max_points=1.0,
        professor_mark="check",
        student_answer="1.0 mol",
        notes="",
    )


def _config() -> ServerConfig:
    return ServerConfig(base_url="http://example.test", model="qwen-test")


class TruncatedGraderOutputContract(unittest.TestCase):
    """When the grader's JSON output gets cut off at the token ceiling,
    the resulting Prediction must explicitly record "no prediction" via
    null sentinels, not "the model scored zero." Any consumer that wants
    to tell "model said 0" apart from "model said nothing" must be able
    to do so without string-matching on model_reasoning."""

    def test_length_truncation_records_null_score_not_zero(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=(
                '{"model_re',  # cut mid-key, unparseable
                "very long reasoning that burned the whole budget",
                "length",
            ),
        ):
            pred = grade_single_item(
                _item(),
                page_image=b"png",
                config=_config(),
            )

        # The model never committed to a score. Recording 0.0 here is a
        # lie — it reads as "model confidently said zero" downstream.
        self.assertIsNone(
            pred.model_score,
            "truncated rows must record model_score as None, not 0.0",
        )
        self.assertIsNone(
            pred.model_confidence,
            "truncated rows must record model_confidence as None, not 0.0",
        )

    def test_length_truncation_sets_truncated_flag_true(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=("", "reasoning", "length"),
        ):
            pred = grade_single_item(
                _item(),
                page_image=b"png",
                config=_config(),
            )

        self.assertTrue(
            getattr(pred, "truncated", False),
            "truncated rows must carry an explicit truncated=True flag "
            "so downstream consumers don't have to string-match on "
            "model_reasoning to detect them",
        )

    def test_length_truncation_preserves_raw_reasoning_for_forensics(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=(
                "",
                "very long reasoning that burned the whole budget",
                "length",
            ),
        ):
            pred = grade_single_item(
                _item(),
                page_image=b"png",
                config=_config(),
            )

        self.assertEqual(
            pred.raw_reasoning,
            "very long reasoning that burned the whole budget",
            "raw_reasoning must still be preserved verbatim on truncated "
            "rows so the post-hoc critic and human forensics can inspect "
            "what the model was chewing on when it ran out of tokens",
        )

    def test_length_truncation_does_not_raise(self):
        # Degrade-instead-of-crash is the existing integration-branch
        # behavior and must be preserved by the forward fix. A single
        # truncated item must not kill the whole grading run.
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=("", "reasoning", "length"),
        ):
            try:
                grade_single_item(
                    _item(),
                    page_image=b"png",
                    config=_config(),
                )
            except Exception as exc:
                self.fail(
                    f"grade_single_item must degrade instead of raising "
                    f"on truncated output, got {type(exc).__name__}: {exc}"
                )

    def test_unparseable_nonlength_output_also_records_null_sentinel(self):
        # finish_reason == "stop" with nonsense content is structurally
        # the same failure — the model didn't commit to a score. It
        # should carry the same sentinel shape, but we preserve the
        # distinction in whatever diagnostic field the implementation
        # uses. The contract requirement is: null score, null
        # confidence, truncated=True marks *it* as a non-prediction.
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=("definitely not json", "reasoning", "stop"),
        ):
            pred = grade_single_item(
                _item(),
                page_image=b"png",
                config=_config(),
            )

        self.assertIsNone(
            pred.model_score,
            "unparseable rows must record model_score as None so they "
            "are not silently counted as a confident zero in MAE",
        )
        self.assertIsNone(pred.model_confidence)
        self.assertTrue(
            getattr(pred, "truncated", False),
            "unparseable rows must carry truncated=True — "
            "the distinction between 'ran out of tokens' and 'emitted "
            "garbage' does not matter for the data contract; both mean "
            "'the model did not commit to a score'",
        )


if __name__ == "__main__":
    unittest.main()
