from __future__ import annotations

import json
from unittest import mock
import unittest

from auto_grader.eval_harness import EvalItem, Prediction
from auto_grader.thinking_narrator import ThinkingNarrator


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.delta_modes: list[str] = []
        self.rollbacks = 0
        self.commits: list[str] = []
        self.drops: list[tuple[str, str]] = []
        self.topics: list[tuple[str, str | None]] = []
        self.checkpoints: list[str] = []

    def write_delta(self, text: str, *, mode: str = "thought") -> None:
        self.deltas.append(text)
        self.delta_modes.append(mode)

    def rollback_live(self) -> None:
        self.rollbacks += 1

    def commit_live(self, *, mode: str = "thought") -> None:
        self.commits.append(mode)

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))

    def write_topic(self, text: str, verdict: str | None = None, **kwargs) -> None:
        self.topics.append((text, verdict))

    def write_checkpoint(self, text: str) -> None:
        self.checkpoints.append(text)


class _RetryNarrator(ThinkingNarrator):
    _PLAYBACK_CHUNK_DELAY_S = 0.0

    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)
        self.calls: list[str] = []

    def _chat_completion_stream(self, messages, on_delta, **kwargs):  # type: ignore[override]
        system = messages[0]["content"]
        if "present-participle status line" in system:
            text = "Rechecking the same unit conversion."
            self.calls.append("status")
        else:
            text = "I'm tracing the same unit conversion mistake."
            self.calls.append("thought")
        for token in text.split():
            on_delta(token + " ")
        return text


class _BadStatusRetryNarrator(ThinkingNarrator):
    _PLAYBACK_CHUNK_DELAY_S = 0.0

    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)

    def _chat_completion_stream(self, messages, on_delta, **kwargs):  # type: ignore[override]
        system = messages[0]["content"]
        if "present-participle status line" in system:
            return "I'm noticing the same unit conversion."
        return "I'm tracing the same unit conversion mistake."


class _AfterActionNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink, response: str) -> None:
        super().__init__(sink)
        self._response = response

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return self._response


class _CheckpointNarrator(ThinkingNarrator):
    _PLAYBACK_CHUNK_DELAY_S = 0.0

    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)
        self._thoughts = [
            "I'm tracing the limiting reagent path.",
            "I'm checking the NH3 mole ratio.",
            "I'm weighing the boxed answer against stoichiometry.",
            "I'm catching the moles-versus-grams mixup.",
        ]

    def _chat_completion_stream(self, messages, on_delta, **kwargs):  # type: ignore[override]
        text = self._thoughts.pop(0)
        for token in text.split():
            on_delta(token + " ")
        return text

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "Core issue: stoichiometry path is broken."


class ThinkingNarratorContract(unittest.TestCase):
    def test_duplicate_first_person_line_retries_as_status_and_commits(self):
        sink = _DummySink()
        narrator = _RetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._prior_statuses = ["Tracing the setup."]
        narrator._thoughts_since_status = [
            "I'm tracing the unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(narrator.calls, ["thought", "status"])
        self.assertEqual(sink.rollbacks, 0)
        self.assertEqual(sink.commits, ["status"])
        self.assertEqual(sink.drops, [])
        self.assertEqual(
            "".join(sink.deltas),
            "Rechecking the same unit conversion.",
        )
        self.assertEqual(sink.delta_modes, ["status"] * len(sink.deltas))
        self.assertEqual(
            narrator._prior_statuses[-1],
            "Rechecking the same unit conversion.",
        )
        self.assertEqual(narrator._thoughts_since_status, [])

    def test_status_retry_rejects_first_person_leak(self):
        sink = _DummySink()
        narrator = _BadStatusRetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._prior_statuses = ["Tracing the setup."]
        narrator._thoughts_since_status = [
            "I'm tracing the unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commits, [])
        self.assertEqual(
            sink.drops,
            [
                ("dedup", "I'm tracing the same unit conversion mistake."),
                ("dedup-status", "I'm noticing the same unit conversion."),
            ],
        )
        self.assertEqual(sink.deltas, [])

    def test_thought_prompt_uses_current_status_and_last_four_thoughts_only(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        narrator._current_status = "Tracing the net ionic form."
        prior_thoughts = [f"I'm thought {idx}." for idx in range(6)]

        user_content = narrator._build_thought_user_content("chunk", prior_thoughts)

        self.assertIn("Current status lane:\n- Tracing the net ionic form.", user_content)
        self.assertNotIn("I'm thought 0.", user_content)
        self.assertNotIn("I'm thought 1.", user_content)
        for idx in range(2, 6):
            self.assertIn(f"I'm thought {idx}.", user_content)

    def test_status_prompt_uses_last_five_statuses_only(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        prior_statuses = [f"Tracing status {idx}." for idx in range(7)]

        user_content = narrator._build_status_user_content("chunk", prior_statuses)

        self.assertNotIn("Tracing status 0.", user_content)
        self.assertNotIn("Tracing status 1.", user_content)
        for idx in range(2, 7):
            self.assertIn(f"Tracing status {idx}.", user_content)

    def test_double_dedup_sets_exponential_backoff(self):
        sink = _DummySink()
        narrator = _RetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._prior_statuses = ["Rechecking the same unit conversion."]
        narrator._thoughts_since_status = ["I'm tracing the same unit conversion mistake."]
        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commits, [])
        self.assertEqual(
            sink.drops,
            [
                ("dedup", "I'm tracing the same unit conversion mistake."),
                ("dedup-status", "Rechecking the same unit conversion."),
            ],
        )
        self.assertEqual(sink.deltas, [])
        self.assertEqual(narrator._dedupe_backoff_s, narrator._DEDUP_BACKOFF_INITIAL_S * 2)
        self.assertGreater(narrator._dedupe_backoff_until, 0.0)

    def test_feed_respects_dedupe_backoff_until_it_expires(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._buffer = "existing " * 250
        narrator._last_dispatch = 0.0
        narrator._dedupe_backoff_until = 15.0

        with mock.patch("auto_grader.thinking_narrator.time.monotonic", return_value=10.0):
            narrator.feed("token ")
        self.assertFalse(narrator._pending_dispatch)
        self.assertGreater(len(narrator._buffer), 0)

        with mock.patch("auto_grader.thinking_narrator.threading.Thread") as thread_mock:
            thread = thread_mock.return_value
            with mock.patch("auto_grader.thinking_narrator.time.monotonic", return_value=16.0):
                narrator.feed("token ")

        thread_mock.assert_called_once()
        thread.start.assert_called_once()

    def test_streaming_bonsai_default_sampler_splits_the_difference(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)

        def _raise_and_capture(request, timeout):
            body = json.loads(request.data.decode())
            self.assertEqual(body["temperature"], 0.8)
            self.assertEqual(body["presence_penalty"], 1.0)
            self.assertEqual(body["repetition_penalty"], 1.005)
            raise RuntimeError("boom")

        with mock.patch("urllib.request.urlopen", side_effect=_raise_and_capture):
            with self.assertRaises(RuntimeError):
                narrator._chat_completion_stream([{"role": "system", "content": "x"}], lambda _: None)

    def test_after_action_bonsai_call_uses_split_difference_sampler(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        prediction = type(
            "P",
            (),
            {"model_score": 2, "model_read": "6.98 g/mL", "model_reasoning": "Used m/V cleanly."},
        )()
        item = type(
            "I",
            (),
            {
                "professor_score": 2,
                "exam_id": "15-blue",
                "question_id": "fr-1",
                "answer_type": "numeric",
                "max_points": 2.0,
                "student_answer": "6.98 g/mL",
                "professor_mark": "2/2",
                "notes": "clean",
            },
        )()
        with mock.patch.object(narrator, "_chat_completion", return_value="Tracing the answer.") as completion_mock:
            narrator._produce_after_action(
                12.0,
                prediction=prediction,
                item=item,
                template_question={"answer": "density", "notes": "n/a"},
            )

        self.assertEqual(completion_mock.call_args.kwargs["temperature"], 0.8)
        self.assertEqual(completion_mock.call_args.kwargs["presence_penalty"], 1.0)
        self.assertEqual(completion_mock.call_args.kwargs["repetition_penalty"], 1.005)

    def test_first_person_prompt_discourages_defaulting_to_im_noticing(self):
        prompt = ThinkingNarrator._compose_system_prompt(ThinkingNarrator(_DummySink()))
        self.assertIn(
            'Do not default to "I\'m noticing" or "I\'m seeing"; use stronger verbs unless the point is literal OCR or legibility.',
            prompt,
        )

    def test_every_fourth_accepted_line_emits_history_checkpoint(self):
        sink = _DummySink()
        narrator = _CheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-5b")

        for idx in range(4):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(
            sink.checkpoints,
            ["Core issue: stoichiometry path is broken."],
        )

    def test_after_action_prompt_avoids_indexable_stock_examples(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        prediction = type(
            "P",
            (),
            {"model_score": 2, "model_read": "6.98 g/mL", "model_reasoning": "Used m/V cleanly."},
        )()
        item = type(
            "I",
            (),
            {
                "professor_score": 2,
                "exam_id": "15-blue",
                "question_id": "fr-1",
                "answer_type": "numeric",
                "max_points": 2.0,
                "student_answer": "6.98 g/mL",
                "professor_mark": "2/2",
                "notes": "clean",
            },
        )()

        with mock.patch.object(narrator, "_chat_completion", return_value="Tracing the answer.") as completion_mock:
            narrator._produce_after_action(
                12.0,
                prediction=prediction,
                item=item,
                template_question={"answer": "density", "notes": "n/a"},
            )

        user_prompt = completion_mock.call_args.args[0][1]["content"]
        self.assertNotIn("Examples:", user_prompt)
        self.assertNotIn("Even the 1-bit kid called this one.", user_prompt)
        self.assertNotIn("judges in lockstep", user_prompt)
        self.assertIn("Prefer no coda to a stock phrase.", user_prompt)
        self.assertIn("When the scores differ, explicitly describe the disagreement.", user_prompt)

    def test_after_action_normalizes_score_denominators_to_item_max_points(self):
        sink = _DummySink()
        narrator = _AfterActionNarrator(
            sink,
            "Grader: 0.5/1 (correct relationship but wrong sign and units). "
            "Prof: 1.5/2 (partial credit for setup, full credit for execution). "
            "· student plain missed sign and units, judges in lockstep.",
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-10a",
            answer_type="numeric",
            page=1,
            professor_score=1.5,
            max_points=3.0,
            professor_mark="partial",
            student_answer="...",
            notes="",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-10a",
            model_score=0.5,
            model_confidence=0.9,
            model_reasoning="Correct formula, wrong energy value.",
            model_read="E = hν ...",
        )

        narrator._produce_after_action(146.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.topics,
            [
                (
                    "146s · Grader: 0.5/3 (correct relationship but wrong sign and units). "
                    "Prof: 1.5/3 (partial credit for setup, full credit for execution). "
                    "· student plain missed sign and units, judges in lockstep.",
                    "undershoot",
                )
            ],
        )

    def test_after_action_clamps_multiline_repeat_to_one_normalized_line(self):
        sink = _DummySink()
        narrator = _AfterActionNarrator(
            sink,
            "Grader: 0/2 (student added moles instead of using stoichiometry).\n"
            "Prof: 2/2 (same reasoning). · student plainly missed stoichiometry, judges in lockstep.\n"
            "Grader: 0/2 (student added moles instead of using stoichiometry). "
            "Prof: 2/2 (same reasoning). · student plainly missed stoichiometry, judges in lockstep.\n"
            "Grader:",
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=1,
            professor_score=0.0,
            max_points=2.0,
            professor_mark="0/2",
            student_answer="...",
            notes="same reasoning",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=0.0,
            model_confidence=0.9,
            model_reasoning="Used addition instead of stoichiometry.",
            model_read="13.839 + 13.839",
        )

        narrator._produce_after_action(287.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.topics,
            [
                (
                    "287s · Grader: 0/2 (student added moles instead of using stoichiometry). "
                    "Prof: 0/2 (same reasoning). · student plainly missed stoichiometry, judges in lockstep.",
                    "match",
                )
            ],
        )

    def test_after_action_uses_truth_for_corrected_historical_scores(self):
        sink = _DummySink()
        narrator = _AfterActionNarrator(
            sink,
            "Grader: 4/4 (correct Hess's Law combination). "
            "Truth: 4/4 (corrected after review). · "
            "Historical prof: 2/4 (batch-mark partial for confused work).",
        )
        item = EvalItem(
            exam_id="34-blue",
            question_id="fr-8",
            answer_type="numeric",
            page=2,
            professor_score=2.0,
            max_points=4.0,
            professor_mark="partial",
            student_answer="-186.2 kJ",
            notes="Partial. Correct answer but confused intermediate work.",
            corrected_score=4.0,
            correction_reason="Reviewed from page image: Hess's Law reversal, cancellation, and final enthalpy are coherent and correct.",
        )
        prediction = Prediction(
            exam_id="34-blue",
            question_id="fr-8",
            model_score=4.0,
            model_confidence=1.0,
            model_reasoning="Correct Hess's Law manipulation and final enthalpy.",
            model_read="SnCl2 + Cl2 -> SnCl4 ; -186.2 kJ",
        )

        narrator._produce_after_action(50.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.topics,
            [
                (
                    "50s · Grader: 4/4 (correct Hess's Law combination). "
                    "Truth: 4/4 (corrected after review). · "
                    "Historical prof: 2/4 (batch-mark partial for confused work).",
                    "match",
                )
            ],
        )

    def test_after_action_prompt_mentions_corrected_truth_when_present(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        item = EvalItem(
            exam_id="34-blue",
            question_id="fr-8",
            answer_type="numeric",
            page=2,
            professor_score=2.0,
            max_points=4.0,
            professor_mark="partial",
            student_answer="-186.2 kJ",
            notes="Partial. Correct answer but confused intermediate work.",
            corrected_score=4.0,
            correction_reason="Reviewed from page image: Hess's Law reversal, cancellation, and final enthalpy are coherent and correct.",
        )
        prediction = Prediction(
            exam_id="34-blue",
            question_id="fr-8",
            model_score=4.0,
            model_confidence=1.0,
            model_reasoning="Correct Hess's Law manipulation and final enthalpy.",
            model_read="SnCl2 + Cl2 -> SnCl4 ; -186.2 kJ",
        )

        with mock.patch.object(
            narrator,
            "_chat_completion",
            return_value="Grader: 4/4 (correct Hess's Law combination). Truth: 4/4 (corrected after review). · Historical prof: 2/4 (batch-mark partial for confused work).",
        ) as completion_mock:
            narrator._produce_after_action(50.0, prediction, item, template_question=None)

        user_prompt = completion_mock.call_args.args[0][1]["content"]
        self.assertIn("Truth awarded: 4.0 pts", user_prompt)
        self.assertIn("Historical professor awarded: 2.0 pts", user_prompt)
        self.assertIn("Correction reason:", user_prompt)
        self.assertIn("Truth: <score>", user_prompt)


if __name__ == "__main__":
    unittest.main()
