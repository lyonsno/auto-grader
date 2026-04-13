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
        self.basis_rows: list[str] = []
        self.ambiguity_rows: list[str] = []
        self.credit_preserved_rows: list[str] = []
        self.deduction_rows: list[str] = []
        self.review_markers: list[str] = []
        self.professor_mismatches: list[str] = []

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

    def write_basis(self, text: str) -> None:
        self.basis_rows.append(text)

    def write_ambiguity(self, text: str) -> None:
        self.ambiguity_rows.append(text)

    def write_credit_preserved(self, text: str) -> None:
        self.credit_preserved_rows.append(text)

    def write_deduction(self, text: str) -> None:
        self.deduction_rows.append(text)

    def write_review_marker(self, text: str) -> None:
        self.review_markers.append(text)

    def write_professor_mismatch(self, text: str) -> None:
        self.professor_mismatches.append(text)


class _LegacyStructuredRowSink:
    def __init__(self) -> None:
        self.drops: list[tuple[str, str]] = []

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))

    def write_basis(self, text: str) -> None:
        pass

    def write_ambiguity(self, text: str) -> None:
        pass

    def write_review_marker(self, text: str) -> None:
        pass


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


class _QueuedLegibilityNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink, responses: list[str]) -> None:
        super().__init__(sink)
        self._responses = responses
        self.prompts: list[str] = []

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        self.prompts.append(messages[-1]["content"])
        return self._responses.pop(0)


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


class _DuplicateCheckpointNarrator(ThinkingNarrator):
    _PLAYBACK_CHUNK_DELAY_S = 0.0

    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)
        self._thoughts = [
            "I'm tracing the limiting reagent path.",
            "I'm checking the NH3 mole ratio.",
            "I'm weighing the boxed answer against stoichiometry.",
            "I'm catching the moles-versus-grams mixup.",
            "I'm checking whether the boxed unit is missing.",
            "I'm tracing the carry-forward from part 5a.",
            "I'm eyeing the stoichiometric coefficient on NH3.",
            "I'm wondering whether the reaction setup itself drifted.",
        ]

    def _chat_completion_stream(self, messages, on_delta, **kwargs):  # type: ignore[override]
        text = self._thoughts.pop(0)
        for token in text.split():
            on_delta(token + " ")
        return text

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "Core issue: stoichiometry path is broken."


class _FirstPersonCheckpointNarrator(_CheckpointNarrator):
    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "I'm noticing the student's answer is written in mL instead of cm³."


class _StatusLikeCheckpointNarrator(_CheckpointNarrator):
    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "Rechecking ion balancing for net ionic equation."


class _RunObservedUnitConversionCheckpointNarrator(_CheckpointNarrator):
    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "I'm tracing the student's unit conversions — they used 28.014 for N2 and 2.016 for H2, but the question asked for moles of NH3, not N2."


class _RunObservedDashPrefixedCheckpointNarrator(_CheckpointNarrator):
    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "- I'm noticing the student used the initial energy instead of the energy change for the photon calculation."


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

    def test_status_retry_rejects_first_person_leak_as_contract_failure(self):
        sink = _DummySink()
        narrator = _BadStatusRetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._thoughts_since_status = [
            "I'm tracing the unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commits, [])
        self.assertEqual(
            sink.drops,
            [
                ("dedup", "I'm tracing the same unit conversion mistake."),
                ("contract-status", "I'm noticing the same unit conversion."),
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

    def test_double_dedup_triggers_checkpoint_grooming_without_backoff(self):
        sink = _DummySink()
        narrator = _RetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._current_status = "Rechecking the same unit conversion."
        narrator._prior_statuses = ["Rechecking the same unit conversion."]
        narrator._thoughts_since_status = ["I'm tracing the same unit conversion mistake."]
        narrator._dedupe_streak = 1
        narrator._chat_completion = mock.Mock(  # type: ignore[method-assign]
            return_value="Core issue: unit conversion path is looping on the same mismatch."
        )

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
        self.assertEqual(
            sink.checkpoints,
            ["Core issue: unit conversion path is looping on the same mismatch."],
        )
        self.assertEqual(narrator._dedupe_streak, 2)
        self.assertEqual(narrator._item_turbulence_dedup_count, 1)
        self.assertEqual(narrator._item_turbulence_status_dedup_count, 1)
        self.assertEqual(narrator._item_turbulence_grooming_count, 1)

    def test_feed_keeps_dispatching_during_dedupe_streak(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._buffer = "existing " * 250
        narrator._last_dispatch = 0.0
        narrator._dedupe_streak = 2

        with mock.patch("auto_grader.thinking_narrator.threading.Thread") as thread_mock:
            thread = thread_mock.return_value
            with mock.patch("auto_grader.thinking_narrator.time.monotonic", return_value=10.0):
                narrator.feed("token ")

        thread_mock.assert_called_once()
        thread.start.assert_called_once()

    def test_accepted_line_resets_dedupe_streak(self):
        sink = _DummySink()
        narrator = _CheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._dedupe_streak = 2
        narrator._prior_statuses = ["Tracing the setup."]

        narrator._dispatch("fresh reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(narrator._dedupe_streak, 0)

    def test_idle_legibility_missing_sink_writer_drops_row_instead_of_crashing(self):
        sink = _LegacyStructuredRowSink()
        narrator = ThinkingNarrator(sink)  # type: ignore[arg-type]
        narrator._legibility_jobs = [
            {
                "kind": "generated",
                "row_type": "deduction",
                "prompt": "Write one short row body for 'Deduction:'.",
            }
        ]
        narrator._chat_completion = mock.Mock(return_value="Wrong final target species.")  # type: ignore[method-assign]

        emitted = narrator._flush_idle_legibility_once()

        self.assertFalse(emitted)
        self.assertEqual(
            sink.drops,
            [("missing-sink-row", "deduction: Wrong final target species.")],
        )

    def test_immediate_legibility_missing_sink_writer_drops_row_instead_of_crashing(self):
        sink = _LegacyStructuredRowSink()
        narrator = ThinkingNarrator(sink)  # type: ignore[arg-type]

        emitted = narrator._write_legibility_row_now(
            "professor_mismatch",
            "Historical professor awarded 2/4; corrected truth is 4/4.",
        )

        self.assertFalse(emitted)
        self.assertEqual(
            sink.drops,
            [("missing-sink-row", "professor_mismatch: Historical professor awarded 2/4; corrected truth is 4/4.")],
        )

    def test_idle_legibility_thread_logs_flush_exception_without_unboundlocal(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)
        narrator._idle_legibility_pending = True
        narrator._legibility_jobs = [{"kind": "literal", "row_type": "basis", "text": "x"}]

        with mock.patch("auto_grader.thinking_narrator.time.sleep"):
            with mock.patch.object(
                narrator,
                "_flush_idle_legibility_once",
                side_effect=RuntimeError("boom"),
            ):
                with mock.patch("auto_grader.thinking_narrator.logger.exception") as exc_mock:
                    narrator._run_idle_legibility_after_delay(narrator._idle_legibility_generation)

        exc_mock.assert_called_once()
        self.assertFalse(narrator._idle_legibility_pending)

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

    def test_duplicate_checkpoint_is_dropped_instead_of_persisted_twice(self):
        sink = _DummySink()
        narrator = _DuplicateCheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-5b")

        for idx in range(8):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(
            sink.checkpoints,
            ["Core issue: stoichiometry path is broken."],
        )
        self.assertIn(
            ("dedup-checkpoint", "Core issue: stoichiometry path is broken."),
            sink.drops,
        )

    def test_checkpoint_bonsai_call_uses_colder_more_canonical_sampler(self):
        sink = _DummySink()
        narrator = _CheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-5b")

        with mock.patch.object(
            narrator,
            "_chat_completion",
            return_value="Core issue: stoichiometry path is broken.",
        ) as completion_mock:
            for idx in range(4):
                narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(completion_mock.call_args.kwargs["temperature"], 0.5)
        self.assertEqual(completion_mock.call_args.kwargs["presence_penalty"], 0.0)
        self.assertEqual(completion_mock.call_args.kwargs["repetition_penalty"], 1.0)

    def test_checkpoint_prompt_prefers_reusing_canonical_wording_over_paraphrase(self):
        self.assertIn(
            "If the issue matches a recent checkpoint, reuse its wording instead of paraphrasing it.",
            ThinkingNarrator._build_checkpoint_user_content(
                ThinkingNarrator(_DummySink()),
                "chunk",
                "I'm tracing the limiting reagent path.",
                ["I'm tracing the limiting reagent path."],
                ["Checking NH3 stoichiometry."],
                ["Core issue: stoichiometry path is broken."],
            ),
        )

    def test_first_person_checkpoint_is_dropped_instead_of_persisted(self):
        sink = _DummySink()
        narrator = _FirstPersonCheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")

        for idx in range(4):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(sink.checkpoints, [])
        self.assertIn(
            (
                "contract-checkpoint",
                "I'm noticing the student's answer is written in mL instead of cm³.",
            ),
            sink.drops,
        )

    def test_unlabeled_status_like_checkpoint_is_dropped_instead_of_persisted(self):
        sink = _DummySink()
        narrator = _StatusLikeCheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-3")

        for idx in range(4):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(sink.checkpoints, [])
        self.assertIn(
            (
                "contract-checkpoint",
                "Rechecking ion balancing for net ionic equation.",
            ),
            sink.drops,
        )

    def test_run_observed_first_person_checkpoint_shape_is_rejected(self):
        sink = _DummySink()
        narrator = _RunObservedUnitConversionCheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-3")

        for idx in range(4):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(sink.checkpoints, [])
        self.assertIn(
            (
                "contract-checkpoint",
                "I'm tracing the student's unit conversions — they used 28.014 for N2 and 2.016 for H2, but the question asked for moles of NH3, not N2.",
            ),
            sink.drops,
        )

    def test_run_observed_dash_prefixed_checkpoint_shape_is_rejected(self):
        sink = _DummySink()
        narrator = _RunObservedDashPrefixedCheckpointNarrator(sink)
        narrator.start(item_header="15-blue/fr-10a")

        for idx in range(4):
            narrator._dispatch(f"chunk {idx}", narrator._dispatch_generation)

        self.assertEqual(sink.checkpoints, [])
        self.assertIn(
            (
                "contract-checkpoint",
                "- I'm noticing the student used the initial energy instead of the energy change for the photon calculation.",
            ),
            sink.drops,
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

    def test_idle_legibility_queue_emits_generated_partial_credit_rows_after_basis(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 1/2 (correct setup, wrong final species). Prof: 1/2 (same).",
                "Correct stoichiometric setup and mole relationship.",
                "Final answer awards NH3 credit to reactant-mole addition.",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=2.0,
            professor_mark="1/2",
            student_answer="14.2031 moles",
            notes="partial",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=1.0,
            model_confidence=0.8,
            model_reasoning="Correct setup, wrong final target species.",
            model_read="14.2031 moles",
            score_basis="Correct setup, lost final credit for target-species drift.",
        )

        narrator._produce_after_action(20.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.basis_rows,
            ["Correct setup, lost final credit for target-species drift."],
        )

        self.assertTrue(narrator._flush_idle_legibility_once())
        self.assertTrue(narrator._flush_idle_legibility_once())
        self.assertFalse(narrator._flush_idle_legibility_once())

        self.assertEqual(
            sink.credit_preserved_rows,
            ["Correct stoichiometric setup and mole relationship."],
        )
        self.assertEqual(
            sink.deduction_rows,
            ["Final answer awards NH3 credit to reactant-mole addition."],
        )

    def test_after_action_collapses_clean_full_credit_match_without_basis_row(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 3/3 (correct ground-state configuration). Prof: 3/3 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-11a",
            answer_type="electron_config",
            page=1,
            professor_score=3.0,
            max_points=3.0,
            professor_mark="3/3",
            student_answer="[Ne] 3s^2 3p^5 with correct orbital boxes",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-11a",
            model_score=3.0,
            model_confidence=0.98,
            model_reasoning="Correct noble gas core and valence orbital box notation.",
            model_read="[Ne] 3s^2 3p^5",
            score_basis="Correct noble gas core and valence orbital box notation (3s^2 3p^5).",
        )

        narrator._produce_after_action(55.0, prediction, item, template_question=None)

        self.assertEqual(sink.basis_rows, [])
        self.assertEqual(sink.review_markers, [])
        self.assertEqual(sink.professor_mismatches, [])
        self.assertEqual(sink.credit_preserved_rows, [])
        self.assertEqual(sink.deduction_rows, [])
        self.assertEqual(narrator._legibility_jobs, [])

    def test_after_action_emits_professor_mismatch_immediately_without_idle_queue(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 4/4 (correct Hess's Law combination). Truth: 4/4 (corrected after review). · Historical prof: 2/4 (batch-mark partial).",
            ],
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
            correction_reason="Reviewed from page image: Hess's Law reversal and cancellation are coherent.",
        )
        prediction = Prediction(
            exam_id="34-blue",
            question_id="fr-8",
            model_score=4.0,
            model_confidence=0.45,
            model_reasoning="After one careful pass, human review is warranted because the cancellation handwriting is ambiguous.",
            model_read="-186.2 kJ",
            score_basis="Correct Hess's Law manipulation and final enthalpy.",
        )

        narrator._produce_after_action(50.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.basis_rows,
            ["Correct Hess's Law manipulation and final enthalpy."],
        )
        self.assertEqual(
            sink.review_markers,
            ["Human review warranted because the cancellation handwriting remains ambiguity-sensitive after a bounded pass."],
        )
        self.assertEqual(
            sink.professor_mismatches,
            ["Historical professor awarded 2/4; corrected truth is 4/4."],
        )
        self.assertEqual(narrator._legibility_jobs, [])

    def test_after_action_does_not_emit_review_needed_from_low_confidence_alone(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 0/0.5 (student answered 2, correct answer is 1). Prof: 0/0.5 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-11c",
            answer_type="exact_match",
            page=1,
            professor_score=0.0,
            max_points=0.5,
            professor_mark="0/0.5",
            student_answer="2",
            notes="same miss",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-11c",
            model_score=0.0,
            model_confidence=0.22,
            model_reasoning="The student answered 2, but the correct answer is 1.",
            model_read="2",
            score_basis="Student answered 2; correct answer is 1.",
        )

        narrator._produce_after_action(188.0, prediction, item, template_question=None)

        self.assertEqual(sink.review_markers, [])

    def test_after_action_queues_ambiguity_row_from_turbulence_context(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 0/0.5 (student answered 2, correct answer is 1). Prof: 0/0.5 (same).",
                "Crossed-out cancellation digit kept reading as either 1 or 2, which changes the count.",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-11c",
            answer_type="exact_match",
            page=1,
            professor_score=0.0,
            max_points=0.5,
            professor_mark="0/0.5",
            student_answer="2",
            notes="same miss",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-11c",
            model_score=0.0,
            model_confidence=0.55,
            model_reasoning="The cancellation mark makes one digit hard to read, but the final keyed answer is 1.",
            model_read="2",
            score_basis="Student answered 2; keyed answer is 1.",
        )
        narrator._prior_statuses = [
            "Rechecking the crossed-out cancellation digit.",
            "Staying on whether the digit is 1 or 2.",
        ]
        narrator._thoughts_since_status = [
            "I'm torn between a 1 and a 2 in the cancellation trail.",
        ]
        narrator._prior_checkpoints = [
            "Core issue: crossed-out digit changes the final count.",
        ]
        narrator._item_turbulence_dedup_count = 1
        narrator._item_turbulence_status_dedup_count = 1
        narrator._item_turbulence_grooming_count = 1

        narrator._produce_after_action(188.0, prediction, item, template_question=None)

        self.assertEqual(
            [job["row_type"] for job in narrator._legibility_jobs],
            ["ambiguity", "deduction"],
        )
        self.assertTrue(narrator._flush_idle_legibility_once())
        self.assertIn(
            "Rechecking the crossed-out cancellation digit.",
            narrator.prompts[-1],
        )
        self.assertIn(
            "crossed-out digit changes the final count",
            narrator.prompts[-1],
        )
        self.assertEqual(
            sink.ambiguity_rows,
            ["Crossed-out cancellation digit kept reading as either 1 or 2, which changes the count."],
        )

    def test_after_action_queues_only_generated_partial_credit_rows(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 1/2 (correct setup, wrong final species). Prof: 1/2 (same).",
                "Correct stoichiometric setup and mole relationship.",
                "Final answer awards NH3 credit to reactant-mole addition.",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=2.0,
            professor_mark="1/2",
            student_answer="14.2031 moles",
            notes="partial",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=1.0,
            model_confidence=0.8,
            model_reasoning="Correct setup, wrong final target species.",
            model_read="14.2031 moles",
            score_basis="Correct setup, lost final credit for target-species drift.",
        )

        narrator._produce_after_action(20.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.basis_rows,
            ["Correct setup, lost final credit for target-species drift."],
        )
        self.assertEqual(sink.review_markers, [])
        self.assertEqual(sink.professor_mismatches, [])
        self.assertEqual(len(narrator._legibility_jobs), 2)
        self.assertEqual(
            [job["row_type"] for job in narrator._legibility_jobs],
            ["credit_preserved", "deduction"],
        )

    def test_after_action_emits_review_needed_and_professor_mismatch_immediately(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 4/4 (correct Hess's Law combination). Truth: 4/4 (corrected after review). · Historical prof: 2/4 (batch-mark partial).",
            ],
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
            correction_reason="Reviewed from page image: Hess's Law reversal and cancellation are coherent.",
        )
        prediction = Prediction(
            exam_id="34-blue",
            question_id="fr-8",
            model_score=4.0,
            model_confidence=0.45,
            model_reasoning="After one careful pass, human review is warranted because the cancellation handwriting is ambiguous.",
            model_read="-186.2 kJ",
            score_basis="Correct Hess's Law manipulation and final enthalpy.",
        )

        narrator._produce_after_action(50.0, prediction, item, template_question=None)

        self.assertEqual(
            sink.basis_rows,
            ["Correct Hess's Law manipulation and final enthalpy."],
        )
        self.assertEqual(
            sink.review_markers,
            ["Human review warranted because the cancellation handwriting remains ambiguity-sensitive after a bounded pass."],
        )
        self.assertEqual(
            sink.professor_mismatches,
            ["Historical professor awarded 2/4; corrected truth is 4/4."],
        )
        self.assertFalse(narrator._flush_idle_legibility_once())

    def test_idle_legibility_queue_starts_background_scheduler_when_rows_are_pending(self):
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 1/2 (setup right, finish wrong). Prof: 1/2 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=2.0,
            professor_mark="1/2",
            student_answer="14.2031 moles",
            notes="partial",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=1.0,
            model_confidence=0.8,
            model_reasoning="Correct setup, wrong final target species.",
            model_read="14.2031 moles",
            score_basis="Correct setup, lost final credit for target-species drift.",
        )

        with mock.patch("auto_grader.thinking_narrator.threading.Thread") as thread_mock:
            thread = thread_mock.return_value
            narrator._produce_after_action(20.0, prediction, item, template_question=None)

        thread_mock.assert_called_once()
        thread.start.assert_called_once()


    # ------------------------------------------------------------------
    # Selective bloom from narrator turbulence (Operation Paint Flakes)
    # ------------------------------------------------------------------

    def test_should_emit_basis_row_returns_true_when_turbulence_is_high(self):
        """_should_emit_basis_row must accept a turbulence_is_high flag and
        return True for a full-credit match when that flag is set, even with
        no ambiguity, review, or professor mismatch.  This is the core
        contract for the selective-history-bloom attractor: turbulence is a
        direct, first-class bloom trigger."""
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="2/2",
            student_answer="6.98 mL",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.95,
            model_reasoning="Correct.",
            model_read="6.98 mL",
            score_basis="Correct density calculation.",
        )
        result = ThinkingNarrator._should_emit_basis_row(
            prediction,
            item,
            ambiguity_needed=False,
            review_needed=None,
            professor_mismatch=None,
            turbulence_is_high=True,
        )
        self.assertTrue(
            result,
            "High turbulence on a full-credit match must trigger basis row bloom",
        )

    def test_should_emit_basis_row_returns_false_for_quiet_full_credit(self):
        """A full-credit match with turbulence_is_high=False and no other
        expansion signals must NOT emit a basis row.  Boring stays collapsed."""
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="2/2",
            student_answer="6.98 mL",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.95,
            model_reasoning="Correct.",
            model_read="6.98 mL",
            score_basis="Correct density calculation.",
        )
        result = ThinkingNarrator._should_emit_basis_row(
            prediction,
            item,
            ambiguity_needed=False,
            review_needed=None,
            professor_mismatch=None,
            turbulence_is_high=False,
        )
        self.assertFalse(
            result,
            "Quiet full-credit match must stay collapsed",
        )

    def test_handle_legibility_rows_blooms_turbulent_full_credit_match(self):
        """End-to-end: a full-credit match with high narrator turbulence
        counters gets a basis row emitted during _handle_legibility_rows.
        This tests the wiring from turbulence counters through the bloom
        decision into the sink."""
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 2/2 (correct density). Prof: 2/2 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="2/2",
            student_answer="6.98 mL",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.92,
            model_reasoning="Density = mass/volume = 6.98 mL. Correct.",
            model_read="6.98 mL",
            score_basis="Correct density calculation, mass/volume yields 6.98 mL.",
        )
        # High turbulence — many dedup drops and grooming passes
        narrator._item_turbulence_dedup_count = 12
        narrator._item_turbulence_status_dedup_count = 6
        narrator._item_turbulence_grooming_count = 3

        narrator._handle_legibility_rows(prediction, item)

        self.assertEqual(
            sink.basis_rows,
            ["Correct density calculation, mass/volume yields 6.98 mL."],
            "Turbulent full-credit match should bloom with a basis row",
        )

    def test_handle_legibility_rows_collapses_quiet_full_credit_match(self):
        """End-to-end: a full-credit match with zero turbulence stays
        collapsed — no basis row, no queued legibility jobs."""
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 2/2 (correct). Prof: 2/2 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="2/2",
            student_answer="6.98 mL",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.95,
            model_reasoning="Straightforward.",
            model_read="6.98 mL",
            score_basis="Correct density calculation.",
        )
        narrator._item_turbulence_dedup_count = 0
        narrator._item_turbulence_status_dedup_count = 0
        narrator._item_turbulence_grooming_count = 0

        narrator._handle_legibility_rows(prediction, item)

        self.assertEqual(sink.basis_rows, [])
        self.assertEqual(narrator._legibility_jobs, [])

    def test_handle_legibility_rows_does_not_bloom_moderate_turbulence(self):
        """Moderate turbulence (a couple of dedup drops, no grooming) on a
        full-credit match must NOT trigger bloom.  Selective bloom, not
        universal elaboration."""
        sink = _DummySink()
        narrator = _QueuedLegibilityNarrator(
            sink,
            responses=[
                "Grader: 2/2 (correct). Prof: 2/2 (same).",
            ],
        )
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="2/2",
            student_answer="6.98 mL",
            notes="full credit",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.95,
            model_reasoning="Correct.",
            model_read="6.98 mL",
            score_basis="Correct density calculation.",
        )
        # Below bloom threshold: 1 dedup, no grooming
        narrator._item_turbulence_dedup_count = 1
        narrator._item_turbulence_status_dedup_count = 0
        narrator._item_turbulence_grooming_count = 0

        narrator._handle_legibility_rows(prediction, item)

        self.assertEqual(sink.basis_rows, [])


if __name__ == "__main__":
    unittest.main()
