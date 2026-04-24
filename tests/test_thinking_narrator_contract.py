from __future__ import annotations

import json
import unittest
from unittest import mock

from auto_grader.thinking_narrator import (
    ThinkingNarrator,
    _SYSTEM_PROMPT,
    _classify_score_against_band,
    _checkpoint_line_breaks_contract,
)


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.rollbacks = 0
        self.commits = 0
        self.drops: list[tuple[str, str]] = []
        self.topics: list[dict[str, object]] = []
        self.structured_rows: list[tuple[str, str]] = []

    def write_delta(self, text: str, *, mode: str = "thought") -> None:
        self.deltas.append(text)

    def rollback_live(self) -> None:
        self.rollbacks += 1

    def commit_live(self, *, mode: str = "thought") -> None:
        self.commits += 1

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))

    def write_topic(self, text: str, verdict: str | None = None, **kwargs) -> None:
        self.topics.append({"text": text, "verdict": verdict, **kwargs})

    def write_basis(self, text: str) -> None:
        return None

    def write_review_marker(self, text: str) -> None:
        return None

    def write_read(self, text: str, **_kwargs) -> None:
        self.structured_rows.append(("read", text))

    def write_salvage(self, text: str, **_kwargs) -> None:
        self.structured_rows.append(("salvage", text))

    def write_hinge(self, text: str, **_kwargs) -> None:
        self.structured_rows.append(("hinge", text))


class _RetryNarrator(ThinkingNarrator):
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


class _AfterActionNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink, *, response: str = "") -> None:
        super().__init__(sink)
        self._response = response
        self.chat_calls: list[dict[str, object]] = []

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return self._response

    def _handle_legibility_rows(self, prediction, item):  # type: ignore[override]
        return None

    def _schedule_idle_legibility_if_needed(self) -> None:  # type: ignore[override]
        return None


class _DossierAfterActionNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        system = messages[0]["content"]
        if "background dossier" in system:
            return json.dumps(
                {
                    "read": "Final mark could be a corrected 1 or a messy 2; context leans 1.",
                    "salvage": "The student's setup still supports the intended chemistry method.",
                    "hinge": "The score turns on whether the final handwritten glyph outweighs the surrounding coherent work.",
                }
            )
        return ""

    def _handle_legibility_rows(self, prediction, item):  # type: ignore[override]
        return None

    def _schedule_idle_legibility_if_needed(self) -> None:  # type: ignore[override]
        return None


class _PromptCapturingDossierNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)
        self.enqueued_dossier_prompts: list[tuple[str, str | None]] = []

    def _enqueue_dossier_job(
        self,
        prompt: str,
        *,
        target: str | None = None,
    ) -> None:  # type: ignore[override]
        self.enqueued_dossier_prompts.append((prompt, target))

    def _handle_legibility_rows(self, prediction, item):  # type: ignore[override]
        return None

    def _schedule_idle_legibility_if_needed(self) -> None:  # type: ignore[override]
        return None


class _FakeJSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


class _FakeStreamResponse:
    def __iter__(self):
        yield b"data: [DONE]\n\n"

    def close(self) -> None:
        return None


class ThinkingNarratorContract(unittest.TestCase):
    def test_system_prompt_does_not_claim_professor_score_as_live_context(self):
        self.assertNotIn(
            "professor's ground-truth score",
            _SYSTEM_PROMPT,
            "the live narrator should not be told it will receive professor-side score truth as context; that leaks exactly the kind of hindsight we were trying to keep out of live commentary",
        )

    def test_checkpoint_contract_accepts_context_label(self):
        self.assertFalse(
            _checkpoint_line_breaks_contract(
                "Context: Rubric denies method credit because the setup uses initial energy."
            )
        )

    def test_checkpoint_canonicalizer_promotes_context_label(self):
        self.assertEqual(
            ThinkingNarrator._canonicalize_checkpoint_text(
                "context: rubric denies method credit for the setup."
            ),
            "Context: rubric denies method credit for the setup.",
        )

    def test_checkpoint_prompt_treats_prior_context_as_update_target(self):
        narrator = ThinkingNarrator(_DummySink())

        content = narrator._build_checkpoint_user_content(
            "student used initial energy instead of delta E",
            "I'm checking the photon-energy setup.",
            prior_thoughts=[],
            prior_statuses=[],
            prior_checkpoints=[
                "Core issue: Student used initial energy instead of delta E.",
                "Context: Student used initial energy instead of delta E, yielding a negative frequency.",
            ],
        )

        self.assertIn(
            "Latest Context to update",
            content,
            "when a prior Context row exists, checkpoint prompting should treat it as a rolling summary target instead of inviting an append-only pileup",
        )
        self.assertIn(
            "If you write another Context:",
            content,
        )

    def test_duplicate_first_person_line_retries_as_status_and_commits(self):
        sink = _DummySink()
        narrator = _RetryNarrator(sink)
        narrator.start(item_header="15-blue/fr-1")
        narrator._thoughts_since_status = [
            "I'm tracing the unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(narrator.calls, ["thought", "status"])
        self.assertEqual(sink.rollbacks, 0)
        self.assertEqual(sink.commits, 1)
        self.assertEqual(sink.drops, [])
        self.assertEqual(
            narrator._prior_statuses[-1],
            "Rechecking the same unit conversion.",
        )

    def test_classify_score_against_band_keeps_truth_as_exact_hit_target(self):
        item = type(
            "Item",
            (),
            {
                "truth_score": 1.0,
                "professor_score": 1.0,
                "acceptable_score_floor": 1.0,
                "acceptable_score_ceiling": 1.5,
            },
        )()

        exact = _classify_score_against_band(1.0, item)
        ceiling = _classify_score_against_band(1.5, item)
        overshoot = _classify_score_against_band(2.0, item)
        undershoot = _classify_score_against_band(0.0, item)

        self.assertEqual(
            exact.verdict_short,
            "match",
            "truth_score should remain the exact-hit target even when an acceptable band exists",
        )
        self.assertEqual(
            ceiling.verdict_short,
            "within_band",
            "acceptable-band ceiling hits that exceed truth_score are lawful-range calls, not exact matches",
        )
        self.assertEqual(overshoot.verdict_short, "overshoot")
        self.assertEqual(undershoot.verdict_short, "undershoot")
        self.assertTrue(exact.band_present)
        self.assertTrue(ceiling.band_present)

    def test_after_action_keeps_exact_truth_match_even_when_band_excludes_truth(self):
        sink = _DummySink()
        narrator = _AfterActionNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-12b",
                "answer_type": "lewis_structure",
                "max_points": 2.0,
                "student_answer": "two resonance structures",
                "professor_score": 1.0,
                "truth_score": 1.0,
                "professor_mark": "partial",
                "notes": "half annotation",
                "acceptable_score_floor": 1.5,
                "acceptable_score_ceiling": 2.0,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 1.0,
                "model_read": "two resonance structures",
                "model_reasoning": "matches the current truth baseline",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(12.0, prediction, item, template_question=None)

        self.assertEqual(len(sink.topics), 1)
        self.assertEqual(
            narrator.chat_calls,
            [],
            "the compact outcome block is intentional here; after-action should not re-enter the Bonsai one-line/coda path",
        )
        self.assertEqual(
            sink.topics[0]["verdict"],
            "match",
            "the narrator's primary verdict must stay aligned with truth_score even when an advisory acceptable band excludes that truth",
        )
        self.assertEqual(
            sink.topics[0]["text"],
            "12s · Grader: 1/2 · Prof: 1/2\nBand: 1.5/2 to 2/2",
        )
        self.assertEqual(sink.topics[0]["acceptable_score_floor"], 1.5)
        self.assertEqual(sink.topics[0]["acceptable_score_ceiling"], 2.0)

    def test_after_action_uses_compact_truth_and_historical_prof_block_for_corrected_items(
        self,
    ):
        sink = _DummySink()
        narrator = _AfterActionNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-5b",
                "answer_type": "numeric",
                "max_points": 2.0,
                "student_answer": "14.2031",
                "professor_score": 2.0,
                "truth_score": 0.0,
                "corrected_score": 0.0,
                "professor_mark": "x",
                "notes": "historical overcredit",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 0.0,
                "model_read": "14.2031",
                "model_reasoning": "methodology is invalid",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(72.0, prediction, item, template_question=None)

        self.assertEqual(len(sink.topics), 1)
        self.assertEqual(narrator.chat_calls, [])
        self.assertEqual(
            sink.topics[0]["text"],
            "72s · Grader: 0/2 · Truth: 0/2\nHistorical prof: 2/2",
        )

    def test_after_action_enqueues_and_flushes_background_dossier_for_interesting_item(self):
        sink = _DummySink()
        narrator = _DossierAfterActionNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-11c",
                "answer_type": "electron_config",
                "max_points": 2.0,
                "student_answer": "1",
                "professor_score": 2.0,
                "truth_score": 2.0,
                "professor_mark": "partial",
                "notes": "glyph ambiguity",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 1.0,
                "model_read": "1",
                "model_reasoning": (
                    "The final glyph remains ambiguous between 1 and 2, "
                    "but the surrounding chemistry work supports 1 and still deserves partial credit."
                ),
                "score_basis": "Preserved partial credit for the coherent orbital-box work despite the ambiguous final digit.",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(95.0, prediction, item, template_question=None)

        self.assertEqual(len(sink.topics), 1)
        self.assertEqual(len(narrator._legibility_jobs), 1)
        self.assertTrue(narrator._flush_idle_legibility_once())
        self.assertEqual(
            sink.structured_rows,
            [
                ("read", "Final mark could be a corrected 1 or a messy 2; context leans 1."),
                ("salvage", "The student's setup still supports the intended chemistry method."),
                ("hinge", "The score turns on whether the final handwritten glyph outweighs the surrounding coherent work."),
            ],
        )

    def test_start_flushes_pending_background_dossier_before_resetting_item_state(self):
        sink = _DummySink()
        narrator = _DossierAfterActionNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-11a",
                "answer_type": "electron_config",
                "max_points": 2.0,
                "student_answer": "1",
                "professor_score": 2.0,
                "truth_score": 2.0,
                "professor_mark": "partial",
                "notes": "ambiguous final digit with partially creditable setup",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 1.0,
                "model_read": "1",
                "model_reasoning": (
                    "The final glyph remains ambiguous between 1 and 2, "
                    "but the surrounding chemistry work supports 1."
                ),
                "score_basis": "Accepted the coherent orbital-box work despite the ambiguous final digit.",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(95.0, prediction, item, template_question=None)
        narrator.start(item_header="15-blue/fr-12a")

        self.assertEqual(
            sink.structured_rows,
            [
                ("read", "Final mark could be a corrected 1 or a messy 2; context leans 1."),
                ("salvage", "The student's setup still supports the intended chemistry method."),
                ("hinge", "The score turns on whether the final handwritten glyph outweighs the surrounding coherent work."),
            ],
            "starting the next item should not discard the previous item's queued dossier rows",
        )
        self.assertEqual(narrator._legibility_jobs, [])

    def test_after_action_skips_background_dossier_for_quick_stable_item(self):
        sink = _DummySink()
        narrator = _DossierAfterActionNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-10a",
                "answer_type": "numeric",
                "max_points": 1.0,
                "student_answer": "8.225e14",
                "professor_score": 1.0,
                "truth_score": 1.0,
                "professor_mark": "correct",
                "notes": "clean match",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 1.0,
                "model_read": "8.225e14",
                "model_reasoning": "The student's frequency calculation matches the expected value.",
                "score_basis": "Correct setup, arithmetic, and final answer.",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(24.0, prediction, item, template_question=None)

        self.assertEqual(len(sink.topics), 1)
        self.assertEqual(narrator._legibility_jobs, [])
        self.assertFalse(narrator._flush_idle_legibility_once())
        self.assertEqual(sink.structured_rows, [])

    def test_truncated_long_item_still_enqueues_background_dossier(self):
        sink = _DummySink()
        narrator = _PromptCapturingDossierNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-11c",
                "answer_type": "exact_match",
                "max_points": 0.5,
                "student_answer": "1",
                "professor_score": 0.5,
                "truth_score": 0.5,
                "professor_mark": "correct",
                "notes": "ambiguous crossed-out numeral",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 0.0,
                "model_read": "2?",
                "model_reasoning": (
                    "Ambiguous crossed-out 2 obscures whether the student intended 2 or 1."
                ),
                "score_basis": "No final score committed before truncation.",
                "truncated": True,
            },
        )()

        narrator._produce_after_action(194.0, prediction, item, template_question=None)

        self.assertEqual(len(sink.topics), 1)
        self.assertEqual(
            sink.topics[0]["text"],
            "194s · grader did not commit to a score (truncated)",
        )
        self.assertEqual(
            len(narrator.enqueued_dossier_prompts),
            1,
            "long ambiguity-shaped truncated items should still earn a trailing dossier instead of being categorically excluded",
        )

    def test_interpretive_disagreement_item_can_enqueue_dossier_before_general_threshold(self):
        sink = _DummySink()
        narrator = _PromptCapturingDossierNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "34-blue",
                "question_id": "fr-12a",
                "answer_type": "lewis_structure",
                "max_points": 2.0,
                "student_answer": "linear ozone",
                "professor_score": 1.5,
                "truth_score": 1.5,
                "professor_mark": "partial",
                "notes": "wrong Lewis structure",
                "acceptable_score_floor": 1.0,
                "acceptable_score_ceiling": 1.5,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 0.0,
                "model_read": "linear O=O=O",
                "model_reasoning": "The student drew linear ozone with 14 electrons and missed the bent resonance structure.",
                "score_basis": "No valid Lewis reasoning survives from the linear ozone drawing.",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(44.0, prediction, item, template_question=None)

        self.assertEqual(
            len(narrator.enqueued_dossier_prompts),
            1,
            "long-ish interpretive disagreement items should start earning dossiers before the generic elapsed threshold so the operator can study the hard cases more aggressively",
        )

    def test_interpretive_truncated_item_can_enqueue_dossier_before_general_threshold(self):
        sink = _DummySink()
        narrator = _PromptCapturingDossierNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-11c",
                "answer_type": "exact_match",
                "max_points": 0.5,
                "student_answer": "1",
                "professor_score": 0.5,
                "truth_score": 0.5,
                "professor_mark": "correct",
                "notes": "crossed-out digit ambiguity",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 0.0,
                "model_read": "2?",
                "model_reasoning": "Ambiguous crossed-out digit conflicts with the orbital diagram's single unpaired electron.",
                "score_basis": "No final score committed before truncation.",
                "truncated": True,
            },
        )()

        narrator._produce_after_action(41.0, prediction, item, template_question=None)

        self.assertEqual(
            len(narrator.enqueued_dossier_prompts),
            1,
            "interpretive truncated items should not have to wait for the generic long-item threshold before earning a trailing dossier",
        )

    def test_truncated_dossier_prompt_marks_unstabilized_score(self):
        sink = _DummySink()
        narrator = _PromptCapturingDossierNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "15-blue",
                "question_id": "fr-12b",
                "answer_type": "lewis_structure",
                "max_points": 2.0,
                "student_answer": "two resonance structures",
                "professor_score": 1.0,
                "truth_score": 1.0,
                "professor_mark": "partial",
                "notes": "lone-pair ambiguity",
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 0.0,
                "model_read": "two resonance structures",
                "model_reasoning": "The resonance connectivity is right but the lone pairs remain ambiguous.",
                "score_basis": "The grader never stabilized to a final score before truncation.",
                "truncated": True,
            },
        )()

        prompt = narrator._build_dossier_prompt(
            elapsed=226.0,
            prediction=prediction,
            item=item,
        )

        self.assertIn(
            "Final score state: no final score committed before truncation",
            prompt,
            "truncated dossier prompts should tell the model plainly that the grader never stabilized to a final score",
        )
        self.assertIn(
            "If the grader never stabilized to a final score, say that explicitly when it matters.",
            prompt,
        )

    def test_full_credit_exact_hit_skips_background_dossier_even_if_reasoning_is_ambiguous(self):
        sink = _DummySink()
        narrator = _PromptCapturingDossierNarrator(sink)
        item = type(
            "Item",
            (),
            {
                "exam_id": "27-blue-2023",
                "question_id": "fr-12a",
                "answer_type": "lewis_structure",
                "max_points": 2.0,
                "student_answer": "two resonance structures",
                "professor_score": 2.0,
                "truth_score": 2.0,
                "professor_mark": "correct",
                "notes": "clean full credit",
                "acceptable_score_floor": None,
                "acceptable_score_ceiling": None,
            },
        )()
        prediction = type(
            "Prediction",
            (),
            {
                "model_score": 2.0,
                "model_read": "two resonance structures",
                "model_reasoning": "The omitted lone pairs are a little ambiguous, but the final two drawings are chemically complete enough for full credit.",
                "score_basis": "Full credit awarded for correct resonance structures and complete electron accounting.",
                "truncated": False,
            },
        )()

        narrator._produce_after_action(71.0, prediction, item, template_question=None)

        self.assertEqual(
            narrator.enqueued_dossier_prompts,
            [],
            "exact full-credit hits should not emit trailing dossier rows just because the reasoning mentions ambiguity in passing",
        )

    def test_qwen36_sync_narrator_requests_disable_thinking(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink, model="Qwen3.6-35B-A3B-bf16")
        captured_bodies: list[dict[str, object]] = []

        def _capture_request(url, data=None, headers=None, method=None):
            assert data is not None
            captured_bodies.append(json.loads(data.decode()))
            return object()

        with (
            mock.patch("urllib.request.Request", side_effect=_capture_request),
            mock.patch(
                "urllib.request.urlopen",
                return_value=_FakeJSONResponse(
                    {"choices": [{"message": {"content": "ok"}}]}
                ),
            ),
        ):
            text = narrator._chat_completion(
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(text, "ok")
        self.assertEqual(
            captured_bodies[-1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_non_qwen_sync_narrator_requests_do_not_force_chat_template_kwargs(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink, model="Bonsai-8B-mlx-1bit")
        captured_bodies: list[dict[str, object]] = []

        def _capture_request(url, data=None, headers=None, method=None):
            assert data is not None
            captured_bodies.append(json.loads(data.decode()))
            return object()

        with (
            mock.patch("urllib.request.Request", side_effect=_capture_request),
            mock.patch(
                "urllib.request.urlopen",
                return_value=_FakeJSONResponse(
                    {"choices": [{"message": {"content": "ok"}}]}
                ),
            ),
        ):
            narrator._chat_completion(
                [{"role": "user", "content": "hello"}],
            )

        self.assertNotIn("chat_template_kwargs", captured_bodies[-1])

    def test_qwen36_streaming_narrator_requests_disable_thinking(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink, model="Qwen3.6-35B-A3B-bf16")
        captured_bodies: list[dict[str, object]] = []

        def _capture_request(url, data=None, headers=None, method=None):
            assert data is not None
            captured_bodies.append(json.loads(data.decode()))
            return object()

        with (
            mock.patch("urllib.request.Request", side_effect=_capture_request),
            mock.patch(
                "urllib.request.urlopen",
                return_value=_FakeStreamResponse(),
            ),
        ):
            text = narrator._chat_completion_stream(
                [{"role": "user", "content": "hello"}],
                on_delta=lambda _delta: None,
            )

        self.assertEqual(text, "")
        self.assertEqual(
            captured_bodies[-1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )


if __name__ == "__main__":
    unittest.main()
