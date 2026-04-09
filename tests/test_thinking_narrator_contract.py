from __future__ import annotations

import json
from unittest import mock
import unittest

from auto_grader.thinking_narrator import ThinkingNarrator


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.rollbacks = 0
        self.commits: list[str] = []
        self.drops: list[tuple[str, str]] = []
        self.topics: list[tuple[str, str | None]] = []

    def write_delta(self, text: str) -> None:
        self.deltas.append(text)

    def rollback_live(self) -> None:
        self.rollbacks += 1

    def commit_live(self, *, mode: str = "thought") -> None:
        self.commits.append(mode)

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))

    def write_topic(self, text: str, verdict: str | None = None) -> None:
        self.topics.append((text, verdict))


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
        self.assertEqual(sink.rollbacks, 1)
        self.assertEqual(sink.commits, ["status"])
        self.assertEqual(sink.drops, [])
        self.assertEqual(
            narrator._prior_statuses[-1],
            "Rechecking the same unit conversion.",
        )
        self.assertEqual(narrator._thoughts_since_status, [])

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
        self.assertEqual(sink.drops, [("dedup", "I'm tracing the same unit conversion mistake.")])
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

    def test_streaming_bonsai_default_temperature_is_one(self):
        sink = _DummySink()
        narrator = ThinkingNarrator(sink)

        def _raise_and_capture(request, timeout):
            body = json.loads(request.data.decode())
            self.assertEqual(body["temperature"], 1.0)
            raise RuntimeError("boom")

        with mock.patch("urllib.request.urlopen", side_effect=_raise_and_capture):
            with self.assertRaises(RuntimeError):
                narrator._chat_completion_stream([{"role": "system", "content": "x"}], lambda _: None)

    def test_after_action_bonsai_call_uses_temperature_one(self):
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

        self.assertEqual(completion_mock.call_args.kwargs["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
