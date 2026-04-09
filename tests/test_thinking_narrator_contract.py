from __future__ import annotations

import unittest

from auto_grader.thinking_narrator import (
    _MAX_INTERVAL_S,
    _MIN_INTERVAL_S,
    ThinkingNarrator,
)


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.rollbacks = 0
        self.commit_modes: list[str] = []
        self.drops: list[tuple[str, str]] = []

    def write_delta(self, text: str) -> None:
        self.deltas.append(text)

    def rollback_live(self) -> None:
        self.rollbacks += 1

    def commit_live(self, *, mode: str = "thought") -> None:
        self.commit_modes.append(mode)

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))


class _QueuedNarrator(ThinkingNarrator):
    def __init__(self, sink: _DummySink, responses: list[str]) -> None:
        super().__init__(sink)
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    def _chat_completion_stream(self, messages, on_delta, **kwargs):  # type: ignore[override]
        system = messages[0]["content"]
        mode = "status" if "present-participle status line" in system else "thought"
        user_content = messages[-1]["content"]
        self.calls.append({"mode": mode, "user_content": user_content})
        text = self._responses.pop(0)
        for token in text.split():
            on_delta(token + " ")
        return text


class ThinkingNarratorContract(unittest.TestCase):
    def test_duplicate_first_person_retries_as_status_with_segmented_memory(self):
        sink = _DummySink()
        narrator = _QueuedNarrator(
            sink,
            [
                "I'm tracing the same unit conversion mistake.",
                "Rechecking the same unit conversion.",
            ],
        )
        narrator.start(item_header="15-blue/fr-1")
        narrator._prior_statuses = [
            "Checking the setup.",
            "Tracing the units.",
            "Comparing the denominator.",
            "Revisiting the arithmetic.",
            "Weighing the sig figs.",
            "Following the same unit conversion.",
        ]
        narrator._thoughts_since_status = [
            "I'm tracing the same unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual([call["mode"] for call in narrator.calls], ["thought", "status"])
        self.assertEqual(sink.rollbacks, 1)
        self.assertEqual(sink.commit_modes, ["status"])
        self.assertEqual(
            narrator._prior_statuses[-1],
            "Rechecking the same unit conversion.",
        )
        self.assertEqual(narrator._thoughts_since_status, [])

        status_user_content = narrator.calls[1]["user_content"]
        self.assertNotIn("Checking the setup.", status_user_content)
        self.assertIn("Tracing the units.", status_user_content)
        self.assertIn("Following the same unit conversion.", status_user_content)

    def test_thought_prompt_uses_current_status_and_last_four_thoughts(self):
        sink = _DummySink()
        narrator = _QueuedNarrator(
            sink,
            ["I'm checking the sig figs on the final value."],
        )
        narrator.start(item_header="15-blue/fr-3")
        narrator._prior_statuses = [
            "Checking the setup.",
            "Tracing the denominator swap.",
        ]
        narrator._thoughts_since_status = [
            "I'm reading the units.",
            "I'm weighing the denominator.",
            "I'm checking the division.",
            "I'm comparing the setup.",
            "I'm tracing the denominator slip.",
        ]

        narrator._dispatch("fresh reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commit_modes, ["thought"])
        thought_user_content = narrator.calls[0]["user_content"]
        self.assertIn("Tracing the denominator swap.", thought_user_content)
        self.assertNotIn("Checking the setup.", thought_user_content)
        self.assertNotIn("I'm reading the units.", thought_user_content)
        self.assertIn("I'm weighing the denominator.", thought_user_content)
        self.assertIn("I'm tracing the denominator slip.", thought_user_content)
        self.assertEqual(
            narrator._thoughts_since_status[-1],
            "I'm checking the sig figs on the final value.",
        )

    def test_double_dedup_backs_off_until_reset(self):
        sink = _DummySink()
        narrator = _QueuedNarrator(
            sink,
            [
                "I'm tracing the same unit conversion mistake.",
                "Rechecking the same unit conversion.",
            ],
        )
        narrator.start(item_header="15-blue/fr-5b")
        narrator._prior_statuses = [
            "Rechecking the same unit conversion."
        ]
        narrator._thoughts_since_status = [
            "I'm tracing the same unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commit_modes, [])
        self.assertEqual(sink.rollbacks, 2)
        self.assertEqual(sink.drops, [("dedup", "I'm tracing the same unit conversion mistake.")])
        self.assertEqual(narrator._dispatch_backoff_factor, 2.0)
        self.assertEqual(narrator._current_min_interval_s(), _MIN_INTERVAL_S * 2.0)
        self.assertEqual(narrator._current_max_interval_s(), _MAX_INTERVAL_S * 2.0)

        narrator.start(item_header="15-blue/fr-6")

        self.assertEqual(narrator._dispatch_backoff_factor, 1.0)
        self.assertEqual(narrator._current_min_interval_s(), _MIN_INTERVAL_S)
        self.assertEqual(narrator._current_max_interval_s(), _MAX_INTERVAL_S)

    def test_successful_commit_resets_backoff_factor(self):
        sink = _DummySink()
        narrator = _QueuedNarrator(
            sink,
            ["I'm noticing the denominator swap."],
        )
        narrator.start(item_header="15-blue/fr-10a")
        narrator._dispatch_backoff_factor = 4.0
        narrator._prior_statuses = ["Tracing the denominator swap."]
        narrator._thoughts_since_status = []

        narrator._dispatch("fresh reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(sink.commit_modes, ["thought"])
        self.assertEqual(narrator._dispatch_backoff_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
