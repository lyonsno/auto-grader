from __future__ import annotations

import unittest

from auto_grader.thinking_narrator import ThinkingNarrator


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.rollbacks = 0
        self.commits = 0
        self.drops: list[tuple[str, str]] = []

    def write_delta(self, text: str) -> None:
        self.deltas.append(text)

    def rollback_live(self) -> None:
        self.rollbacks += 1

    def commit_live(self) -> None:
        self.commits += 1

    def write_drop(self, reason: str, text: str) -> None:
        self.drops.append((reason, text))


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
        narrator._prior_summaries = [
            "I'm tracing the unit conversion mistake."
        ]

        narrator._dispatch("same reasoning chunk", narrator._dispatch_generation)

        self.assertEqual(narrator.calls, ["thought", "status"])
        self.assertEqual(sink.rollbacks, 1)
        self.assertEqual(sink.commits, 1)
        self.assertEqual(sink.drops, [])
        self.assertEqual(
            narrator._prior_summaries[-1],
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

if __name__ == "__main__":
    unittest.main()
