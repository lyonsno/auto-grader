from __future__ import annotations

import unittest

from auto_grader.thinking_narrator import (
    ThinkingNarrator,
    _classify_score_against_band,
)


class _DummySink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.rollbacks = 0
        self.commits = 0
        self.drops: list[tuple[str, str]] = []
        self.topics: list[dict[str, object]] = []

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
    def __init__(self, sink: _DummySink) -> None:
        super().__init__(sink)

    def _chat_completion(self, messages, **kwargs):  # type: ignore[override]
        return "Grader: 1 (exact truth). Prof: 1 (same score)."

    def _handle_legibility_rows(self, prediction, item):  # type: ignore[override]
        return None

    def _schedule_idle_legibility_if_needed(self) -> None:  # type: ignore[override]
        return None


class ThinkingNarratorContract(unittest.TestCase):
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
            sink.topics[0]["verdict"],
            "match",
            "the narrator's primary verdict must stay aligned with truth_score even when an advisory acceptable band excludes that truth",
        )
        self.assertEqual(sink.topics[0]["acceptable_score_floor"], 1.5)
        self.assertEqual(sink.topics[0]["acceptable_score_ceiling"], 2.0)


if __name__ == "__main__":
    unittest.main()
