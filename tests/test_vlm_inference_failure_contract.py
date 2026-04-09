from __future__ import annotations

from unittest import mock
import unittest

from auto_grader.eval_harness import EvalItem
from auto_grader.vlm_inference import (
    ServerConfig,
    _consume_streaming_response,
    grade_single_item,
)


class _DummyResponse:
    def close(self) -> None:
        return None


class VlmInferenceFailureContract(unittest.TestCase):
    def _item(self) -> EvalItem:
        return EvalItem(
            exam_id="15-blue",
            question_id="fr-11a",
            answer_type="electron_config",
            page=1,
            professor_score=3.0,
            max_points=3.0,
            professor_mark="check",
            student_answer="[Ne] ...",
            notes="",
        )

    def _config(self) -> ServerConfig:
        return ServerConfig(base_url="http://example.test", model="gemma-test")

    def test_default_max_tokens_is_lower_than_old_16384_ceiling(self):
        config = self._config()
        self.assertLess(
            config.max_tokens,
            16384,
            "the default token ceiling should be lower than the old runaway-friendly 16384 cap",
        )

    def test_length_truncation_returns_failure_prediction_instead_of_raising(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=("", "very long reasoning", "length"),
        ):
            pred = grade_single_item(
                self._item(),
                page_image=b"png",
                config=self._config(),
            )

        self.assertEqual(pred.exam_id, "15-blue")
        self.assertEqual(pred.question_id, "fr-11a")
        self.assertEqual(pred.model_score, 0.0)
        self.assertEqual(pred.model_confidence, 0.0)
        self.assertIsNone(pred.is_obviously_fully_correct)
        self.assertIsNone(pred.is_obviously_wrong)
        self.assertEqual(pred.raw_reasoning, "very long reasoning")
        self.assertIn("truncated", pred.model_reasoning.lower())
        self.assertIn("max", pred.model_reasoning.lower())

    def test_unparseable_output_returns_failure_prediction_instead_of_raising(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_DummyResponse(),
        ), mock.patch(
            "auto_grader.vlm_inference._consume_streaming_response",
            return_value=("definitely not json", "reasoning trace", "stop"),
        ):
            pred = grade_single_item(
                self._item(),
                page_image=b"png",
                config=self._config(),
            )

        self.assertEqual(pred.model_score, 0.0)
        self.assertEqual(pred.model_confidence, 0.0)
        self.assertIsNone(pred.is_obviously_fully_correct)
        self.assertIsNone(pred.is_obviously_wrong)
        self.assertEqual(pred.raw_assistant, "definitely not json")
        self.assertEqual(pred.raw_reasoning, "reasoning trace")
        self.assertIn("parse", pred.model_reasoning.lower())

    def test_stream_consumer_falls_back_to_model_reasoning_content_when_reasoning_channel_missing(self):
        seen: list[str] = []
        resp = [
            b'data: {"choices":[{"delta":{"content":"```json\\n{\\n  \\"model_read\\": \\"d = m/v\\",\\n  \\"model_reasoning\\": \\"The student "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"set the ratio correctly\\\\nthen botched the arithmetic.\\",\\n  \\"model_score\\": 1.0\\n}\\n```"}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n',
            b'data: [DONE]\n',
        ]

        content, reasoning, finish_reason = _consume_streaming_response(
            resp, seen.append
        )

        self.assertIn('"model_reasoning"', content)
        self.assertEqual(
            reasoning,
            "The student set the ratio correctly then botched the arithmetic.",
        )
        self.assertEqual("".join(seen), reasoning)
        self.assertEqual(finish_reason, "stop")

    def test_stream_consumer_prefers_reasoning_channel_when_present(self):
        seen: list[str] = []
        resp = [
            b'data: {"choices":[{"delta":{"reasoning_content":"Tracing the unit conversion. ","content":"{\\"model_reasoning\\": \\"ignored fallback\\""}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n',
            b'data: [DONE]\n',
        ]

        _content, reasoning, _finish_reason = _consume_streaming_response(
            resp, seen.append
        )

        self.assertEqual(reasoning, "Tracing the unit conversion. ")
        self.assertEqual("".join(seen), "Tracing the unit conversion. ")

    def test_stream_consumer_ignores_whitespace_only_reasoning_channel_and_uses_fallback(self):
        seen: list[str] = []
        resp = [
            b'data: {"choices":[{"delta":{"reasoning_content":"\\n","content":"```json\\n{\\"model_reasoning\\": \\"The student set up the ratio correctly. "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"Then they slipped on the arithmetic.\\"}\\n```"}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n',
            b'data: [DONE]\n',
        ]

        _content, reasoning, _finish_reason = _consume_streaming_response(
            resp, seen.append
        )

        self.assertEqual(
            reasoning,
            "The student set up the ratio correctly. Then they slipped on the arithmetic.",
        )
        self.assertEqual("".join(seen), reasoning)


if __name__ == "__main__":
    unittest.main()
