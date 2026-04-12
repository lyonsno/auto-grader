from __future__ import annotations

import tempfile
from unittest import mock
import unittest

from pathlib import Path

from auto_grader.eval_harness import EvalItem, Prediction
from auto_grader.vlm_inference import (
    ServerConfig,
    _consume_streaming_response,
    grade_all_items,
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

    def test_default_qwen_presence_penalty_returns_to_coding_preset_zero(self):
        config = ServerConfig(base_url="http://example.test")

        self.assertEqual(
            config.model,
            "qwen3p5-35B-A3B",
        )
        self.assertEqual(
            config.presence_penalty,
            0.0,
            "Qwen's default preset should return to the coding-preset zero presence penalty",
        )

    def test_length_truncation_returns_failure_prediction_instead_of_raising(self):
        # Contract updated by Operation Zilch Reaper (forward lane):
        # truncated rows carry null sentinels and truncated=True, not
        # a lying model_score=0.0. See
        # test_vlm_inference_truncation_contract.py for the primary
        # fail-first tests. This test preserves the degrade-instead-of-
        # crash invariant and the human-readable reasoning message.
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
        self.assertIsNone(pred.model_score)
        self.assertIsNone(pred.model_confidence)
        self.assertTrue(pred.truncated)
        self.assertIsNone(pred.is_obviously_fully_correct)
        self.assertIsNone(pred.is_obviously_wrong)
        self.assertEqual(pred.raw_reasoning, "very long reasoning")
        self.assertIn("truncated", pred.model_reasoning.lower())
        self.assertIn("max", pred.model_reasoning.lower())

    def test_unparseable_output_returns_failure_prediction_instead_of_raising(self):
        # Contract updated by Operation Zilch Reaper (forward lane):
        # unparseable rows are non-predictions, not confident zeros.
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

        self.assertIsNone(pred.model_score)
        self.assertIsNone(pred.model_confidence)
        self.assertTrue(pred.truncated)
        self.assertIsNone(pred.is_obviously_fully_correct)
        self.assertIsNone(pred.is_obviously_wrong)
        self.assertEqual(pred.raw_assistant, "definitely not json")
        self.assertEqual(pred.raw_reasoning, "reasoning trace")
        self.assertIn("parse", pred.model_reasoning.lower())

    def test_grade_single_item_uses_longer_first_timeout_then_shorter_retries(self):
        timeouts: list[float] = []

        def fake_urlopen(_req, timeout):
            timeouts.append(timeout)
            raise TimeoutError("server wedged")

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ), mock.patch(
            "time.sleep",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                TimeoutError,
                "VLM request failed after 3 attempts",
            ):
                grade_single_item(
                    self._item(),
                    page_image=b"png",
                    config=self._config(),
                )

        self.assertEqual(
            timeouts,
            [180, 60, 60],
            "smoke grading should allow a longer first request for cold model load, then fail faster on later retries",
        )

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

    def test_focus_preview_callback_failure_surfaces_drop_and_grading_continues(self):
        item = self._item()
        pred = Prediction(
            exam_id=item.exam_id,
            question_id=item.question_id,
            model_score=2.0,
            model_confidence=0.8,
            model_reasoning="setup right, arithmetic wrong",
            model_read=item.student_answer,
        )
        sink = mock.Mock()
        narrator = mock.Mock()
        focus_preview_callback = mock.Mock(
            side_effect=RuntimeError("preview blew up")
        )

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "auto_grader.vlm_inference.extract_page_image",
            return_value=b"png",
        ), mock.patch(
            "auto_grader.vlm_inference.grade_single_item",
            return_value=pred,
        ):
            scans_dir = Path(tmpdir)
            (scans_dir / "15 blue.pdf").write_bytes(b"%PDF-1.4")

            predictions = grade_all_items(
                [item],
                scans_dir=scans_dir,
                config=self._config(),
                narrator=narrator,
                sink=sink,
                focus_preview_callback=focus_preview_callback,
            )

        self.assertEqual(predictions, [pred])
        sink.write_header.assert_called_once()
        sink.write_drop.assert_called_once_with(
            "focus-preview-error",
            "15-blue/fr-11a · focus preview unavailable (RuntimeError)",
        )
        narrator.start.assert_called_once()
        narrator.stop_and_summarize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
