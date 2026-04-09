from __future__ import annotations

from unittest import mock
import unittest

from auto_grader.eval_harness import EvalItem
from auto_grader.vlm_inference import ServerConfig, grade_single_item


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
        self.assertEqual(pred.raw_assistant, "definitely not json")
        self.assertEqual(pred.raw_reasoning, "reasoning trace")
        self.assertIn("parse", pred.model_reasoning.lower())


if __name__ == "__main__":
    unittest.main()
