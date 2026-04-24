from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_grader.eval_harness import EvalItem, Prediction


class SamplingPresetContract(unittest.TestCase):
    def test_gemma4_variant_uses_family_defaults(self):
        from auto_grader.vlm_inference import ServerConfig, apply_model_sampling_preset

        config = ServerConfig(
            base_url="http://example.test",
            model="google/gemma-4-31b-it",
        )

        resolved = apply_model_sampling_preset(config)

        self.assertEqual(
            resolved.temperature,
            1.0,
            "gemma-4 variants should inherit the gemma-4 family temperature instead of silently keeping Qwen defaults",
        )
        self.assertEqual(
            resolved.top_p,
            0.95,
            "gemma-4 variants should inherit the gemma-4 family top_p",
        )
        self.assertEqual(
            resolved.top_k,
            64,
            "gemma-4 variants should inherit the gemma-4 family top_k instead of Qwen's 20",
        )

    def test_unregistered_model_requires_explicit_family(self):
        from auto_grader.vlm_inference import ServerConfig, apply_model_sampling_preset

        config = ServerConfig(
            base_url="http://example.test",
            model="Step3-VL-10B",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unregistered model.*Step3-VL-10B.*model-family",
        ):
            apply_model_sampling_preset(config)

    def test_grade_all_items_rejects_contaminated_legacy_fifteen_blue_scan(self):
        from auto_grader.vlm_inference import ServerConfig, grade_all_items

        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=1.0,
            professor_mark="check",
            student_answer="6.98 cm^3",
            notes="mock",
        )
        config = ServerConfig(
            base_url="http://example.test",
            model="qwen3p5-35B-A3B",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=1.0,
            model_confidence=0.9,
            model_reasoning="mock",
            model_read="6.98 cm^3",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scans_dir = Path(tmpdir)
            (scans_dir / "15 blue.pdf").write_bytes(b"%PDF-1.4 legacy")

            with (
                mock.patch(
                    "auto_grader.vlm_inference.extract_page_image",
                    return_value=b"page-bytes",
                ),
                mock.patch(
                    "auto_grader.vlm_inference.grade_single_item",
                    return_value=prediction,
                ) as grade_single_item_mock,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Refusing contaminated fallback.*15 blue\\.pdf",
                ):
                    grade_all_items(
                        [item],
                        scans_dir,
                        config,
                    )

        self.assertEqual(grade_single_item_mock.call_count, 0)

    def test_grade_all_items_keeps_professor_score_out_of_live_narrator_context(self):
        from auto_grader.vlm_inference import ServerConfig, grade_all_items

        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="check",
            student_answer="14.2031 mol",
            notes="historical overcredit",
            corrected_score=0.0,
            correction_reason="boxed answer is chemically invalid",
        )
        config = ServerConfig(
            base_url="http://example.test",
            model="qwen3p5-35B-A3B",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=0.0,
            model_confidence=0.9,
            model_reasoning="method invalid",
            model_read="14.2031 mol",
        )

        class _FakeNarrator:
            def __init__(self):
                self.item_headers: list[str] = []

            def start(self, item_header: str | None = None) -> None:
                self.item_headers.append(item_header or "")

            def feed(self, _token: str) -> None:
                return None

            def stop_and_summarize(self, **_kwargs) -> None:
                return None

        class _FakeSink:
            def write_header(self, _text: str) -> None:
                return None

        narrator = _FakeNarrator()
        sink = _FakeSink()

        with tempfile.TemporaryDirectory() as tmpdir:
            scans_dir = Path(tmpdir)
            (scans_dir / "15 blue_professor_markings_hidden.pdf").write_bytes(b"%PDF-1.4 clean")
            template_path = Path(tmpdir) / "template.yaml"
            template_path.write_text(
                "fr-5b:\n"
                "  prompt: How many moles of NH3 can be produced?\n"
                "  correct:\n"
                "    value: 18.6 mol\n",
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "auto_grader.vlm_inference.extract_page_image",
                    return_value=b"page-bytes",
                ),
                mock.patch(
                    "auto_grader.vlm_inference.grade_single_item",
                    return_value=prediction,
                ),
            ):
                grade_all_items(
                    [item],
                    scans_dir,
                    config,
                    template_path=template_path,
                    narrator=narrator,
                    sink=sink,
                )

        self.assertEqual(len(narrator.item_headers), 1)
        header = narrator.item_headers[0]
        self.assertIn("[1/1] exam 15-blue · problem fr-5b", header)
        self.assertIn('Student wrote: "14.2031 mol"', header)
        self.assertNotIn("Professor scored:", header)
        self.assertNotIn("mark: check", header)

    def test_grade_single_item_uses_bounded_stream_idle_timeout(self):
        from auto_grader.vlm_inference import (
            ServerConfig,
            _STREAM_IDLE_TIMEOUT_S,
            grade_single_item,
        )

        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-10b",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=1.0,
            professor_mark="check",
            student_answer="-2.415",
            notes="mock",
        )
        config = ServerConfig(
            base_url="http://example.test",
            model="qwen3p5-35B-A3B",
        )

        with mock.patch(
            "auto_grader.vlm_inference._stream_vision_completion_with_finish",
            return_value=('{"model_score": 0, "model_confidence": 0.5, "model_read": "", "model_reasoning": "", "upstream_dependency": "none", "if_dependent_then_consistent": null, "score_basis": "", "is_obviously_fully_correct": null, "is_obviously_wrong": null}', "", "stop"),
        ) as stream_mock:
            grade_single_item(
                item,
                page_image=b"page-bytes",
                config=config,
            )

        self.assertEqual(
            stream_mock.call_args.kwargs["timeout"],
            _STREAM_IDLE_TIMEOUT_S,
            "grade_single_item should use the bounded stream idle timeout so a wedged upstream completion fails fast instead of freezing Paint Dry for ten minutes",
        )

    def test_describe_only_stream_uses_bounded_idle_timeout(self):
        from auto_grader.vlm_inference import (
            ServerConfig,
            _STREAM_IDLE_TIMEOUT_S,
            stream_vision_completion,
        )

        config = ServerConfig(
            base_url="http://example.test",
            model="qwen3p5-35B-A3B",
        )

        with mock.patch(
            "auto_grader.vlm_inference._stream_vision_completion_with_finish",
            return_value=("visible student work", "brief reasoning", "stop"),
        ) as stream_mock:
            stream_vision_completion(
                config=config,
                prompt_text="Describe the page.",
                page_image=b"page-bytes",
            )

        self.assertEqual(
            stream_mock.call_args.kwargs["timeout"],
            _STREAM_IDLE_TIMEOUT_S,
            "describe/smoke paths should share the same bounded idle timeout so the preview surface cannot hang indefinitely on a silent upstream stream",
        )
if __name__ == "__main__":
    unittest.main()
