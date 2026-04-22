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


if __name__ == "__main__":
    unittest.main()
