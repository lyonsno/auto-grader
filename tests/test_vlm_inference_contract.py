from __future__ import annotations

import unittest


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


if __name__ == "__main__":
    unittest.main()
