"""Contract tests for generating a reviewable short-answer sibling variant."""

from __future__ import annotations

from pathlib import Path
import unittest


_ASSET_ROOT = Path("/Users/noahlyons/dev/auto-grader-assets/exams")
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class ShortAnswerVariantGenerationContractTests(unittest.TestCase):
    def _family(self):
        from auto_grader.short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        return reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])

    def _generated(self):
        from auto_grader.short_answer_reconstruction import (
            build_generated_short_answer_variant,
        )

        return build_generated_short_answer_variant(self._family(), variant_id="C")

    def test_variant_builder_exists(self) -> None:
        from auto_grader.short_answer_reconstruction import (
            build_generated_short_answer_variant,
        )

        self.assertTrue(callable(build_generated_short_answer_variant))

    def test_generated_carries_new_variant_id_and_deterministic_variable_set(self) -> None:
        generated = self._generated()

        self.assertEqual(generated["variant_id"], "C")
        self.assertEqual(
            generated["variables"],
            {
                "acid_molarity": 0.00974,
                "acid_species": "hydrochloric acid",
                "base_molarity": 0.0339,
                "bronsted_acid": "acetic acid CH3COOH",
                "bronsted_base": "methylamine CH3NH2",
                "kc_q5": 0.0039,
                "kc_q6": 0.0714,
                "target_ph": 2.25,
            },
        )

    def test_generated_variant_is_reviewable_without_rereading_raw_pdfs(self) -> None:
        generated = self._generated()

        prompts = {entry["id"]: entry["prompt"] for entry in generated["question_prompts"]}
        response_boxes = {entry["id"]: entry["response_box_label"] for entry in generated["question_prompts"]}
        answers = generated["answer_preview"]

        self.assertEqual(prompts["q1a"], "Write a net ionic equation to show how methylamine CH3NH2 behaves as a Bronsted base in water.")
        self.assertEqual(prompts["q1b"], "Write a net ionic equation to show how acetic acid CH3COOH behaves as a Bronsted acid in water.")
        self.assertIn("0.00974 M hydrochloric acid", prompts["q2"])
        self.assertIn("0.0339 M aqueous solution of sodium hydroxide", prompts["q3"])
        self.assertIn("pH of 2.25", prompts["q4"])
        self.assertEqual(response_boxes["q6-ccl4"], "6.[CCl4]=")
        self.assertAlmostEqual(answers["q2"], 2.0114, places=4)
        self.assertAlmostEqual(answers["q3"], 12.5302, places=4)
        self.assertAlmostEqual(answers["q4"], 0.005623413, places=9)
