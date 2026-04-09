from __future__ import annotations

import unittest


class GradingPromptContract(unittest.TestCase):
    def test_system_prompt_is_concise_and_has_no_worked_example(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertNotIn(
            "Worked example:",
            prompt,
            "system prompt should state the rule directly, not sermonize with a worked example",
        )
        self.assertLess(
            len(prompt),
            2200,
            "system prompt should stay compact enough that easy items do not pay for repeated policy prose",
        )

    def test_system_prompt_states_each_major_rule_once(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT.lower()
        self.assertEqual(
            prompt.count("answered-form rule"),
            1,
            "answered-form rule should be introduced once, clearly",
        )
        self.assertLessEqual(
            prompt.count("upstream"),
            4,
            "upstream-dependency handling should be clear without being repeated to death",
        )
        self.assertLessEqual(
            prompt.count("consistency"),
            2,
            "consistency rule should be stated cleanly, not reiterated in multiple phrasings",
        )

    def test_system_prompt_uses_explicit_rescue_credit_language(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertNotIn(
            "Be charitable.",
            prompt,
            "generic charity language is too vague for the grading target here",
        )
        self.assertNotIn(
            "erring on the side of generosity",
            prompt,
            "prompt should ask for the actual rescue objective, not vague generosity phrasing",
        )
        self.assertIn(
            "Award the highest score justified by the student's written work under the rubric.",
            prompt,
            "prompt should optimize for the highest justified score, not generic leniency",
        )
        self.assertIn(
            "Actively rescue as much lawful partial credit as possible",
            prompt,
            "prompt should explicitly say to rescue rubric-grounded partial credit",
        )

    def test_system_prompt_defaults_dependency_to_none_unless_clear(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            'Use upstream_dependency = "none" unless this answer clearly carries forward an earlier part.',
            prompt,
            "dependency handling should default to none unless the carry-forward is clear",
        )

    def test_system_prompt_limits_charitable_reread_loops(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Grade what is written, not a more favorable answer you can imagine.",
            prompt,
            "prompt should stop the model from rescuing borderline OCR reads through speculation",
        )
        self.assertIn(
            "If two readings are plausible and neither is clearly better supported, choose the best-supported reading and move on.",
            prompt,
            "prompt should narrow ambiguity handling so the model does not loop on rereads",
        )

    def test_system_prompt_tells_easy_wrong_form_items_to_stop_early(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "If the student plainly did not provide the requested answer form, stop once that is established and score only what is actually on the page.",
            prompt,
            "easy wrong-form items should not invite long re-litigation after the missing form is already clear",
        )

    def test_system_prompt_uses_bounded_effort_handoff_for_hard_ambiguity(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "If ambiguity still materially affects the score after one careful pass, choose the best-supported reading, say in model_reasoning that human review is warranted, lower model_confidence, and stop.",
            prompt,
            "hard ambiguous items should hand off cleanly once bounded effort is exhausted",
        )


if __name__ == "__main__":
    unittest.main()
