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

    def test_system_prompt_prefers_lawful_full_credit_and_equivalent_units(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "If the student's work supports a lawful full-credit interpretation, take it and stop.",
            prompt,
            "prompt should prefer a supportable full-credit reading over continued nitpicking",
        )
        self.assertIn(
            "Equivalent volume units such as mL and cm³ count as the same quantity unless the question explicitly tests a specific form.",
            prompt,
            "prompt should treat mL and cm³ as equivalent when the form itself is not being tested",
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

    def test_system_prompt_declares_obvious_correctness_buckets(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Use is_obviously_fully_correct = true only when the answer is clearly correct and needs no human rescue.",
            prompt,
            "prompt should expose a high-trust obvious-full-credit bucket",
        )
        self.assertIn(
            "Use is_obviously_wrong = true only when the answer is clearly wrong and no lawful rescue path remains.",
            prompt,
            "prompt should expose a high-trust obvious-wrong bucket",
        )
        self.assertIn(
            '"is_obviously_fully_correct": <true | false | null>',
            prompt,
            "the JSON schema should persist the obvious-full-credit bucket",
        )
        self.assertIn(
            '"is_obviously_wrong": <true | false | null>',
            prompt,
            "the JSON schema should persist the obvious-wrong bucket",
        )

    def test_system_prompt_keeps_obvious_wrong_out_of_partial_credit_cases(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Do not use is_obviously_wrong = true if any lawful partial-credit path remains.",
            prompt,
            "obvious-wrong should be reserved for true zero-credit cases, not harsh partial-credit judgments",
        )

    def test_system_prompt_blocks_rescue_credit_on_answered_form_failures(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "When the requested form is itself the thing being graded, do not award rescue credit for nearby ingredients of the answer unless the rubric explicitly does so.",
            prompt,
            "answered-form failures should not pick up rescue points just for mentioning nearby chemistry",
        )


if __name__ == "__main__":
    unittest.main()
