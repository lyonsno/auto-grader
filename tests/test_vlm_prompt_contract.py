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


if __name__ == "__main__":
    unittest.main()
