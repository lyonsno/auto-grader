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

    def test_prompt_contract_publishes_version_and_hash(self):
        from auto_grader import vlm_inference

        version = getattr(vlm_inference, "GRADING_PROMPT_VERSION", "")
        self.assertRegex(
            version,
            r"^\d{4}-\d{2}-\d{2}-",
            "grading prompt should declare a human-readable version string in code",
        )

        metadata_fn = getattr(vlm_inference, "grading_prompt_metadata", None)
        self.assertTrue(
            callable(metadata_fn),
            "vlm prompt contract should publish grading_prompt_metadata() for run manifests",
        )
        metadata = metadata_fn()
        self.assertEqual(metadata["version"], version)
        self.assertRegex(metadata["content_hash"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
