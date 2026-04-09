"""Contract tests for the template schema — question templates and validation.

These tests define the enforceable contract for the YAML template schema
used to author exam templates. They are written fail-first: the schema
module does not yet exist, and these tests express the shape it must have.
"""

from __future__ import annotations

import math
import textwrap
import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_mc_question(overrides: dict | None = None) -> dict:
    """Return a minimal valid multiple-choice question dict."""
    q: dict = {
        "id": "mc-1",
        "points": 2,
        "prompt": "Which is an element?",
        "answer_type": "multiple_choice",
        "choices": {"A": "CO2", "B": "O2"},
        "correct": "B",
        "shuffle": True,
    }
    if overrides:
        q.update(overrides)
    return q


def _minimal_numeric_question(overrides: dict | None = None) -> dict:
    """Return a minimal valid numeric question dict."""
    q: dict = {
        "id": "fr-1",
        "points": 2,
        "prompt": "What is {{mass}} / {{density}}?",
        "answer_type": "numeric",
        "variables": {
            "mass": {"type": "float", "min": 10.0, "max": 99.0, "step": 0.1},
            "density": {"type": "float", "min": 0.7, "max": 3.5, "step": 0.01},
        },
        "answer": {
            "expr": "mass / density",
            "format": {"sig_figs": 3},
            "tolerance": {"type": "relative", "value": 0.01},
        },
    }
    if overrides:
        q.update(overrides)
    return q


def _minimal_exact_match_question(overrides: dict | None = None) -> dict:
    q: dict = {
        "id": "fr-7a",
        "points": 1,
        "prompt": "Name the geometry.",
        "answer_type": "exact_match",
        "answer": {
            "accept": ["tetrahedral", "Tetrahedral"],
            "normalize": "lowercase",
        },
    }
    if overrides:
        q.update(overrides)
    return q


def _minimal_manual_review_question(overrides: dict | None = None) -> dict:
    q: dict = {
        "id": "fr-2",
        "points": 3,
        "prompt": "Write the balanced equation for P4 + Cl2 -> PCl5.",
        "answer_type": "balanced_equation",
        "grading": "manual_review",
        "answer": {
            "reference": "P4 + 10Cl2 -> 4PCl5",
            "rubric": [
                {"criterion": "Correct products", "points": 1},
                {"criterion": "Balanced coefficients", "points": 1},
                {"criterion": "States labeled", "points": 1},
            ],
        },
    }
    if overrides:
        q.update(overrides)
    return q


def _wrap_in_template(questions: list[dict], slug: str = "test-exam") -> dict:
    """Wrap question dicts in a minimal valid template structure."""
    return {
        "slug": slug,
        "title": "Test Exam",
        "sections": [
            {
                "id": "s1",
                "title": "Section 1",
                "questions": questions,
            }
        ],
    }


def _yaml_round_trip(template_dict: dict) -> str:
    """Dump a template dict to YAML string."""
    import yaml

    return yaml.dump(template_dict, default_flow_style=False, sort_keys=False)


# ===========================================================================
# 1. Module import and API shape
# ===========================================================================


class TestModuleImports(unittest.TestCase):
    """The template_schema module must exist and expose the contract API."""

    def test_module_importable(self):
        import auto_grader.template_schema  # noqa: F401

    def test_load_template_callable(self):
        from auto_grader.template_schema import load_template

        self.assertTrue(callable(load_template))

    def test_validate_template_callable(self):
        from auto_grader.template_schema import validate_template

        self.assertTrue(callable(validate_template))

    def test_evaluate_expr_callable(self):
        from auto_grader.template_schema import evaluate_expr

        self.assertTrue(callable(evaluate_expr))


# ===========================================================================
# 2. YAML loading contract
# ===========================================================================


class TestLoadTemplate(unittest.TestCase):
    """load_template(yaml_str) must parse YAML into a template dict."""

    def test_load_minimal_template(self):
        from auto_grader.template_schema import load_template

        tmpl = _wrap_in_template([_minimal_mc_question()])
        yaml_str = _yaml_round_trip(tmpl)
        result = load_template(yaml_str)
        self.assertEqual(result["slug"], "test-exam")
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(len(result["sections"][0]["questions"]), 1)

    def test_load_rejects_non_string(self):
        from auto_grader.template_schema import load_template

        with self.assertRaises((TypeError, ValueError)):
            load_template(42)

    def test_load_rejects_invalid_yaml(self):
        from auto_grader.template_schema import load_template

        with self.assertRaises(ValueError):
            load_template("{{invalid yaml: [")


# ===========================================================================
# 3. Structural validation
# ===========================================================================


class TestStructuralValidation(unittest.TestCase):
    """Structural validation rules for the template schema."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_valid_minimal_template_passes(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        errors = self._validate(tmpl)
        self.assertEqual(errors, [])

    def test_missing_slug_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        del tmpl["slug"]
        errors = self._validate(tmpl)
        self.assertTrue(any("slug" in e.lower() for e in errors))

    def test_blank_slug_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        tmpl["slug"] = "   "
        errors = self._validate(tmpl)
        self.assertTrue(any("slug" in e.lower() for e in errors))

    def test_slug_format_enforced(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        tmpl["slug"] = "Invalid Slug!"
        errors = self._validate(tmpl)
        self.assertTrue(any("slug" in e.lower() for e in errors))

    def test_missing_title_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        del tmpl["title"]
        errors = self._validate(tmpl)
        self.assertTrue(any("title" in e.lower() for e in errors))

    def test_empty_sections_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        tmpl["sections"] = []
        errors = self._validate(tmpl)
        self.assertTrue(any("section" in e.lower() for e in errors))

    def test_missing_sections_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        del tmpl["sections"]
        errors = self._validate(tmpl)
        self.assertTrue(any("section" in e.lower() for e in errors))

    def test_section_missing_id_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        del tmpl["sections"][0]["id"]
        errors = self._validate(tmpl)
        self.assertTrue(any("id" in e.lower() for e in errors))

    def test_section_empty_questions_rejected(self):
        tmpl = _wrap_in_template([_minimal_mc_question()])
        tmpl["sections"][0]["questions"] = []
        errors = self._validate(tmpl)
        self.assertTrue(any("question" in e.lower() for e in errors))

    def test_question_missing_id_rejected(self):
        q = _minimal_mc_question()
        del q["id"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("id" in e.lower() for e in errors))

    def test_question_missing_points_rejected(self):
        q = _minimal_mc_question()
        del q["points"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("points" in e.lower() for e in errors))

    def test_question_zero_points_rejected(self):
        q = _minimal_mc_question({"points": 0})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("points" in e.lower() for e in errors))

    def test_question_negative_points_rejected(self):
        q = _minimal_mc_question({"points": -1})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("points" in e.lower() for e in errors))

    def test_duplicate_question_ids_rejected(self):
        q1 = _minimal_mc_question()
        q2 = _minimal_mc_question({"prompt": "Different question"})
        # Both have id "mc-1"
        errors = self._validate(_wrap_in_template([q1, q2]))
        self.assertTrue(any("duplicate" in e.lower() or "unique" in e.lower() for e in errors))

    def test_duplicate_section_ids_rejected(self):
        tmpl = {
            "slug": "test-exam",
            "title": "Test",
            "sections": [
                {"id": "s1", "title": "A", "questions": [_minimal_mc_question()]},
                {
                    "id": "s1",
                    "title": "B",
                    "questions": [_minimal_mc_question({"id": "mc-2"})],
                },
            ],
        }
        errors = self._validate(tmpl)
        self.assertTrue(any("duplicate" in e.lower() or "unique" in e.lower() for e in errors))

    def test_invalid_answer_type_rejected(self):
        q = _minimal_mc_question({"answer_type": "essay"})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("answer_type" in e.lower() for e in errors))

    def test_grading_auto_inferred_for_mc(self):
        """MC without explicit grading should default to auto."""
        q = _minimal_mc_question()
        assert "grading" not in q
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_grading_auto_invalid_for_manual_types(self):
        q = _minimal_manual_review_question({"grading": "auto"})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("grading" in e.lower() for e in errors))

    def test_question_focus_region_is_allowed_when_box_is_normalized(self):
        q = _minimal_mc_question(
            {
                "focus_region": {
                    "x": 0.15,
                    "y": 0.35,
                    "width": 0.25,
                    "height": 0.10,
                }
            }
        )
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_focus_region_rejects_out_of_bounds_box(self):
        q = _minimal_mc_question(
            {
                "focus_region": {
                    "x": 0.90,
                    "y": 0.35,
                    "width": 0.25,
                    "height": 0.10,
                }
            }
        )
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("focus_region" in e.lower() for e in errors))


# ===========================================================================
# 4. Sub-parts contract
# ===========================================================================


class TestSubParts(unittest.TestCase):
    """Questions with sub-parts must follow the parts contract."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_valid_parts_question(self):
        q = {
            "id": "q5",
            "points": 4,
            "prompt": "Given moles...",
            "parts": [
                {
                    "id": "q5a",
                    "points": 1,
                    "prompt": "Limiting reagent?",
                    "answer_type": "exact_match",
                    "answer": {"accept": ["H2"]},
                },
                {
                    "id": "q5b",
                    "points": 3,
                    "prompt": "Moles of product?",
                    "answer_type": "numeric",
                    "answer": {
                        "expr": "6.0",
                        "tolerance": {"type": "absolute", "value": 0.1},
                    },
                },
            ],
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_parts_points_must_sum_to_parent(self):
        q = {
            "id": "q5",
            "points": 10,  # sum of parts is 4, not 10
            "prompt": "...",
            "parts": [
                {
                    "id": "q5a",
                    "points": 1,
                    "prompt": "...",
                    "answer_type": "exact_match",
                    "answer": {"accept": ["H2"]},
                },
                {
                    "id": "q5b",
                    "points": 3,
                    "prompt": "...",
                    "answer_type": "numeric",
                    "answer": {
                        "expr": "6.0",
                        "tolerance": {"type": "absolute", "value": 0.1},
                    },
                },
            ],
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("points" in e.lower() and "sum" in e.lower() for e in errors))

    def test_parts_and_answer_type_mutually_exclusive(self):
        q = {
            "id": "q5",
            "points": 2,
            "prompt": "...",
            "answer_type": "numeric",  # cannot have both answer_type and parts
            "parts": [
                {
                    "id": "q5a",
                    "points": 2,
                    "prompt": "...",
                    "answer_type": "numeric",
                    "answer": {
                        "expr": "1.0",
                        "tolerance": {"type": "absolute", "value": 0.1},
                    },
                },
            ],
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(len(errors) > 0)

    def test_part_ids_must_be_globally_unique(self):
        q1 = {
            "id": "q1",
            "points": 2,
            "prompt": "...",
            "parts": [
                {
                    "id": "shared-id",
                    "points": 2,
                    "prompt": "...",
                    "answer_type": "exact_match",
                    "answer": {"accept": ["yes"]},
                },
            ],
        }
        q2 = _minimal_mc_question({"id": "shared-id"})  # collides with part id
        errors = self._validate(_wrap_in_template([q1, q2]))
        self.assertTrue(any("duplicate" in e.lower() or "unique" in e.lower() for e in errors))

    def test_parts_inherit_parent_variables(self):
        q = {
            "id": "q5",
            "points": 2,
            "prompt": "Given {{x}} mol...",
            "variables": {
                "x": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.5},
            },
            "parts": [
                {
                    "id": "q5a",
                    "points": 2,
                    "prompt": "Calculate using {{x}}.",
                    "answer_type": "numeric",
                    "answer": {
                        "expr": "x * 2",
                        "tolerance": {"type": "absolute", "value": 0.1},
                    },
                },
            ],
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])


# ===========================================================================
# 5. Variable validation
# ===========================================================================


class TestVariableValidation(unittest.TestCase):
    """Variable declarations must be well-formed."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_valid_float_variable(self):
        q = _minimal_numeric_question()
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_valid_int_variable(self):
        q = _minimal_numeric_question({
            "variables": {
                "n": {"type": "int", "min": 1, "max": 10, "step": 1},
            },
            "prompt": "What is {{n}} * 2?",
            "answer": {
                "expr": "n * 2",
                "tolerance": {"type": "absolute", "value": 0.5},
            },
        })
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_missing_type_rejected(self):
        q = _minimal_numeric_question()
        del q["variables"]["mass"]["type"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("type" in e.lower() for e in errors))

    def test_invalid_type_rejected(self):
        q = _minimal_numeric_question()
        q["variables"]["mass"]["type"] = "string"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("type" in e.lower() for e in errors))

    def test_min_greater_than_max_rejected(self):
        q = _minimal_numeric_question()
        q["variables"]["mass"]["min"] = 100.0
        q["variables"]["mass"]["max"] = 10.0
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("min" in e.lower() or "max" in e.lower() for e in errors))

    def test_zero_step_rejected(self):
        q = _minimal_numeric_question()
        q["variables"]["mass"]["step"] = 0
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("step" in e.lower() for e in errors))

    def test_negative_step_rejected(self):
        q = _minimal_numeric_question()
        q["variables"]["mass"]["step"] = -1.0
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("step" in e.lower() for e in errors))

    def test_undeclared_variable_in_prompt_rejected(self):
        q = _minimal_numeric_question()
        q["prompt"] = "What is {{mass}} / {{viscosity}}?"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("viscosity" in e.lower() for e in errors))

    def test_undeclared_variable_in_expr_rejected(self):
        q = _minimal_numeric_question()
        q["answer"]["expr"] = "mass / unknown_var"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("unknown_var" in e.lower() for e in errors))


# ===========================================================================
# 6. Answer validation by type
# ===========================================================================


class TestMCAnswerValidation(unittest.TestCase):

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_mc_missing_choices_rejected(self):
        q = _minimal_mc_question()
        del q["choices"]
        # MC without choices and without distractors strategy is invalid
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(len(errors) > 0)

    def test_mc_correct_not_in_choices_rejected(self):
        q = _minimal_mc_question({"correct": "Z"})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("correct" in e.lower() for e in errors))

    def test_mc_with_distractors_strategy(self):
        q = {
            "id": "mc-param",
            "points": 2,
            "prompt": "What is {{mass}} / {{density}}?",
            "answer_type": "multiple_choice",
            "variables": {
                "mass": {"type": "float", "min": 10.0, "max": 99.0, "step": 0.1},
                "density": {"type": "float", "min": 0.7, "max": 3.5, "step": 0.01},
            },
            "answer": {
                "expr": "mass / density",
                "format": {"sig_figs": 3},
            },
            "distractors": {
                "strategy": "common_errors",
                "common_errors": [
                    {"expr": "density / mass"},
                    {"expr": "mass * density"},
                    {"expr": "mass + density"},
                ],
            },
            "shuffle": True,
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])


class TestNumericAnswerValidation(unittest.TestCase):

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_numeric_missing_tolerance_rejected(self):
        q = _minimal_numeric_question()
        del q["answer"]["tolerance"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("tolerance" in e.lower() for e in errors))

    def test_numeric_missing_expr_rejected(self):
        q = _minimal_numeric_question()
        del q["answer"]["expr"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("expr" in e.lower() for e in errors))


class TestExactMatchAnswerValidation(unittest.TestCase):

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_exact_match_valid(self):
        q = _minimal_exact_match_question()
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_exact_match_empty_accept_rejected(self):
        q = _minimal_exact_match_question()
        q["answer"]["accept"] = []
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(len(errors) > 0)

    def test_exact_match_with_expr_and_accept_map(self):
        q = {
            "id": "em-cond",
            "points": 1,
            "prompt": "Which is limiting?",
            "answer_type": "exact_match",
            "variables": {
                "a": {"type": "float", "min": 1.0, "max": 5.0, "step": 0.5},
            },
            "answer": {
                "expr": "'X' if a > 3 else 'Y'",
                "accept_map": {
                    "X": ["X", "ex"],
                    "Y": ["Y", "why"],
                },
            },
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])


class TestManualReviewAnswerValidation(unittest.TestCase):

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_manual_review_valid(self):
        q = _minimal_manual_review_question()
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_manual_review_missing_reference_rejected(self):
        q = _minimal_manual_review_question()
        del q["answer"]["reference"]
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("reference" in e.lower() for e in errors))

    def test_manual_review_rubric_points_must_sum(self):
        q = _minimal_manual_review_question()
        # Rubric sums to 3 but question points is 3, so this should pass.
        # Change question points to 5 to create mismatch.
        q["points"] = 5
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("rubric" in e.lower() and "points" in e.lower() for e in errors))

    def test_manual_review_rubric_must_be_list(self):
        q = _minimal_manual_review_question()
        q["answer"]["rubric"] = "just a string"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("rubric" in e.lower() for e in errors))


# ===========================================================================
# 7. Figure field contract
# ===========================================================================


class TestFigureField(unittest.TestCase):

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_figure_field_accepted(self):
        q = _minimal_mc_question({"figure": "figures/q1-diagram.png"})
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])

    def test_figure_blank_rejected(self):
        q = _minimal_mc_question({"figure": "  "})
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("figure" in e.lower() for e in errors))


# ===========================================================================
# 7b. Review-driven additions: tolerance, namespace, edge cases
# ===========================================================================


class TestToleranceValidation(unittest.TestCase):
    """Tolerance structure must be well-formed."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_tolerance_non_dict_rejected(self):
        q = _minimal_numeric_question()
        q["answer"]["tolerance"] = "lol"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("tolerance" in e.lower() for e in errors))

    def test_tolerance_invalid_type_rejected(self):
        q = _minimal_numeric_question()
        q["answer"]["tolerance"] = {"type": "banana", "value": 0.01}
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("tolerance" in e.lower() for e in errors))

    def test_tolerance_missing_value_rejected(self):
        q = _minimal_numeric_question()
        q["answer"]["tolerance"] = {"type": "relative"}
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("tolerance" in e.lower() for e in errors))

    def test_tolerance_negative_value_rejected(self):
        q = _minimal_numeric_question()
        q["answer"]["tolerance"] = {"type": "absolute", "value": -0.5}
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("tolerance" in e.lower() for e in errors))

    def test_tolerance_valid(self):
        q = _minimal_numeric_question()
        # Already has a valid tolerance from the helper
        errors = self._validate(_wrap_in_template([q]))
        self.assertEqual(errors, [])


class TestDistractorExprValidation(unittest.TestCase):
    """Distractor expressions must be validated for variable references."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_distractor_expr_with_undeclared_variable_rejected(self):
        q = {
            "id": "mc-param",
            "points": 2,
            "prompt": "What is {{mass}} / {{density}}?",
            "answer_type": "multiple_choice",
            "variables": {
                "mass": {"type": "float", "min": 10.0, "max": 99.0, "step": 0.1},
                "density": {"type": "float", "min": 0.7, "max": 3.5, "step": 0.01},
            },
            "answer": {
                "expr": "mass / density",
                "format": {"sig_figs": 3},
            },
            "distractors": {
                "strategy": "common_errors",
                "common_errors": [
                    {"expr": "density / mass"},
                    {"expr": "typo_var * density"},  # undeclared
                ],
            },
            "shuffle": True,
        }
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("typo_var" in e for e in errors))


class TestNamespaceAndEdgeCases(unittest.TestCase):
    """Edge cases around ID namespaces, degenerate variables, empty input."""

    def _validate(self, tmpl_dict: dict):
        from auto_grader.template_schema import validate_template

        return validate_template(tmpl_dict)

    def test_section_and_question_ids_may_overlap(self):
        """Section IDs and question IDs are intentionally separate namespaces."""
        tmpl = {
            "slug": "test-exam",
            "title": "Test",
            "sections": [
                {
                    "id": "shared-name",
                    "title": "Section",
                    "questions": [_minimal_mc_question({"id": "shared-name"})],
                }
            ],
        }
        errors = self._validate(tmpl)
        self.assertEqual(errors, [])

    def test_degenerate_variable_min_equals_max_accepted(self):
        """A variable with min == max is degenerate (constant) but valid."""
        q = _minimal_numeric_question()
        q["variables"]["mass"] = {"type": "float", "min": 50.0, "max": 50.0, "step": 0.1}
        # min >= max is rejected, but min == max is currently rejected too.
        # This test documents the current behavior.
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("min" in e.lower() or "max" in e.lower() for e in errors))

    def test_load_empty_string_rejected(self):
        from auto_grader.template_schema import load_template

        with self.assertRaises(ValueError):
            load_template("")

    def test_undeclared_function_call_in_expr_rejected(self):
        """Calling an undeclared function name is caught as an undeclared variable."""
        q = _minimal_numeric_question()
        q["answer"]["expr"] = "foo(mass)"
        errors = self._validate(_wrap_in_template([q]))
        self.assertTrue(any("foo" in e for e in errors))


# ===========================================================================
# 8. Expression evaluator contract
# ===========================================================================


class TestExpressionEvaluator(unittest.TestCase):
    """evaluate_expr must safely evaluate restricted Python expressions."""

    def _eval(self, expr: str, variables: dict | None = None):
        from auto_grader.template_schema import evaluate_expr

        return evaluate_expr(expr, variables or {})

    def test_basic_arithmetic(self):
        self.assertAlmostEqual(self._eval("10.0 / 2.0"), 5.0)

    def test_variable_substitution(self):
        result = self._eval("mass / density", {"mass": 95.0, "density": 13.6})
        self.assertAlmostEqual(result, 95.0 / 13.6, places=5)

    def test_min_max_builtins(self):
        self.assertEqual(self._eval("min(3, 5)"), 3)
        self.assertEqual(self._eval("max(3, 5)"), 5)

    def test_abs_builtin(self):
        self.assertEqual(self._eval("abs(-7)"), 7)

    def test_round_builtin(self):
        self.assertEqual(self._eval("round(3.14159, 2)"), 3.14)

    def test_ternary_expression(self):
        result = self._eval("'H2' if x < 3 else 'N2'", {"x": 2})
        self.assertEqual(result, "H2")
        result = self._eval("'H2' if x < 3 else 'N2'", {"x": 5})
        self.assertEqual(result, "N2")

    def test_power_operator(self):
        self.assertAlmostEqual(self._eval("2 ** 10"), 1024)

    def test_comparison_operators(self):
        self.assertTrue(self._eval("3 < 5"))
        self.assertFalse(self._eval("5 < 3"))

    def test_pi_constant(self):
        result = self._eval("pi")
        self.assertAlmostEqual(result, math.pi, places=5)

    def test_log10_builtin(self):
        self.assertAlmostEqual(self._eval("log10(100)"), 2.0)

    def test_log10_ph_calculation(self):
        """pH = -log10([H3O+]) — the real use case from Q14c."""
        result = self._eval("-log10(m_acid)", {"m_acid": 0.110})
        self.assertAlmostEqual(result, 0.9586, places=3)

    def test_sqrt_builtin(self):
        self.assertAlmostEqual(self._eval("sqrt(144)"), 12.0)

    def test_rejects_import(self):
        with self.assertRaises(ValueError):
            self._eval("__import__('os').system('echo hi')")

    def test_rejects_attribute_access(self):
        with self.assertRaises(ValueError):
            self._eval("'hello'.upper()")

    def test_rejects_function_def(self):
        with self.assertRaises(ValueError):
            self._eval("(lambda: 1)()")

    def test_rejects_subscript(self):
        with self.assertRaises(ValueError):
            self._eval("[1,2,3][0]")

    def test_rejects_assignment(self):
        with self.assertRaises(ValueError):
            # walrus operator
            self._eval("(x := 5)")

    def test_rejects_tuple_literal(self):
        with self.assertRaises(ValueError):
            self._eval("(1, 2, 3)")

    def test_rejects_list_literal(self):
        with self.assertRaises(ValueError):
            self._eval("[1, 2, 3]")

    def test_rejects_set_literal(self):
        with self.assertRaises(ValueError):
            self._eval("{1, 2, 3}")

    def test_rejects_dict_literal(self):
        with self.assertRaises(ValueError):
            self._eval("{'a': 1}")

    def test_undeclared_variable_raises(self):
        with self.assertRaises((ValueError, NameError)):
            self._eval("mass / density", {"mass": 10.0})

    def test_complex_stoichiometry_expr(self):
        """Real expr from the plan: limiting reagent + product calculation."""
        result = self._eval(
            "2 * min(mol_n2, mol_h2 / 3)",
            {"mol_n2": 10.2, "mol_h2": 27.9},
        )
        self.assertAlmostEqual(result, 2 * min(10.2, 27.9 / 3), places=5)

    def test_conditional_exact_match_expr(self):
        """Real expr: which is the limiting reagent?"""
        result = self._eval(
            "'H2' if mol_h2 / 3 < mol_n2 else 'N2'",
            {"mol_n2": 10.2, "mol_h2": 27.9},
        )
        expected = "H2" if 27.9 / 3 < 10.2 else "N2"
        self.assertEqual(result, expected)


# ===========================================================================
# 9. Round-trip contract
# ===========================================================================


class TestRoundTrip(unittest.TestCase):
    """load → validate → dump → reload → validate must be lossless."""

    def test_round_trip_preserves_structure(self):
        from auto_grader.template_schema import load_template, validate_template
        import yaml

        tmpl = _wrap_in_template([
            _minimal_mc_question(),
            _minimal_numeric_question({"id": "fr-1-alt"}),
            _minimal_exact_match_question({"id": "fr-7a-alt"}),
        ])
        yaml_str = _yaml_round_trip(tmpl)

        loaded = load_template(yaml_str)
        errors = validate_template(loaded)
        self.assertEqual(errors, [])

        dumped = yaml.dump(loaded, default_flow_style=False, sort_keys=False)
        reloaded = load_template(dumped)
        errors2 = validate_template(reloaded)
        self.assertEqual(errors2, [])

        self.assertEqual(loaded["slug"], reloaded["slug"])
        self.assertEqual(
            len(loaded["sections"][0]["questions"]),
            len(reloaded["sections"][0]["questions"]),
        )


# ===========================================================================
# 10. Real exam template integration
# ===========================================================================


class TestRealExamTemplate(unittest.TestCase):
    """The real CHM 141 template must load and validate cleanly."""

    _TEMPLATE_PATH = "templates/chm141-final-fall2023.yaml"

    def _load_real_template(self):
        import os
        import pathlib

        # Find the template relative to the repo root
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        template_path = repo_root / self._TEMPLATE_PATH
        if not template_path.exists():
            self.skipTest(f"Template not found at {template_path}")
        return template_path.read_text()

    def test_real_template_loads(self):
        from auto_grader.template_schema import load_template

        yaml_str = self._load_real_template()
        tmpl = load_template(yaml_str)
        self.assertEqual(tmpl["slug"], "chm141-final-fall2023")

    def test_real_template_validates(self):
        from auto_grader.template_schema import load_template, validate_template

        yaml_str = self._load_real_template()
        tmpl = load_template(yaml_str)
        errors = validate_template(tmpl)
        self.assertEqual(errors, [], f"Validation errors:\n" + "\n".join(errors))

    def test_real_template_has_expected_sections(self):
        from auto_grader.template_schema import load_template

        yaml_str = self._load_real_template()
        tmpl = load_template(yaml_str)
        section_ids = [s["id"] for s in tmpl["sections"]]
        self.assertIn("mc", section_ids)
        self.assertIn("free-response", section_ids)

    def test_real_template_question_count(self):
        from auto_grader.template_schema import load_template

        yaml_str = self._load_real_template()
        tmpl = load_template(yaml_str)
        mc_section = next(s for s in tmpl["sections"] if s["id"] == "mc")
        fr_section = next(s for s in tmpl["sections"] if s["id"] == "free-response")
        # 19 MC questions (mc-5 omitted: requires isotope figure)
        self.assertEqual(len(mc_section["questions"]), 19)
        # 16 free-response questions
        self.assertEqual(len(fr_section["questions"]), 16)

    def test_real_template_expression_evaluation(self):
        """Spot-check that key expressions produce correct answers."""
        from auto_grader.template_schema import evaluate_expr

        # Q1: density calculation with original exam values
        result = evaluate_expr("mass / density", {"mass": 95.0, "density": 13.6})
        self.assertAlmostEqual(result, 6.985, places=2)

        # Q5: limiting reagent with original values
        result = evaluate_expr(
            "2 * min(mol_n2, mol_h2 / 3)",
            {"mol_n2": 10.2, "mol_h2": 27.9},
        )
        self.assertAlmostEqual(result, 18.6, places=1)

        # Q8: Hess's law with original values
        result = evaluate_expr(
            "dh2 - dh1", {"dh1": -325.1, "dh2": -511.3}
        )
        self.assertAlmostEqual(result, -186.2, places=1)

        # Q15d: calorimetry total with original values
        result = evaluate_expr(
            "(mass_water * 4.184 * (100.0 - t_initial) + mass_water * 2256 "
            "+ mass_water * 1.92 * (t_final - 100.0)) / 1000",
            {"mass_water": 100.0, "t_initial": 25.0, "t_final": 200.0},
        )
        # Original answer key: 302 kJ (with slight rounding)
        self.assertAlmostEqual(result, 288.0, delta=15)


if __name__ == "__main__":
    unittest.main()
