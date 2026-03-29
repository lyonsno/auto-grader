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


if __name__ == "__main__":
    unittest.main()
