"""Template schema: loading, validation, and expression evaluation for exam templates.

This module defines the contract API for YAML exam templates:
- load_template(yaml_str) -> dict: parse YAML into a template dict
- validate_template(tmpl) -> list[str]: return validation errors (empty = valid)
- evaluate_expr(expr_str, variables) -> float | str: safely evaluate a restricted
  Python expression with variable substitution
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_template(yaml_str: Any) -> dict:
    """Parse a YAML string into a template dict.

    Raises TypeError for non-string input, ValueError for unparseable YAML
    or YAML that does not produce a dict.
    """
    if not isinstance(yaml_str, str):
        raise TypeError(f"Expected str, got {type(yaml_str).__name__}")

    import yaml

    try:
        result = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("YAML must produce a mapping at the top level")

    return result


def validate_template(tmpl: dict) -> list[str]:
    """Validate a template dict against the schema contract.

    Returns a list of human-readable error strings. An empty list means
    the template is valid.
    """
    errors: list[str] = []
    _validate_top_level(tmpl, errors)
    return errors


def evaluate_expr(expr_str: str, variables: dict[str, Any] | None = None) -> Any:
    """Safely evaluate a restricted Python expression.

    Allowed: arithmetic, comparison, ternary, min/max/abs/round, pi,
    and declared variable names. Everything else raises ValueError.
    """
    variables = variables or {}
    tree = _parse_and_validate_ast(expr_str)
    return _eval_ast(tree, variables)


# ---------------------------------------------------------------------------
# Validation internals
# ---------------------------------------------------------------------------

_VALID_ANSWER_TYPES = frozenset({
    "multiple_choice",
    "numeric",
    "exact_match",
    "balanced_equation",
    "lewis_structure",
    "electron_config",
    "text_reasoning",
})

_AUTO_GRADABLE_TYPES = frozenset({
    "multiple_choice",
    "numeric",
    "exact_match",
})

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _validate_top_level(tmpl: dict, errors: list[str]) -> None:
    # slug
    slug = tmpl.get("slug")
    if slug is None:
        errors.append("Missing required field: slug")
    elif not isinstance(slug, str) or not slug.strip():
        errors.append("slug must be a non-blank string")
    elif not _SLUG_RE.match(slug):
        errors.append(
            f"slug must match [a-z0-9][a-z0-9-]*, got: {slug!r}"
        )

    # title
    if "title" not in tmpl:
        errors.append("Missing required field: title")
    elif not isinstance(tmpl["title"], str) or not tmpl["title"].strip():
        errors.append("title must be a non-blank string")

    # sections
    sections = tmpl.get("sections")
    if sections is None:
        errors.append("Missing required field: sections")
        return
    if not isinstance(sections, list) or len(sections) == 0:
        errors.append("sections must be a non-empty list")
        return

    all_ids: set[str] = set()
    section_ids: set[str] = set()

    for i, section in enumerate(sections):
        _validate_section(section, i, errors, all_ids, section_ids)


def _validate_section(
    section: dict,
    index: int,
    errors: list[str],
    all_ids: set[str],
    section_ids: set[str],
) -> None:
    prefix = f"sections[{index}]"

    sid = section.get("id")
    if sid is None:
        errors.append(f"{prefix}: Missing required field: id")
    elif sid in section_ids:
        errors.append(f"{prefix}: Duplicate section id: {sid!r}")
    else:
        section_ids.add(sid)

    questions = section.get("questions")
    if questions is None or not isinstance(questions, list) or len(questions) == 0:
        errors.append(f"{prefix}: questions must be a non-empty list")
        return

    for j, question in enumerate(questions):
        _validate_question(question, f"{prefix}.questions[{j}]", errors, all_ids, variables=None)


def _validate_question(
    q: dict,
    path: str,
    errors: list[str],
    all_ids: set[str],
    variables: dict | None,
) -> None:
    # id
    qid = q.get("id")
    if qid is None:
        errors.append(f"{path}: Missing required field: id")
    elif qid in all_ids:
        errors.append(f"{path}: Duplicate id: {qid!r} (ids must be unique across the template)")
    else:
        all_ids.add(qid)

    # points
    points = q.get("points")
    if points is None:
        errors.append(f"{path}: Missing required field: points")
    elif not isinstance(points, (int, float)) or points <= 0:
        errors.append(f"{path}: points must be a positive number, got {points!r}")

    # figure
    figure = q.get("figure")
    if figure is not None:
        if not isinstance(figure, str) or not figure.strip():
            errors.append(f"{path}: figure must be a non-blank string")

    # Resolve variables: merge parent variables with question-level variables
    q_vars = q.get("variables", {})
    merged_vars = dict(variables or {})
    merged_vars.update(q_vars)

    # Validate variable declarations
    if q_vars:
        _validate_variables(q_vars, path, errors)

    # parts vs answer_type mutual exclusivity
    has_parts = "parts" in q
    has_answer_type = "answer_type" in q

    if has_parts and has_answer_type:
        errors.append(
            f"{path}: Cannot have both 'answer_type' and 'parts' — use one or the other"
        )
        return

    if has_parts:
        _validate_parts(q, path, errors, all_ids, merged_vars)
    elif has_answer_type:
        _validate_answer_typed_question(q, path, errors, merged_vars)
    else:
        errors.append(f"{path}: Question must have either 'answer_type' or 'parts'")


def _validate_parts(
    q: dict,
    path: str,
    errors: list[str],
    all_ids: set[str],
    parent_vars: dict,
) -> None:
    parts = q["parts"]
    if not isinstance(parts, list) or len(parts) == 0:
        errors.append(f"{path}: parts must be a non-empty list")
        return

    points_sum = 0
    for k, part in enumerate(parts):
        part_path = f"{path}.parts[{k}]"
        _validate_question_or_part(part, part_path, errors, all_ids, parent_vars)
        part_points = part.get("points")
        if isinstance(part_points, (int, float)) and part_points > 0:
            points_sum += part_points

    parent_points = q.get("points")
    if isinstance(parent_points, (int, float)) and parent_points > 0:
        if abs(points_sum - parent_points) > 1e-9:
            errors.append(
                f"{path}: Parts points sum ({points_sum}) does not match "
                f"question points ({parent_points})"
            )


def _validate_question_or_part(
    part: dict,
    path: str,
    errors: list[str],
    all_ids: set[str],
    parent_vars: dict,
) -> None:
    """Validate a sub-part (which looks like a question without parts)."""
    pid = part.get("id")
    if pid is None:
        errors.append(f"{path}: Missing required field: id")
    elif pid in all_ids:
        errors.append(f"{path}: Duplicate id: {pid!r} (ids must be unique across the template)")
    else:
        all_ids.add(pid)

    points = part.get("points")
    if points is None:
        errors.append(f"{path}: Missing required field: points")
    elif not isinstance(points, (int, float)) or points <= 0:
        errors.append(f"{path}: points must be a positive number")

    # figure on parts
    figure = part.get("figure")
    if figure is not None:
        if not isinstance(figure, str) or not figure.strip():
            errors.append(f"{path}: figure must be a non-blank string")

    # Merge any part-level variables with parent
    part_vars = part.get("variables", {})
    merged = dict(parent_vars)
    merged.update(part_vars)
    if part_vars:
        _validate_variables(part_vars, path, errors)

    if "answer_type" in part:
        _validate_answer_typed_question(part, path, errors, merged)
    elif "parts" in part:
        # Nested parts (unusual but not banned)
        _validate_parts(part, path, errors, all_ids, merged)
    else:
        errors.append(f"{path}: Part must have 'answer_type' or 'parts'")


def _validate_answer_typed_question(
    q: dict,
    path: str,
    errors: list[str],
    variables: dict,
) -> None:
    answer_type = q["answer_type"]
    if answer_type not in _VALID_ANSWER_TYPES:
        errors.append(
            f"{path}: Invalid answer_type: {answer_type!r}. "
            f"Must be one of: {sorted(_VALID_ANSWER_TYPES)}"
        )
        return

    # grading inference and validation
    grading = q.get("grading")
    if grading is None:
        grading = "auto" if answer_type in _AUTO_GRADABLE_TYPES else "manual_review"
    if grading == "auto" and answer_type not in _AUTO_GRADABLE_TYPES:
        errors.append(
            f"{path}: grading: auto is not valid for answer_type: {answer_type!r}"
        )

    # Validate prompt placeholders against variables
    prompt = q.get("prompt", "")
    if isinstance(prompt, str) and variables is not None:
        placeholders = set(_PLACEHOLDER_RE.findall(prompt))
        for ph in placeholders:
            if ph not in variables:
                errors.append(
                    f"{path}: Prompt placeholder '{{{{{ph}}}}}' references "
                    f"undeclared variable: {ph!r}"
                )

    # Type-specific answer validation
    if answer_type == "multiple_choice":
        _validate_mc_answer(q, path, errors, variables)
    elif answer_type == "numeric":
        _validate_numeric_answer(q, path, errors, variables)
    elif answer_type == "exact_match":
        _validate_exact_match_answer(q, path, errors, variables)
    elif answer_type in ("balanced_equation", "lewis_structure", "electron_config", "text_reasoning"):
        _validate_manual_review_answer(q, path, errors)


def _validate_mc_answer(
    q: dict, path: str, errors: list[str], variables: dict
) -> None:
    has_choices = "choices" in q
    has_distractors = "distractors" in q

    if has_choices:
        choices = q["choices"]
        correct = q.get("correct")
        if correct is not None and correct not in choices:
            errors.append(
                f"{path}: correct value {correct!r} not found in choices keys"
            )
    elif has_distractors:
        # Must have answer.expr
        answer = q.get("answer", {})
        if "expr" not in answer:
            errors.append(
                f"{path}: MC with distractors strategy requires answer.expr"
            )
        else:
            _check_expr_variables(answer["expr"], path + ".answer.expr", errors, variables)
    else:
        errors.append(
            f"{path}: multiple_choice requires either 'choices' or 'distractors'"
        )


def _validate_numeric_answer(
    q: dict, path: str, errors: list[str], variables: dict
) -> None:
    answer = q.get("answer", {})
    if not isinstance(answer, dict):
        errors.append(f"{path}: answer must be a mapping")
        return

    if "expr" not in answer:
        errors.append(f"{path}: numeric answer requires 'expr'")
    else:
        _check_expr_variables(answer["expr"], path + ".answer.expr", errors, variables)

    if "tolerance" not in answer:
        errors.append(f"{path}: numeric answer requires 'tolerance'")


def _validate_exact_match_answer(
    q: dict, path: str, errors: list[str], variables: dict
) -> None:
    answer = q.get("answer", {})
    if not isinstance(answer, dict):
        errors.append(f"{path}: answer must be a mapping")
        return

    has_accept = "accept" in answer
    has_expr = "expr" in answer

    if has_accept:
        accept = answer["accept"]
        if not isinstance(accept, list) or len(accept) == 0:
            errors.append(f"{path}: answer.accept must be a non-empty list")
    elif has_expr:
        if "accept_map" not in answer:
            errors.append(
                f"{path}: exact_match with expr requires accept_map"
            )
        _check_expr_variables(answer["expr"], path + ".answer.expr", errors, variables)
    else:
        errors.append(
            f"{path}: exact_match answer requires 'accept' or 'expr' + 'accept_map'"
        )


def _validate_manual_review_answer(
    q: dict, path: str, errors: list[str]
) -> None:
    answer = q.get("answer", {})
    if not isinstance(answer, dict):
        errors.append(f"{path}: answer must be a mapping")
        return

    if "reference" not in answer:
        errors.append(f"{path}: manual_review answer requires 'reference'")

    rubric = answer.get("rubric")
    if rubric is not None:
        if not isinstance(rubric, list):
            errors.append(f"{path}: answer.rubric must be a list of {{criterion, points}} entries")
        else:
            rubric_total = 0
            for entry in rubric:
                if isinstance(entry, dict) and "points" in entry:
                    rubric_total += entry["points"]

            q_points = q.get("points", 0)
            if isinstance(q_points, (int, float)) and abs(rubric_total - q_points) > 1e-9:
                errors.append(
                    f"{path}: Rubric points sum ({rubric_total}) does not match "
                    f"question/part points ({q_points})"
                )


# ---------------------------------------------------------------------------
# Variable validation
# ---------------------------------------------------------------------------


def _validate_variables(variables: dict, path: str, errors: list[str]) -> None:
    for name, spec in variables.items():
        vpath = f"{path}.variables.{name}"
        if not isinstance(spec, dict):
            errors.append(f"{vpath}: variable spec must be a mapping")
            continue

        vtype = spec.get("type")
        if vtype not in ("int", "float"):
            errors.append(f"{vpath}: variable type must be 'int' or 'float', got {vtype!r}")

        vmin = spec.get("min")
        vmax = spec.get("max")
        step = spec.get("step")

        if vmin is not None and vmax is not None:
            if vmin >= vmax:
                errors.append(f"{vpath}: min ({vmin}) must be less than max ({vmax})")

        if step is not None:
            if not isinstance(step, (int, float)) or step <= 0:
                errors.append(f"{vpath}: step must be a positive number, got {step!r}")

        for field in ("min", "max", "step"):
            if field not in spec:
                errors.append(f"{vpath}: missing required field: {field}")


def _check_expr_variables(
    expr_str: str, path: str, errors: list[str], variables: dict
) -> None:
    """Check that all names in an expression are declared variables or builtins."""
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError:
        errors.append(f"{path}: Invalid expression syntax: {expr_str!r}")
        return

    allowed_names = set(variables or {}) | _EXPR_BUILTINS
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            errors.append(
                f"{path}: Expression references undeclared variable: {node.id!r}"
            )


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

_EXPR_BUILTINS = frozenset({"min", "max", "abs", "round", "pi", "True", "False"})

_ALLOWED_AST_NODES = frozenset({
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    # Comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # Boolean operators
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
})


def _parse_and_validate_ast(expr_str: str) -> ast.Expression:
    """Parse an expression string and validate all AST nodes are allowed."""
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(
                f"Disallowed expression construct: {type(node).__name__}"
            )

    return tree


_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "True": True,
    "False": False,
}


def _eval_ast(tree: ast.Expression, variables: dict[str, Any]) -> Any:
    """Evaluate a validated AST with the given variables."""
    namespace = dict(_SAFE_GLOBALS)
    namespace.update(variables)
    code = compile(tree, "<expr>", "eval")
    return eval(code, namespace)
