"""Reconstruct and generate the Quiz #5 short-answer family from legacy PDFs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import re
from typing import Any

import fitz

from auto_grader.template_schema import evaluate_expr, validate_template


_SCIENTIFIC_RE = re.compile(r"(?P<base>\d+(?:\.\d+)?)×10(?P<exp>[+-]?\d+)")
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_SUPERSCRIPT_DIGITS = str.maketrans({
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
})


def reconstruct_short_answer_quiz_family(pdf_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Reconstruct the canonical Quiz #5 short-answer family from legacy PDFs.

    This first slice is intentionally narrow and honest: it targets the real
    `Quiz #5 A/B` family that surfaced as the forcing case for short-answer
    authoring. The output shape is future-generation-oriented: one canonical
    template plus per-variant substitutions.
    """

    resolved_paths = [Path(path) for path in pdf_paths]
    if not resolved_paths:
        raise ValueError("At least one legacy quiz PDF is required")

    variants: dict[str, dict[str, Any]] = {}
    for path in resolved_paths:
        variant_id = _infer_variant_id(path)
        variants[variant_id] = {"source_path": str(path), "variables": _extract_quiz5_variables(path)}

    template = _build_quiz5_template()
    errors = validate_template(template)
    if errors:
        raise ValueError(f"Reconstructed template is invalid: {'; '.join(errors)}")

    return {
        "slug": "chm142-quiz-5",
        "title": "CHM 142 Quiz #5",
        "template": template,
        "variants": {key: variants[key] for key in sorted(variants)},
    }


def build_generated_short_answer_variant(
    family: dict[str, Any],
    *,
    variant_id: str,
) -> dict[str, Any]:
    """Build a reviewable sibling variant from the reconstructed family."""
    if variant_id != "C":
        raise ValueError("Only sibling variant 'C' is supported in this first generation slice")

    variables = _generate_variant_c_variables(family["variants"])
    template = family["template"]
    return {
        "variant_id": variant_id,
        "strategy": "midpoint_numeric_plus_curated_acid_base_examples_v1",
        "source_variant_ids": sorted(family["variants"]),
        "variables": variables,
        "question_prompts": _render_question_prompts(template, variables),
        "answer_preview": _build_answer_preview(template, variables),
    }


def write_reconstructed_short_answer_quiz_family(
    *,
    pdf_paths: Iterable[str | Path],
    output_dir: str | Path,
    generate_variant_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Write the reconstructed short-answer family to disk as JSON."""
    family = reconstruct_short_answer_quiz_family(pdf_paths)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    family_json_path = output_path / "short-answer-quiz-family.json"
    family_json_path.write_text(
        json.dumps(family, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = {
        "family_json_path": str(family_json_path),
        "slug": family["slug"],
        "variant_ids": sorted(family["variants"]),
    }

    generated_variant_ids = list(generate_variant_ids)
    if generated_variant_ids:
        generated_paths: dict[str, str] = {}
        for variant_id in generated_variant_ids:
            generated = build_generated_short_answer_variant(
                family,
                variant_id=variant_id,
            )
            generated_path = output_path / f"short-answer-quiz-variant-{variant_id}.json"
            generated_path.write_text(
                json.dumps(generated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            generated_paths[variant_id] = str(generated_path)

        result["generated_variant_json_paths"] = generated_paths
        if len(generated_paths) == 1:
            result["generated_variant_json_path"] = next(iter(generated_paths.values()))

    return result


def _infer_variant_id(path: Path) -> str:
    stem = path.stem
    match = re.search(r"\b([A-Z])(?:_answers)?$", stem)
    if not match:
        raise ValueError(f"Could not infer variant code from filename: {path.name!r}")
    return match.group(1)


def _extract_quiz5_variables(path: Path) -> dict[str, Any]:
    text = _normalize_whitespace(_extract_pdf_text(path))

    return {
        "bronsted_base": _capture(
            r"a\. Write a net ionic equation to show how (?P<value>.+?) behaves as a Bronsted base in water",
            text,
        ),
        "bronsted_acid": _capture(
            r"b\. Write a net ionic equation to show how (?P<value>.+?) behaves as a Bronsted acid in water",
            text,
        ),
        "acid_molarity": _parse_float(
            _capture(
                r"2\. What is the pH of an aqueous solution of (?P<value>\d+(?:\.\d+)?(?:×10-?\d+)?) M ",
                text,
            )
        ),
        "acid_species": _capture(
            r"2\. What is the pH of an aqueous solution of \d+(?:\.\d+)?(?:×10-?\d+)? M (?P<value>.+?)\?",
            text,
        ),
        "base_molarity": _parse_float(
            _capture(
                r"3\. What is the pH of a (?P<value>\d+(?:\.\d+)?) M aqueous solution of sodium hydroxide",
                text,
            )
        ),
        "target_ph": _parse_float(
            _capture(
                r"4\. What concentration of nitric acid is needed to make an aqueous solution with a pH of (?P<value>\d+(?:\.\d+)?)",
                text,
            )
        ),
        "kc_q5": _parse_float(
            _capture(
                r"5\. The equilibrium constant, Kc, for the following reaction is (?P<value>\d+(?:\.\d+)?) at 1200 K",
                text,
            )
        ),
        "kc_q6": _parse_float(
            _capture(
                r"6\. The equilibrium constant, Kc, for the following reaction is (?P<value>\d+(?:\.\d+)?) at 548 K",
                text,
            )
        ),
    }


def _extract_pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)

    # Batch-CLI path; synchronous extraction is acceptable in this first slice.
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not extract expected Quiz #5 field with pattern: {pattern}")
    return match.group("value").strip()


def _parse_float(raw: str) -> float:
    sci_match = _SCIENTIFIC_RE.fullmatch(raw)
    if sci_match:
        return float(f"{sci_match.group('base')}e{sci_match.group('exp')}")
    return float(raw)


def _generate_variant_c_variables(observed_variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a_vars = observed_variants["A"]["variables"]
    b_vars = observed_variants["B"]["variables"]
    return {
        "bronsted_base": "methylamine CH3NH2",
        "bronsted_acid": "acetic acid CH3COOH",
        "acid_species": "hydrochloric acid",
        "acid_molarity": round((a_vars["acid_molarity"] + b_vars["acid_molarity"]) / 2, 5),
        "base_molarity": round((a_vars["base_molarity"] + b_vars["base_molarity"]) / 2, 4),
        "target_ph": round((a_vars["target_ph"] + b_vars["target_ph"]) / 2, 2),
        "kc_q5": round((a_vars["kc_q5"] + b_vars["kc_q5"]) / 2, 4),
        "kc_q6": round((a_vars["kc_q6"] + b_vars["kc_q6"]) / 2, 4),
    }


def _render_question_prompts(template: dict[str, Any], variables: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for question in template["sections"][0]["questions"]:
        if "parts" in question:
            for part in question["parts"]:
                entries.append(
                    {
                        "id": part["id"],
                        "prompt": _review_prompt_for_entry(part["id"], part["prompt"], variables),
                        "response_box_label": part["response_box_label"],
                    }
                )
        else:
            entries.append(
                {
                    "id": question["id"],
                    "prompt": _review_prompt_for_entry(question["id"], question["prompt"], variables),
                    "response_box_label": question["response_box_label"],
                }
            )
    return entries


def _review_prompt_for_entry(entry_id: str, base_prompt: str, variables: dict[str, Any]) -> str:
    # The current template schema only supports numeric variables, so the
    # acid/base identity for q1 must still be woven in at review-packet time.
    if entry_id == "q1a":
        return (
            f"Write a net ionic equation to show how {variables['bronsted_base']} "
            "behaves as a Bronsted base in water."
        )
    if entry_id == "q1b":
        return (
            f"Write a net ionic equation to show how {variables['bronsted_acid']} "
            "behaves as a Bronsted acid in water."
        )
    return _render_text(base_prompt, variables)


def _render_text(text: str, variables: dict[str, Any]) -> str:
    return _PLACEHOLDER_RE.sub(
        lambda match: _format_variable(match.group(1), variables[match.group(1)]),
        text,
    )


def _format_variable(name: str, value: Any) -> str:
    if name == "acid_molarity" and isinstance(value, float):
        return _format_scientific_notation(value, sig_figs=3)
    if isinstance(value, float):
        if value >= 0.01:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return str(value)


def _format_scientific_notation(value: float, *, sig_figs: int) -> str:
    mantissa, exponent = f"{value:.{sig_figs - 1}e}".split("e")
    exponent_text = str(int(exponent)).translate(_SUPERSCRIPT_DIGITS)
    return f"{mantissa}×10{exponent_text}"


def _build_answer_preview(template: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for question in template["sections"][0]["questions"]:
        if "parts" in question:
            for part in question["parts"]:
                preview[part["id"]] = _answer_preview_for_entry(part, variables)
        else:
            preview[question["id"]] = _answer_preview_for_entry(question, variables)
    return preview


def _answer_preview_for_entry(entry: dict[str, Any], variables: dict[str, Any]) -> Any:
    answer = entry["answer"]
    if "expr" in answer:
        return evaluate_expr(answer["expr"], variables)
    return answer.get("reference")


def _build_quiz5_template() -> dict[str, Any]:
    return {
        "slug": "chm142-quiz-5",
        "title": "CHM 142 Quiz #5",
        "course": "CHM 142",
        "sections": [
            {
                "id": "short-answer",
                "title": "Short Answer",
                "questions": [
                    {
                        "id": "q1",
                        "points": 2,
                        "prompt": "Net ionic equations",
                        "parts": [
                            {
                                "id": "q1a",
                                "points": 1,
                                "answer_type": "text_reasoning",
                                "grading": "manual_review",
                                "response_box_label": "1a.",
                                "prompt": "Write a net ionic equation for the assigned Bronsted base in water.",
                                "answer": {
                                    "reference": "Net ionic equation showing proton acceptance in water.",
                                    "rubric": [{"criterion": "Correct conjugate acid/base and charge balance", "points": 1}],
                                },
                            },
                            {
                                "id": "q1b",
                                "points": 1,
                                "answer_type": "text_reasoning",
                                "grading": "manual_review",
                                "response_box_label": "1b.",
                                "prompt": "Write a net ionic equation for the assigned Bronsted acid in water.",
                                "answer": {
                                    "reference": "Net ionic equation showing proton donation in water.",
                                    "rubric": [{"criterion": "Correct conjugate base/acid and charge balance", "points": 1}],
                                },
                            },
                        ],
                    },
                    {
                        "id": "q2",
                        "points": 1,
                        "answer_type": "numeric",
                        "response_box_label": "2.",
                        "variables": {
                            "acid_molarity": {"type": "float", "min": 0.001, "max": 0.1, "step": 0.001},
                        },
                        "prompt": "What is the pH of an aqueous solution of {{acid_molarity}} M strong acid?",
                        "answer": {
                            "expr": "-log10(acid_molarity)",
                            "tolerance": {"type": "absolute", "value": 0.01},
                        },
                    },
                    {
                        "id": "q3",
                        "points": 1,
                        "answer_type": "numeric",
                        "response_box_label": "3.",
                        "variables": {
                            "base_molarity": {"type": "float", "min": 0.001, "max": 0.1, "step": 0.001},
                        },
                        "prompt": (
                            "What is the pH of a {{base_molarity}} M aqueous solution of sodium hydroxide?"
                        ),
                        "answer": {
                            "expr": "14 - (-log10(base_molarity))",
                            "tolerance": {"type": "absolute", "value": 0.01},
                        },
                    },
                    {
                        "id": "q4",
                        "points": 1,
                        "answer_type": "numeric",
                        "response_box_label": "4.",
                        "variables": {
                            "target_ph": {"type": "float", "min": 0.0, "max": 14.0, "step": 0.1},
                        },
                        "prompt": (
                            "What concentration of nitric acid is needed to make an aqueous solution with "
                            "a pH of {{target_ph}}?"
                        ),
                        "answer": {
                            "expr": "10 ** (-target_ph)",
                            "tolerance": {"type": "relative", "value": 0.01},
                        },
                    },
                    {
                        "id": "q5",
                        "points": 1,
                        "answer_type": "numeric",
                        "response_box_label": "5.",
                        "variables": {
                            "kc_q5": {"type": "float", "min": 0.0001, "max": 1.0, "step": 0.0001},
                        },
                        "prompt": (
                            "For 2SO3(g) ⇄ 2SO2(g) + O2(g), the equilibrium constant Kc is {{kc_q5}} at "
                            "1200 K. If a 1.00 L equilibrium mixture contains 0.200 mol SO3 and 0.387 mol "
                            "SO2, calculate the equilibrium concentration of O2."
                        ),
                        "answer": {
                            "expr": "kc_q5 * (0.200 ** 2) / (0.387 ** 2)",
                            "tolerance": {"type": "relative", "value": 0.01},
                        },
                    },
                    {
                        "id": "q6",
                        "points": 3,
                        "variables": {
                            "kc_q6": {"type": "float", "min": 0.0001, "max": 1.0, "step": 0.0001},
                        },
                        "prompt": (
                            "For CH4(g) + CCl4(g) ⇄ 2CH2Cl2(g), the equilibrium constant Kc is {{kc_q6}} "
                            "at 548 K. Calculate the equilibrium concentrations when 0.375 mol of CH4 and "
                            "0.375 mol of CCl4 are introduced into a 1.00 L vessel."
                        ),
                        "parts": [
                            {
                                "id": "q6-ch4",
                                "points": 1,
                                "answer_type": "numeric",
                                "response_box_label": "6.[CH4]=",
                                "prompt": "Equilibrium concentration of CH4",
                                "answer": {
                                    "expr": "0.375 - (0.375 * sqrt(kc_q6) / (2 + sqrt(kc_q6)))",
                                    "tolerance": {"type": "relative", "value": 0.01},
                                },
                            },
                            {
                                "id": "q6-ccl4",
                                "points": 1,
                                "answer_type": "numeric",
                                "response_box_label": "6. [CCl4]=",
                                "prompt": "Equilibrium concentration of CCl4",
                                "answer": {
                                    "expr": "0.375 - (0.375 * sqrt(kc_q6) / (2 + sqrt(kc_q6)))",
                                    "tolerance": {"type": "relative", "value": 0.01},
                                },
                            },
                            {
                                "id": "q6-ch2cl2",
                                "points": 1,
                                "answer_type": "numeric",
                                "response_box_label": "6. [CH2Cl2]=",
                                "prompt": "Equilibrium concentration of CH2Cl2",
                                "answer": {
                                    "expr": "2 * (0.375 * sqrt(kc_q6) / (2 + sqrt(kc_q6)))",
                                    "tolerance": {"type": "relative", "value": 0.01},
                                },
                            },
                        ],
                    },
                ],
            }
        ],
    }
