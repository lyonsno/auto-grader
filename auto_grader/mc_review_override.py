"""Apply human review overrides to machine-scored MC questions.

The scoring pipeline flags questions as review_required when it encounters
multiple_marked, ambiguous_mark, or illegible_mark conditions. This module
lets a reviewer supply override decisions on any scored question and produces
corrected results with provenance tracking.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def apply_mc_review_overrides(
    scored_questions: Mapping[str, Mapping[str, Any]],
    answer_key: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(scored_questions, Mapping):
        raise TypeError("scored_questions must be a mapping")
    if not isinstance(answer_key, Mapping):
        raise TypeError("answer_key must be a mapping")
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")

    for question_id in overrides:
        if question_id not in scored_questions:
            raise ValueError(f"Unknown question {question_id!r} in overrides")

    results: dict[str, dict[str, Any]] = {}
    for question_id, scored in scored_questions.items():
        if question_id not in overrides:
            results[question_id] = dict(scored)
            continue

        override = overrides[question_id]
        resolved_bubble_label = override.get("resolved_bubble_label")

        question_answer_key = answer_key.get(question_id)
        if question_answer_key is None:
            raise ValueError(f"No answer key for question {question_id!r}")

        bubble_to_choice = _bubble_to_choice_key(question_answer_key)
        correct_bubble_label = _require_string(
            question_answer_key.get("correct_bubble_label"),
            "answer_key.correct_bubble_label",
        )

        if resolved_bubble_label is None:
            results[question_id] = {
                **scored,
                "status": "blank",
                "is_correct": False,
                "review_required": False,
                "resolved_bubble_labels": [],
                "marked_choice_keys": [],
                "override": {
                    "original_status": scored["status"],
                    "resolved_bubble_label": None,
                },
            }
            continue

        if not isinstance(resolved_bubble_label, str) or resolved_bubble_label == "":
            raise TypeError("override.resolved_bubble_label must be a non-empty string or None")

        choice_key = bubble_to_choice.get(resolved_bubble_label)
        if choice_key is None:
            raise ValueError(
                f"Unknown bubble label {resolved_bubble_label!r} for question {question_id!r}"
            )

        is_correct = resolved_bubble_label == correct_bubble_label
        status = "correct" if is_correct else "incorrect"

        results[question_id] = {
            **scored,
            "status": status,
            "is_correct": is_correct,
            "review_required": False,
            "resolved_bubble_labels": [resolved_bubble_label],
            "marked_choice_keys": [choice_key],
            "override": {
                "original_status": scored["status"],
                "resolved_bubble_label": resolved_bubble_label,
            },
        }

    return results


def _bubble_to_choice_key(question_answer_key: Mapping[str, Any]) -> dict[str, str]:
    raw_choices = question_answer_key.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise ValueError("answer_key.choices must be a non-empty list")

    mapping: dict[str, str] = {}
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, Mapping):
            raise TypeError("answer_key.choices entries must be mappings")
        bubble_label = _require_string(raw_choice.get("bubble_label"), "choice.bubble_label")
        choice_key = _require_string(raw_choice.get("choice_key"), "choice.choice_key")
        mapping[bubble_label] = choice_key
    return mapping


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
