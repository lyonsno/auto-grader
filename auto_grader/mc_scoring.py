"""Turn bubble-readback results plus the per-student answer key into explicit MC decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def score_marked_mc_bubbles(
    marked_labels_by_question: Mapping[str, list[str]],
    answer_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(marked_labels_by_question, Mapping):
        raise TypeError("marked_labels_by_question must be a mapping")
    if not isinstance(answer_key, Mapping):
        raise TypeError("answer_key must be a mapping")

    results: dict[str, dict[str, Any]] = {}
    for question_id, question_answer_key in answer_key.items():
        if not isinstance(question_answer_key, Mapping):
            raise TypeError("answer_key entries must be mappings")
        marked_bubble_labels = list(marked_labels_by_question.get(question_id, []))
        bubble_to_choice = _bubble_to_choice_key(question_answer_key)
        marked_choice_keys = [bubble_to_choice[label] for label in marked_bubble_labels]
        correct_bubble_label = _require_string(
            question_answer_key.get("correct_bubble_label"),
            "answer_key.correct_bubble_label",
        )
        correct_choice_key = _require_string(
            question_answer_key.get("correct_choice_key"),
            "answer_key.correct_choice_key",
        )

        status, is_correct, review_required = _score_status(
            marked_bubble_labels,
            correct_bubble_label,
        )
        results[question_id] = {
            "question_id": question_id,
            "status": status,
            "marked_bubble_labels": marked_bubble_labels,
            "marked_choice_keys": marked_choice_keys,
            "correct_bubble_label": correct_bubble_label,
            "correct_choice_key": correct_choice_key,
            "is_correct": is_correct,
            "review_required": review_required,
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
        bubble_label = _require_string(raw_choice.get("bubble_label"), "answer_key.choice.bubble_label")
        choice_key = _require_string(raw_choice.get("choice_key"), "answer_key.choice.choice_key")
        mapping[bubble_label] = choice_key
    return mapping


def _score_status(
    marked_bubble_labels: list[str],
    correct_bubble_label: str,
) -> tuple[str, bool, bool]:
    if len(marked_bubble_labels) == 0:
        return ("blank", False, False)
    if len(marked_bubble_labels) > 1:
        return ("multiple_marked", False, True)
    if marked_bubble_labels[0] == correct_bubble_label:
        return ("correct", True, False)
    return ("incorrect", False, False)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
