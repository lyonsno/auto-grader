"""Turn bubble-readback results plus the per-student answer key into explicit MC decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def score_marked_mc_bubbles(
    marked_labels_by_question: Mapping[str, Any],
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
        question_surface = _normalize_question_surface(
            marked_labels_by_question.get(question_id, []),
            question_id=question_id,
        )
        marked_bubble_labels = question_surface["marked_bubble_labels"]
        ambiguous_bubble_labels = question_surface["ambiguous_bubble_labels"]
        illegible_bubble_labels = question_surface["illegible_bubble_labels"]

        bubble_to_choice = _bubble_to_choice_key(question_answer_key)
        marked_choice_keys = _map_marked_choice_keys(
            marked_bubble_labels,
            bubble_to_choice,
            question_id=question_id,
        )
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
            ambiguous_bubble_labels,
            illegible_bubble_labels,
            correct_bubble_label,
        )
        results[question_id] = {
            "question_id": question_id,
            "status": status,
            "marked_bubble_labels": marked_bubble_labels,
            "ambiguous_bubble_labels": ambiguous_bubble_labels,
            "illegible_bubble_labels": illegible_bubble_labels,
            "marked_choice_keys": marked_choice_keys,
            "correct_bubble_label": correct_bubble_label,
            "correct_choice_key": correct_choice_key,
            "is_correct": is_correct,
            "review_required": review_required,
        }

    return results


def _normalize_question_surface(
    raw_question_surface: Any,
    *,
    question_id: str,
) -> dict[str, list[str]]:
    if isinstance(raw_question_surface, Mapping):
        return {
            "marked_bubble_labels": _require_string_list(
                raw_question_surface.get("marked_bubble_labels", []),
                f"{question_id}.marked_bubble_labels",
            ),
            "ambiguous_bubble_labels": _require_string_list(
                raw_question_surface.get("ambiguous_bubble_labels", []),
                f"{question_id}.ambiguous_bubble_labels",
            ),
            "illegible_bubble_labels": _require_string_list(
                raw_question_surface.get("illegible_bubble_labels", []),
                f"{question_id}.illegible_bubble_labels",
            ),
        }
    return {
        "marked_bubble_labels": _require_string_list(
            raw_question_surface,
            f"{question_id}.marked_bubble_labels",
        ),
        "ambiguous_bubble_labels": [],
        "illegible_bubble_labels": [],
    }


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


def _map_marked_choice_keys(
    marked_bubble_labels: list[str],
    bubble_to_choice: Mapping[str, str],
    *,
    question_id: str,
) -> list[str]:
    marked_choice_keys: list[str] = []
    for bubble_label in marked_bubble_labels:
        choice_key = bubble_to_choice.get(bubble_label)
        if choice_key is None:
            raise ValueError(
                f"Unknown bubble label {bubble_label!r} for question {question_id!r}"
            )
        marked_choice_keys.append(choice_key)
    return marked_choice_keys


def _score_status(
    marked_bubble_labels: list[str],
    ambiguous_bubble_labels: list[str],
    illegible_bubble_labels: list[str],
    correct_bubble_label: str,
) -> tuple[str, bool, bool]:
    if illegible_bubble_labels:
        return ("illegible_mark", False, True)
    if ambiguous_bubble_labels:
        return ("ambiguous_mark", False, True)
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


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list of non-empty strings")
    normalized: list[str] = []
    for item in value:
        normalized.append(_require_string(item, label))
    return normalized
