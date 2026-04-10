"""Turn bubble-readback results plus the per-student answer key into explicit MC decisions.

This layer also performs narrow question-level dominance arbitration so one clearly
stronger deliberate fill can suppress much weaker secondary traces without
flattening real co-equal multiple marks into fake confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def score_marked_mc_bubbles(
    marked_labels_by_question: Mapping[str, Any],
    answer_key: Mapping[str, Mapping[str, Any]],
    bubble_evidence_by_question: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(marked_labels_by_question, Mapping):
        raise TypeError("marked_labels_by_question must be a mapping")
    if not isinstance(answer_key, Mapping):
        raise TypeError("answer_key must be a mapping")
    if bubble_evidence_by_question is not None and not isinstance(bubble_evidence_by_question, Mapping):
        raise TypeError("bubble_evidence_by_question must be a mapping")

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
        question_evidence = _normalize_question_evidence(
            None if bubble_evidence_by_question is None else bubble_evidence_by_question.get(question_id),
            question_id=question_id,
        )

        bubble_to_choice = _bubble_to_choice_key(question_answer_key)
        correct_bubble_label = _require_string(
            question_answer_key.get("correct_bubble_label"),
            "answer_key.correct_bubble_label",
        )
        correct_choice_key = _require_string(
            question_answer_key.get("correct_choice_key"),
            "answer_key.correct_choice_key",
        )

        (
            status,
            is_correct,
            review_required,
            resolved_bubble_labels,
            ignored_incidental_bubble_labels,
        ) = _score_status(
            marked_bubble_labels,
            ambiguous_bubble_labels,
            illegible_bubble_labels,
            correct_bubble_label,
            question_evidence,
        )
        marked_choice_keys = _map_marked_choice_keys(
            resolved_bubble_labels,
            bubble_to_choice,
            question_id=question_id,
        )
        results[question_id] = {
            "question_id": question_id,
            "status": status,
            "marked_bubble_labels": marked_bubble_labels,
            "ambiguous_bubble_labels": ambiguous_bubble_labels,
            "illegible_bubble_labels": illegible_bubble_labels,
            "resolved_bubble_labels": resolved_bubble_labels,
            "ignored_incidental_bubble_labels": ignored_incidental_bubble_labels,
            "marked_choice_keys": marked_choice_keys,
            "correct_bubble_label": correct_bubble_label,
            "correct_choice_key": correct_choice_key,
            "is_correct": is_correct,
            "review_required": review_required,
        }

    return results


_DOMINANT_FILL_INTENT_SCORE_MIN = 0.65
_DOMINANCE_RATIO_THRESHOLD = 1.6
_DOMINANCE_MARGIN_THRESHOLD = 0.25
_SECONDARY_TRACE_SCORE_MAX = 0.50


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


def _normalize_question_evidence(
    raw_question_evidence: Any,
    *,
    question_id: str,
) -> dict[str, dict[str, float]]:
    if raw_question_evidence is None:
        return {}
    if not isinstance(raw_question_evidence, Mapping):
        raise TypeError(f"{question_id}.bubble_evidence must be a mapping")

    normalized: dict[str, dict[str, float]] = {}
    for bubble_label, evidence in raw_question_evidence.items():
        evidence_mapping = _require_mapping(evidence, f"{question_id}.{bubble_label}.evidence")
        normalized[_require_string(bubble_label, f"{question_id}.bubble_evidence.label")] = {
            "fill_intent_score": float(
                _require_number(
                    evidence_mapping.get("fill_intent_score"),
                    f"{question_id}.{bubble_label}.fill_intent_score",
                )
            ),
            "center_dark_fraction": float(
                _require_number(
                    evidence_mapping.get("center_dark_fraction"),
                    f"{question_id}.{bubble_label}.center_dark_fraction",
                )
            ),
            "center_dark_bbox_fill_ratio": float(
                _require_number(
                    evidence_mapping.get("center_dark_bbox_fill_ratio"),
                    f"{question_id}.{bubble_label}.center_dark_bbox_fill_ratio",
                )
            ),
        }
    return normalized


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
    question_evidence: Mapping[str, Mapping[str, float]],
) -> tuple[str, bool, bool, list[str], list[str]]:
    resolved_marked_bubble_labels, resolved_ambiguous_bubble_labels, ignored_incidental = (
        _apply_dominance_arbitration(
            marked_bubble_labels,
            ambiguous_bubble_labels,
            question_evidence,
        )
    )
    if illegible_bubble_labels:
        return ("illegible_mark", False, True, resolved_marked_bubble_labels, ignored_incidental)
    if resolved_ambiguous_bubble_labels:
        return ("ambiguous_mark", False, True, resolved_marked_bubble_labels, ignored_incidental)
    if len(resolved_marked_bubble_labels) == 0:
        return ("blank", False, False, resolved_marked_bubble_labels, ignored_incidental)
    if len(resolved_marked_bubble_labels) > 1:
        return (
            "multiple_marked",
            False,
            True,
            resolved_marked_bubble_labels,
            ignored_incidental,
        )
    if resolved_marked_bubble_labels[0] == correct_bubble_label:
        return ("correct", True, False, resolved_marked_bubble_labels, ignored_incidental)
    return ("incorrect", False, False, resolved_marked_bubble_labels, ignored_incidental)


def _apply_dominance_arbitration(
    marked_bubble_labels: list[str],
    ambiguous_bubble_labels: list[str],
    question_evidence: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], list[str], list[str]]:
    candidates = marked_bubble_labels + ambiguous_bubble_labels
    if len(candidates) < 2:
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []

    scores = {
        bubble_label: float(question_evidence.get(bubble_label, {}).get("fill_intent_score", 0.0))
        for bubble_label in candidates
    }
    ranked = sorted(candidates, key=lambda bubble_label: scores[bubble_label], reverse=True)
    dominant_label = ranked[0]
    dominant_score = scores[dominant_label]
    runner_up_label = ranked[1]
    runner_up_score = scores[runner_up_label]

    if dominant_label not in marked_bubble_labels:
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []
    if dominant_score < _DOMINANT_FILL_INTENT_SCORE_MIN:
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []
    if runner_up_score > _SECONDARY_TRACE_SCORE_MAX:
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []
    if runner_up_score <= 0.0:
        return [dominant_label], [], [bubble_label for bubble_label in candidates if bubble_label != dominant_label]
    if dominant_score < (runner_up_score * _DOMINANCE_RATIO_THRESHOLD):
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []
    if (dominant_score - runner_up_score) < _DOMINANCE_MARGIN_THRESHOLD:
        return list(marked_bubble_labels), list(ambiguous_bubble_labels), []

    ignored_incidental = [bubble_label for bubble_label in candidates if bubble_label != dominant_label]
    return [dominant_label], [], ignored_incidental


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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return value
