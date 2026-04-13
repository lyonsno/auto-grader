"""Compact demo/export surface over the authoritative DB-backed MC truth."""

from __future__ import annotations

from typing import Any

from auto_grader.mc_results_db import read_current_final_mc_results_from_db


def build_mc_results_demo_export(
    *,
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    current = read_current_final_mc_results_from_db(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    question_results = _require_mapping(
        current.get("question_results"),
        "current.question_results",
    )

    questions: list[dict[str, Any]] = []
    resolved_by_review = 0
    for question_id in sorted(question_results):
        question = _require_mapping(
            question_results[question_id],
            f"question_results[{question_id!r}]",
        )
        source = _require_string(question.get("source"), f"{question_id}.source")
        if source == "review_resolution":
            resolved_by_review += 1

        final_resolved_bubble_labels = _require_string_list(
            question.get("final_resolved_bubble_labels"),
            f"{question_id}.final_resolved_bubble_labels",
        )
        row = {
            "question_id": question_id,
            "page_number": question.get("page_number"),
            "scan_id": _require_string(question.get("scan_id"), f"{question_id}.scan_id"),
            "status": _require_string(question.get("status"), f"{question_id}.status"),
            "is_correct": _require_bool(question.get("is_correct"), f"{question_id}.is_correct"),
            "review_required": _require_bool(
                question.get("review_required"),
                f"{question_id}.review_required",
            ),
            "source": source,
            "machine_status": _require_string(
                question.get("machine_status"),
                f"{question_id}.machine_status",
            ),
            "resolved_bubble_label": (
                None if not final_resolved_bubble_labels else final_resolved_bubble_labels[0]
            ),
            "final_resolved_bubble_labels": final_resolved_bubble_labels,
        }
        questions.append(row)

    review_queue = [
        {
            "question_id": row["question_id"],
            "page_number": row["page_number"],
            "scan_id": row["scan_id"],
            "machine_status": row["machine_status"],
            "marked_bubble_labels": _require_string_list(
                question_results[row["question_id"]].get("marked_bubble_labels"),
                f"{row['question_id']}.marked_bubble_labels",
            ),
        }
        for row in questions
        if row["review_required"]
    ]

    summary = dict(_require_mapping(current.get("summary"), "current.summary"))
    summary["question_count"] = len(questions)
    summary["resolved_by_review"] = resolved_by_review

    return {
        "exam_instance_id": current["exam_instance_id"],
        "mc_scan_session_id": current["mc_scan_session_id"],
        "session_ordinal": current["session_ordinal"],
        "summary": summary,
        "questions": questions,
        "review_queue": review_queue,
    }


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        result.append(_require_string(item, label))
    return result
