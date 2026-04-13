"""Read the current-final MC truth surface from persisted DB state."""

from __future__ import annotations

from typing import Any


def read_current_final_mc_results_from_db(
    *,
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    if isinstance(exam_instance_id, bool) or not isinstance(exam_instance_id, int):
        raise TypeError("exam_instance_id must be an integer")

    session_row = connection.execute(
        "SELECT id, session_ordinal "
        "FROM mc_scan_sessions "
        "WHERE exam_instance_id = %s "
        "ORDER BY session_ordinal DESC "
        "LIMIT 1",
        (exam_instance_id,),
    ).fetchone()
    if session_row is None:
        raise LookupError(
            f"No MC scan session found for exam_instance_id {exam_instance_id}"
        )

    page_rows = connection.execute(
        "SELECT id, scan_id, checksum, status, failure_reason, page_number, "
        "       fallback_page_code, divergence_detected "
        "FROM mc_scan_pages "
        "WHERE mc_scan_session_id = %s "
        "ORDER BY page_number NULLS LAST, scan_id",
        (session_row["id"],),
    ).fetchall()

    outcome_rows = connection.execute(
        "SELECT p.id AS page_id, p.scan_id, p.page_number, "
        "       q.question_id, q.status AS machine_status, q.is_correct AS machine_is_correct, "
        "       q.review_required AS machine_review_required, "
        "       q.marked_bubble_labels, q.resolved_bubble_labels, "
        "       q.correct_bubble_label, q.correct_choice_key, "
        "       r.id AS resolution_id, r.original_status, r.resolved_bubble_label, "
        "       r.final_status, r.final_is_correct "
        "FROM mc_scan_pages p "
        "JOIN mc_question_outcomes q ON q.mc_scan_page_id = p.id "
        "LEFT JOIN mc_review_resolutions r ON r.mc_question_outcome_id = q.id "
        "WHERE p.mc_scan_session_id = %s "
        "ORDER BY p.page_number NULLS LAST, p.scan_id, q.question_id",
        (session_row["id"],),
    ).fetchall()

    scan_pages: list[dict[str, Any]] = []
    scan_pages_by_id: dict[int, dict[str, Any]] = {}
    for row in page_rows:
        page = {
            "scan_id": _require_string(row["scan_id"], "mc_scan_pages.scan_id"),
            "checksum": _require_string(row["checksum"], "mc_scan_pages.checksum"),
            "status": _require_string(row["status"], "mc_scan_pages.status"),
            "failure_reason": row["failure_reason"],
            "page_number": row["page_number"],
            "fallback_page_code": row["fallback_page_code"],
            "divergence_detected": _require_bool(
                row["divergence_detected"],
                "mc_scan_pages.divergence_detected",
            ),
            "question_ids": [],
        }
        scan_pages.append(page)
        scan_pages_by_id[row["id"]] = page

    question_results: dict[str, dict[str, Any]] = {}
    review_required_question_ids: list[str] = []
    status_counts: dict[str, int] = {"correct": 0, "incorrect": 0, "blank": 0}

    for row in outcome_rows:
        question_id = _require_string(row["question_id"], "mc_question_outcomes.question_id")
        machine_status = _require_string(row["machine_status"], "mc_question_outcomes.status")
        machine_is_correct = _require_bool(
            row["machine_is_correct"],
            "mc_question_outcomes.is_correct",
        )
        machine_review_required = _require_bool(
            row["machine_review_required"],
            "mc_question_outcomes.review_required",
        )
        marked_bubble_labels = _require_string_list(
            row["marked_bubble_labels"],
            "mc_question_outcomes.marked_bubble_labels",
        )
        machine_resolved_bubble_labels = _require_string_list(
            row["resolved_bubble_labels"],
            "mc_question_outcomes.resolved_bubble_labels",
        )

        resolution = None
        if row["resolution_id"] is not None:
            resolution = {
                "id": row["resolution_id"],
                "original_status": _require_string(
                    row["original_status"],
                    "mc_review_resolutions.original_status",
                ),
                "resolved_bubble_label": row["resolved_bubble_label"],
                "final_status": _require_string(
                    row["final_status"],
                    "mc_review_resolutions.final_status",
                ),
                "final_is_correct": _require_bool(
                    row["final_is_correct"],
                    "mc_review_resolutions.final_is_correct",
                ),
            }
            status = resolution["final_status"]
            is_correct = resolution["final_is_correct"]
            review_required = False
            source = "review_resolution"
            final_resolved_bubble_labels = (
                []
                if resolution["resolved_bubble_label"] is None
                else [resolution["resolved_bubble_label"]]
            )
        else:
            status = machine_status
            is_correct = machine_is_correct
            review_required = machine_review_required
            source = "machine"
            final_resolved_bubble_labels = list(machine_resolved_bubble_labels)

        if review_required:
            review_required_question_ids.append(question_id)

        if status in status_counts:
            status_counts[status] += 1

        page = scan_pages_by_id[row["page_id"]]
        page["question_ids"].append(question_id)
        question_results[question_id] = {
            "question_id": question_id,
            "scan_id": _require_string(row["scan_id"], "mc_scan_pages.scan_id"),
            "page_number": row["page_number"],
            "status": status,
            "is_correct": is_correct,
            "review_required": review_required,
            "source": source,
            "machine_status": machine_status,
            "machine_is_correct": machine_is_correct,
            "machine_review_required": machine_review_required,
            "marked_bubble_labels": marked_bubble_labels,
            "machine_resolved_bubble_labels": machine_resolved_bubble_labels,
            "final_resolved_bubble_labels": final_resolved_bubble_labels,
            "correct_bubble_label": _require_string(
                row["correct_bubble_label"],
                "mc_question_outcomes.correct_bubble_label",
            ),
            "correct_choice_key": _require_string(
                row["correct_choice_key"],
                "mc_question_outcomes.correct_choice_key",
            ),
            "resolution": resolution,
        }

    return {
        "exam_instance_id": exam_instance_id,
        "mc_scan_session_id": session_row["id"],
        "session_ordinal": session_row["session_ordinal"],
        "scan_pages": scan_pages,
        "question_results": question_results,
        "review_required_question_ids": sorted(review_required_question_ids),
        "summary": {
            "matched": sum(1 for page in scan_pages if page["status"] == "matched"),
            "unmatched": sum(1 for page in scan_pages if page["status"] == "unmatched"),
            "ambiguous": sum(1 for page in scan_pages if page["status"] == "ambiguous"),
            "unresolved_review_required": len(review_required_question_ids),
            "correct": status_counts["correct"],
            "incorrect": status_counts["incorrect"],
            "blank": status_counts["blank"],
        },
    }


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
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
