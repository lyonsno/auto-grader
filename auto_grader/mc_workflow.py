"""Professor-facing MC workflow entrypoint.

Composes the landed DB-backed MC primitives into a usable surface for the
full scan -> grade -> review -> export loop without requiring the professor
to know about internal module boundaries or prepare complex JSON by hand.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from auto_grader.mc_results_db import read_current_final_mc_results_from_db
from auto_grader.mc_results_demo_export import build_mc_results_demo_export
from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db


def build_review_resolutions_from_simple_map(
    *,
    simple_resolutions: dict[str, str | None],
    current_results: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Transform a simple {question_id: bubble_label_or_None} map into the
    full review_resolutions_by_scan_id shape that the DB layer expects.

    The professor only needs to say "mc-1 is B" or "mc-3 is blank (None)".
    This function looks up each question in the current DB truth to get the
    scan_id, machine_status, and correct_bubble_label, then computes the
    final status and builds the override dict.

    Any question_id may be resolved, not only review_required ones. This is
    intentional: the professor may override any machine result (e.g. to
    correct a machine misread on a question that wasn't flagged).

    Raises KeyError if a question_id is not found in current_results.
    """
    question_results = current_results["question_results"]
    by_scan_id: dict[str, dict[str, dict[str, Any]]] = {}

    for question_id, resolved_bubble_label in simple_resolutions.items():
        qr = question_results[question_id]  # KeyError if unknown
        scan_id = qr["scan_id"]
        original_status = qr["machine_status"]
        correct_bubble_label = qr["correct_bubble_label"]

        if resolved_bubble_label is None:
            final_status = "blank"
            final_is_correct = False
        else:
            final_is_correct = resolved_bubble_label == correct_bubble_label
            final_status = "correct" if final_is_correct else "incorrect"

        resolved_question: dict[str, Any] = {
            "question_id": question_id,
            "status": final_status,
            "is_correct": final_is_correct,
            "review_required": False,
            "override": {
                "original_status": original_status,
                "resolved_bubble_label": resolved_bubble_label,
            },
        }
        by_scan_id.setdefault(scan_id, {})[question_id] = resolved_question

    return by_scan_id


def get_review_queue(
    *,
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    """Return the review queue and summary for an exam instance.

    Thin wrapper over build_mc_results_demo_export that extracts just the
    fields a professor needs to see what requires review.
    """
    export = build_mc_results_demo_export(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    return {
        "exam_instance_id": export["exam_instance_id"],
        "mc_scan_session_id": export["mc_scan_session_id"],
        "review_queue": export["review_queue"],
        "summary": export["summary"],
    }


def resolve_and_persist(
    *,
    exam_instance_id: int,
    simple_resolutions: dict[str, str | None],
    connection: object,
) -> dict[str, Any]:
    """Resolve MC questions using a simple map and persist to DB.

    Reads the current truth to look up question context, builds the full
    resolution dicts, persists them atomically across all scan pages, then
    reads updated truth. Any question may be resolved, not only those with
    review_required=True.
    """
    if not isinstance(simple_resolutions, dict):
        raise TypeError("simple_resolutions must be a dict mapping question_id to bubble label or null")

    current = read_current_final_mc_results_from_db(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    mc_scan_session_id = current["mc_scan_session_id"]

    resolutions_by_scan_id = build_review_resolutions_from_simple_map(
        simple_resolutions=simple_resolutions,
        current_results=current,
    )

    created = 0
    updated = 0
    unchanged = 0
    with connection.transaction():
        for scan_id, resolved_questions in resolutions_by_scan_id.items():
            persisted = persist_mc_review_resolutions_to_db(
                mc_scan_session_id=mc_scan_session_id,
                scan_id=scan_id,
                resolved_questions=resolved_questions,
                connection=connection,
            )
            created += persisted["created"]
            updated += persisted["updated"]
            unchanged += persisted["unchanged"]

    updated_results = read_current_final_mc_results_from_db(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )

    return {
        "exam_instance_id": exam_instance_id,
        "mc_scan_session_id": mc_scan_session_id,
        "review_persist": {
            "created": created,
            "updated": updated,
            "unchanged": unchanged,
        },
        "current_results": updated_results,
    }


def export_results(
    *,
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    """Export the current-final MC truth for an exam instance.

    Returns the full export dict from build_mc_results_demo_export.
    """
    return build_mc_results_demo_export(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )


def render_results_csv(export: dict[str, Any]) -> str:
    """Render the export dict as a CSV string.

    Columns: question_id, page_number, status, is_correct, review_required,
    source, machine_status, resolved_bubble_label.
    """
    columns = [
        "question_id",
        "page_number",
        "status",
        "is_correct",
        "review_required",
        "source",
        "machine_status",
        "resolved_bubble_label",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for question in export["questions"]:
        writer.writerow(question)
    return output.getvalue()
