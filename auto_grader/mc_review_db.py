"""Persist durable human MC review resolutions into the database."""

from __future__ import annotations

import json
from typing import Any


def persist_mc_review_resolutions_to_db(
    *,
    mc_scan_session_id: int,
    scan_id: str,
    resolved_questions: dict[str, dict[str, Any]],
    connection: object,
) -> dict[str, Any]:
    if isinstance(mc_scan_session_id, bool) or not isinstance(mc_scan_session_id, int):
        raise TypeError("mc_scan_session_id must be an integer")
    if not isinstance(scan_id, str) or scan_id == "":
        raise TypeError("scan_id must be a non-empty string")
    if not isinstance(resolved_questions, dict):
        raise TypeError("resolved_questions must be a mapping")

    created = 0
    updated = 0
    unchanged = 0
    resolution_ids_by_question_id: dict[str, int] = {}

    with connection.transaction():
        for question_id, resolved_question in resolved_questions.items():
            if not isinstance(question_id, str) or question_id == "":
                raise TypeError("resolved_questions keys must be non-empty strings")
            if not isinstance(resolved_question, dict):
                raise TypeError("resolved_questions values must be mappings")

            override = resolved_question.get("override")
            if override is None:
                continue
            if not isinstance(override, dict):
                raise TypeError(f"{question_id}.override must be a mapping")

            outcome_row = connection.execute(
                "SELECT q.id, q.status "
                "FROM mc_question_outcomes q "
                "JOIN mc_scan_pages p ON p.id = q.mc_scan_page_id "
                "WHERE p.mc_scan_session_id = %s AND p.scan_id = %s AND q.question_id = %s",
                (mc_scan_session_id, scan_id, question_id),
            ).fetchone()
            if outcome_row is None:
                raise ValueError(
                    f"Could not find persisted question outcome for scan {scan_id!r} "
                    f"question {question_id!r} in session {mc_scan_session_id}"
                )

            original_status = _require_string(override.get("original_status"), "override.original_status")
            resolved_bubble_label = override.get("resolved_bubble_label")
            if resolved_bubble_label is not None and (
                not isinstance(resolved_bubble_label, str) or resolved_bubble_label == ""
            ):
                raise TypeError("override.resolved_bubble_label must be a non-empty string or None")

            final_status = _require_string(resolved_question.get("status"), "resolved_question.status")
            final_is_correct = _require_bool(resolved_question.get("is_correct"), "resolved_question.is_correct")

            existing = connection.execute(
                "SELECT * FROM mc_review_resolutions WHERE mc_question_outcome_id = %s",
                (outcome_row["id"],),
            ).fetchone()

            payload_json = json.dumps(
                {
                    "mc_scan_session_id": mc_scan_session_id,
                    "scan_id": scan_id,
                    "question_id": question_id,
                    "original_status": original_status,
                    "resolved_bubble_label": resolved_bubble_label,
                    "final_status": final_status,
                    "final_is_correct": final_is_correct,
                },
                sort_keys=True,
            )

            if existing is None:
                resolution = connection.execute(
                    "INSERT INTO mc_review_resolutions "
                    "(mc_question_outcome_id, original_status, resolved_bubble_label, final_status, final_is_correct) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (
                        outcome_row["id"],
                        original_status,
                        resolved_bubble_label,
                        final_status,
                        final_is_correct,
                    ),
                ).fetchone()
                resolution_id = resolution["id"]
                connection.execute(
                    "INSERT INTO audit_events (entity_type, entity_id, event_type, payload_json) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    ("mc_review_resolution", resolution_id, "created", payload_json),
                )
                created += 1
                resolution_ids_by_question_id[question_id] = resolution_id
                continue

            resolution_id = existing["id"]
            resolution_ids_by_question_id[question_id] = resolution_id
            if (
                existing["original_status"] == original_status
                and existing["resolved_bubble_label"] == resolved_bubble_label
                and existing["final_status"] == final_status
                and existing["final_is_correct"] == final_is_correct
            ):
                unchanged += 1
                continue

            connection.execute(
                "UPDATE mc_review_resolutions "
                "SET original_status = %s, resolved_bubble_label = %s, final_status = %s, "
                "    final_is_correct = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = %s",
                (
                    original_status,
                    resolved_bubble_label,
                    final_status,
                    final_is_correct,
                    resolution_id,
                ),
            )
            connection.execute(
                "INSERT INTO audit_events (entity_type, entity_id, event_type, payload_json) "
                "VALUES (%s, %s, %s, %s::jsonb)",
                ("mc_review_resolution", resolution_id, "updated", payload_json),
            )
            updated += 1

    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "resolution_ids_by_question_id": resolution_ids_by_question_id,
    }


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
