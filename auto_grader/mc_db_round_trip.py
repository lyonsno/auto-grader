"""Thin DB-backed MC/OpenCV round-trip orchestration surface."""

from __future__ import annotations

from typing import Any

from auto_grader.mc_results_db import read_current_final_mc_results_from_db
from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db
from auto_grader.mc_scan_db import persist_scan_session_to_db


def run_mc_db_round_trip(
    *,
    manifest: dict[str, Any],
    exam_instance_id: int,
    connection: object,
    review_resolutions_by_scan_id: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if isinstance(exam_instance_id, bool) or not isinstance(exam_instance_id, int):
        raise TypeError("exam_instance_id must be an integer")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a mapping")
    if review_resolutions_by_scan_id is None:
        review_resolutions_by_scan_id = {}
    if not isinstance(review_resolutions_by_scan_id, dict):
        raise TypeError("review_resolutions_by_scan_id must be a mapping")

    machine_persist = persist_scan_session_to_db(
        manifest=manifest,
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    mc_scan_session_id = machine_persist["mc_scan_session_id"]

    created = 0
    updated = 0
    unchanged = 0
    by_scan_id: dict[str, dict[str, Any]] = {}
    for scan_id, resolved_questions in review_resolutions_by_scan_id.items():
        if not isinstance(scan_id, str) or scan_id == "":
            raise TypeError("review_resolutions_by_scan_id keys must be non-empty strings")
        if not isinstance(resolved_questions, dict):
            raise TypeError("review_resolutions_by_scan_id values must be mappings")
        persisted = persist_mc_review_resolutions_to_db(
            mc_scan_session_id=mc_scan_session_id,
            scan_id=scan_id,
            resolved_questions=resolved_questions,
            connection=connection,
        )
        by_scan_id[scan_id] = persisted
        created += persisted["created"]
        updated += persisted["updated"]
        unchanged += persisted["unchanged"]

    current_results = read_current_final_mc_results_from_db(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )

    return {
        "exam_instance_id": exam_instance_id,
        "mc_scan_session_id": mc_scan_session_id,
        "machine_persist": machine_persist,
        "review_resolution_persist": {
            "created": created,
            "updated": updated,
            "unchanged": unchanged,
            "scan_ids": sorted(by_scan_id),
            "by_scan_id": by_scan_id,
        },
        "current_results": current_results,
    }
