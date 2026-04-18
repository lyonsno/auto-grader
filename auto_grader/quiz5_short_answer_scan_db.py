"""Persist returned Quiz #5 short-answer scan-session manifests into the durable DB model."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def persist_quiz5_short_answer_scan_session_manifest_to_db(
    *,
    manifest: dict[str, Any],
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    if isinstance(exam_instance_id, bool) or not isinstance(exam_instance_id, int):
        raise TypeError("exam_instance_id must be an integer")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a mapping")

    manifest_fingerprint = _manifest_fingerprint(manifest)
    existing = connection.execute(
        """
        SELECT id
        FROM audit_events
        WHERE entity_type = %s
          AND entity_id = %s
          AND event_type = %s
          AND payload_json->>'manifest_fingerprint' = %s
        LIMIT 1
        """,
        (
            "exam_instance",
            exam_instance_id,
            "quiz5_short_answer_scan_session_persisted",
            manifest_fingerprint,
        ),
    ).fetchone()
    if existing is not None:
        return {
            "created": False,
            "exam_instance_id": exam_instance_id,
            "audit_event_id": existing["id"],
            "manifest_fingerprint": manifest_fingerprint,
            "summary": manifest["summary"],
        }

    with connection.transaction():
        persisted_scan_results: list[dict[str, Any]] = []
        for scan_result in manifest["scan_results"]:
            scan_artifact_id = _get_or_create_scan_artifact(
                scan_result=scan_result,
                connection=connection,
            )
            persisted_scan_results.append(
                {
                    "scan_id": scan_result["scan_id"],
                    "scan_artifact_id": scan_artifact_id,
                    "status": scan_result["status"],
                    "checksum": scan_result["checksum"],
                    "failure_reason": scan_result.get("failure_reason"),
                    "page_number": scan_result.get("page_number"),
                    "fallback_page_code": scan_result.get("fallback_page_code"),
                }
            )

        audit_event_id = connection.execute(
            """
            INSERT INTO audit_events (
                entity_type,
                entity_id,
                event_type,
                payload_json
            )
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING id
            """,
            (
                "exam_instance",
                exam_instance_id,
                "quiz5_short_answer_scan_session_persisted",
                json.dumps(
                    {
                        "manifest_fingerprint": manifest_fingerprint,
                        "opaque_instance_code": manifest["opaque_instance_code"],
                        "expected_page_codes": manifest["expected_page_codes"],
                        "summary": manifest["summary"],
                        "scan_results": persisted_scan_results,
                    },
                    sort_keys=True,
                ),
            ),
        ).fetchone()["id"]

    return {
        "created": True,
        "exam_instance_id": exam_instance_id,
        "audit_event_id": audit_event_id,
        "manifest_fingerprint": manifest_fingerprint,
        "summary": manifest["summary"],
        "scan_results": persisted_scan_results,
    }


def get_exam_instance_id_for_opaque_instance_code(
    *,
    opaque_instance_code: str,
    connection: object,
) -> int:
    row = connection.execute(
        "SELECT id FROM exam_instances WHERE opaque_instance_code = %s",
        (opaque_instance_code,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown opaque_instance_code {opaque_instance_code!r}")
    return int(row["id"])


_get_exam_instance_id_for_opaque_instance_code = get_exam_instance_id_for_opaque_instance_code


def _get_or_create_scan_artifact(
    *,
    scan_result: dict[str, Any],
    connection: object,
) -> int:
    existing = connection.execute(
        "SELECT id FROM scan_artifacts WHERE sha256 = %s",
        (scan_result["checksum"],),
    ).fetchone()
    if existing is not None:
        return int(existing["id"])

    return int(
        connection.execute(
            """
            INSERT INTO scan_artifacts (
                sha256,
                original_filename,
                status,
                failure_reason
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                scan_result["checksum"],
                scan_result["scan_id"],
                scan_result["status"],
                scan_result.get("failure_reason"),
            ),
        ).fetchone()["id"]
    )


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
