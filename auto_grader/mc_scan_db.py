"""Persist MC scan session results into the database.

Takes a session manifest (as produced by ``mc_scan_session.persist_scan_session``)
plus an exam_instance_id and writes scan session identity, per-page match status,
per-question machine-scored MC outcomes, and review-required flags.

Idempotency: re-running the same manifest for the same exam instance is a no-op.
Supersession: a new manifest for the same exam instance creates a new session row
with linkage back to the prior session, and flags divergence on scan pages whose
outcomes differ from the prior session.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def persist_scan_session_to_db(
    *,
    manifest: dict[str, Any],
    exam_instance_id: int,
    connection: object,
) -> dict[str, Any]:
    """Write MC scan session results into the database idempotently.

    Returns a dict with ``mc_scan_session_id`` and ``created`` keys.
    """
    fingerprint = _manifest_fingerprint(manifest)

    # Check if this exact session already exists (idempotent path).
    existing = connection.execute(
        "SELECT id FROM mc_scan_sessions "
        "WHERE exam_instance_id = %s AND manifest_fingerprint = %s",
        (exam_instance_id, fingerprint),
    ).fetchone()

    if existing is not None:
        return {"mc_scan_session_id": existing["id"], "created": False}

    # Determine ordinal and supersession linkage.
    prior = connection.execute(
        "SELECT id, session_ordinal FROM mc_scan_sessions "
        "WHERE exam_instance_id = %s "
        "ORDER BY session_ordinal DESC LIMIT 1",
        (exam_instance_id,),
    ).fetchone()

    if prior is None:
        session_ordinal = 1
        supersedes_session_id = None
    else:
        session_ordinal = prior["session_ordinal"] + 1
        supersedes_session_id = prior["id"]

    # Gather prior scan outcomes for divergence detection.
    prior_scan_outcomes: dict[str, dict[str, Any]] = {}
    if supersedes_session_id is not None:
        prior_pages = connection.execute(
            "SELECT id, scan_id, checksum, status, failure_reason, page_number "
            "FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (supersedes_session_id,),
        ).fetchall()
        for pp in prior_pages:
            prior_questions = connection.execute(
                "SELECT question_id, status, is_correct "
                "FROM mc_question_outcomes WHERE mc_scan_page_id = %s",
                (pp["id"],),
            ).fetchall()
            pp["_question_outcomes"] = {
                q["question_id"]: q for q in prior_questions
            }
            prior_scan_outcomes[pp["scan_id"]] = pp

    # All mutations in a single transaction — if anything fails mid-way,
    # nothing is committed. Prevents partial sessions from appearing in the DB.
    with connection.transaction():
        # Insert session row.
        session_row = connection.execute(
            "INSERT INTO mc_scan_sessions "
            "(exam_instance_id, manifest_fingerprint, session_ordinal, supersedes_session_id) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (exam_instance_id, fingerprint, session_ordinal, supersedes_session_id),
        ).fetchone()
        session_id = session_row["id"]

        # Insert scan pages and question outcomes.
        for scan_result in manifest["scan_results"]:
            divergence_detected = _detect_divergence(scan_result, prior_scan_outcomes)

            page_row = connection.execute(
                "INSERT INTO mc_scan_pages "
                "(mc_scan_session_id, scan_id, checksum, status, failure_reason, "
                " page_number, fallback_page_code, divergence_detected) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    session_id,
                    scan_result["scan_id"],
                    scan_result["checksum"],
                    scan_result["status"],
                    scan_result.get("failure_reason"),
                    scan_result.get("page_number"),
                    scan_result.get("fallback_page_code"),
                    divergence_detected,
                ),
            ).fetchone()
            page_id = page_row["id"]

            # Only matched scans carry scored questions.
            scored_questions = scan_result.get("scored_questions")
            if scan_result["status"] == "matched" and scored_questions:
                for question_id, outcome in scored_questions.items():
                    connection.execute(
                        "INSERT INTO mc_question_outcomes "
                        "(mc_scan_page_id, question_id, status, is_correct, "
                        " review_required, marked_bubble_labels, resolved_bubble_labels, "
                        " correct_bubble_label, correct_choice_key, marked_choice_keys) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            page_id,
                            question_id,
                            outcome["status"],
                            outcome["is_correct"],
                            outcome["review_required"],
                            json.dumps(outcome.get("marked_bubble_labels", [])),
                            json.dumps(outcome.get("resolved_bubble_labels", [])),
                            outcome["correct_bubble_label"],
                            outcome["correct_choice_key"],
                            json.dumps(outcome.get("marked_choice_keys", [])),
                        ),
                    )

    return {"mc_scan_session_id": session_id, "created": True}


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a stable content fingerprint of the manifest."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _detect_divergence(
    scan_result: dict[str, Any],
    prior_scan_outcomes: dict[str, dict[str, Any]],
) -> bool:
    """Check whether this scan's outcomes diverge from its prior-session counterpart."""
    if not prior_scan_outcomes:
        return False

    scan_id = scan_result["scan_id"]
    prior = prior_scan_outcomes.get(scan_id)
    if prior is None:
        # New scan not in prior session — no divergence.
        return False

    # Compare page-level fields.
    if scan_result["status"] != prior["status"]:
        return True
    if scan_result.get("page_number") != prior["page_number"]:
        return True
    if scan_result.get("failure_reason") != prior["failure_reason"]:
        return True

    # Compare question-level outcomes for matched scans.
    prior_questions = prior.get("_question_outcomes", {})
    current_questions = scan_result.get("scored_questions") or {}
    if set(current_questions.keys()) != set(prior_questions.keys()):
        return True
    for qid, current_q in current_questions.items():
        prior_q = prior_questions.get(qid)
        if prior_q is None:
            return True
        if current_q["status"] != prior_q["status"]:
            return True
        if current_q["is_correct"] != prior_q["is_correct"]:
            return True

    return False
