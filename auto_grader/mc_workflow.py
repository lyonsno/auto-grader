"""Professor-facing MC workflow entrypoint.

Composes the landed DB-backed MC primitives into a usable surface for the
full scan -> grade -> review -> export loop without requiring the professor
to know about internal module boundaries or prepare complex JSON by hand.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import cv2

from auto_grader.mc_db_round_trip import run_mc_db_round_trip
from auto_grader.mc_results_db import read_current_final_mc_results_from_db
from auto_grader.mc_results_demo_export import build_mc_results_demo_export
from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db
from auto_grader.mc_scan_session import persist_scan_session


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


def list_assessment_definitions(*, connection: object) -> list[dict[str, Any]]:
    """Return authored assessment definitions in professor-facing label form."""
    rows = connection.execute(
        """
        SELECT id, slug, version, title
        FROM exam_definitions
        ORDER BY title, version, id
        """
    ).fetchall()
    return [
        {
            "exam_definition_id": row["id"],
            "slug": row["slug"],
            "version": row["version"],
            "title": row["title"],
            "label": row["title"] if row["version"] == 1 else f"{row['title']} (v{row['version']})",
        }
        for row in rows
    ]


def list_grading_targets(*, connection: object) -> list[dict[str, Any]]:
    """Return saved grading targets with human-readable labels."""
    rows = connection.execute(
        """
        SELECT
            ei.id AS exam_instance_id,
            ed.id AS exam_definition_id,
            ed.title AS exam_title,
            ei.opaque_instance_code AS target_name,
            ei.attempt_number AS attempt_number,
            s.student_key AS student_key
        FROM exam_instances ei
        JOIN exam_definitions ed ON ed.id = ei.exam_definition_id
        JOIN students s ON s.id = ei.student_id
        ORDER BY ei.id
        """
    ).fetchall()
    return [
        {
            "exam_instance_id": row["exam_instance_id"],
            "exam_definition_id": row["exam_definition_id"],
            "exam_title": row["exam_title"],
            "target_name": row["target_name"],
            "attempt_number": row["attempt_number"],
            "student_key": row["student_key"],
            "label": f"{row['exam_title']} - {row['target_name']}",
        }
        for row in rows
    ]


def create_grading_target(
    *,
    exam_definition_id: int,
    target_name: str,
    connection: object,
) -> dict[str, Any]:
    """Create one professor-facing grading target in the durable model."""
    if isinstance(exam_definition_id, bool) or not isinstance(exam_definition_id, int):
        raise TypeError("exam_definition_id must be an integer")
    if not isinstance(target_name, str) or target_name.strip() == "":
        raise ValueError("target_name must be a non-empty string")

    clean_name = target_name.strip()
    exam_definition = connection.execute(
        "SELECT id, title FROM exam_definitions WHERE id = %s",
        (exam_definition_id,),
    ).fetchone()
    if exam_definition is None:
        raise KeyError(f"Unknown exam_definition_id {exam_definition_id}")

    final_target_name = _next_available_target_name(
        base_name=clean_name,
        connection=connection,
    )
    student_key = f"grading-target::{final_target_name}"

    with connection.transaction():
        student_row = connection.execute(
            "INSERT INTO students (student_key) VALUES (%s) RETURNING id",
            (student_key,),
        ).fetchone()
        exam_instance_row = connection.execute(
            """
            INSERT INTO exam_instances
            (exam_definition_id, student_id, attempt_number, opaque_instance_code)
            VALUES (%s, %s, 1, %s)
            RETURNING id
            """,
            (exam_definition_id, student_row["id"], final_target_name),
        ).fetchone()

    return {
        "exam_instance_id": exam_instance_row["id"],
        "exam_definition_id": exam_definition_id,
        "exam_title": exam_definition["title"],
        "target_name": final_target_name,
        "label": f"{exam_definition['title']} - {final_target_name}",
    }


def ingest_and_persist_from_scan_dir(
    *,
    artifact_json_path: str | Path,
    scan_dir: str | Path,
    exam_instance_id: int,
    output_dir: str | Path,
    connection: object,
) -> dict[str, Any]:
    """Ingest a directory of scan images and persist the DB-backed workflow result.

    This is the professor-facing bridge from "scan directory + artifact" to the
    landed MC/OpenCV and DB-backed workflow primitives.
    """
    artifact_path = _require_existing_file(artifact_json_path, "artifact_json_path")
    scan_directory = _require_existing_dir(scan_dir, "scan_dir")
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scan_images = _load_scan_images(scan_directory)
    session = persist_scan_session(
        scan_images=scan_images,
        artifact=artifact,
        output_dir=str(output_directory),
    )
    manifest = json.loads(Path(session["manifest_path"]).read_text(encoding="utf-8"))
    round_trip = run_mc_db_round_trip(
        manifest=manifest,
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    queue = get_review_queue(
        exam_instance_id=exam_instance_id,
        connection=connection,
    )
    return {
        "exam_instance_id": exam_instance_id,
        "artifact_json_path": str(artifact_path),
        "scan_dir": str(scan_directory),
        "output_dir": str(output_directory),
        "manifest_path": str(session["manifest_path"]),
        "mc_scan_session_id": round_trip["mc_scan_session_id"],
        "machine_persist": round_trip["machine_persist"],
        "summary": queue["summary"],
        "review_queue": queue["review_queue"],
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


def _load_scan_images(scan_dir: Path) -> dict[str, Any]:
    scans: dict[str, Any] = {}
    for path in sorted(scan_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load scan image {path}")
        scans[path.name] = image
    if not scans:
        raise ValueError(f"No scan images found in {scan_dir}")
    return scans


def _require_existing_dir(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must point to an existing directory")
    return path


def _next_available_target_name(*, base_name: str, connection: object) -> str:
    candidate = base_name
    suffix = 2
    while True:
        row = connection.execute(
            "SELECT 1 FROM exam_instances WHERE opaque_instance_code = %s",
            (candidate,),
        ).fetchone()
        if row is None:
            return candidate
        candidate = f"{base_name} ({suffix})"
        suffix += 1


def _require_existing_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} must point to an existing file")
    return path
