#!/usr/bin/env python
"""Professor-facing MC workflow CLI.

Subcommands:
    ingest   Ingest scan images into the DB-backed MC/OpenCV workflow.
    review   Show which MC questions need human review after scan ingestion.
    resolve  Submit review resolutions for flagged questions.
    export   Export current-final MC results as JSON, CSV, or text summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from auto_grader.db import create_connection
from auto_grader.mc_workflow import (
    export_results,
    get_review_queue,
    ingest_and_persist_from_scan_dir,
    render_results_csv,
    resolve_and_persist,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exam-instance-id", type=int, required=True,
        help="Exam instance id to operate on.",
    )
    parser.add_argument(
        "--database-url", default=None,
        help="Explicit Postgres URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--schema-name", default=None,
        help="Optional Postgres schema name to prepend to search_path.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where output files should be written.",
    )


def _require_schema_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            "--schema-name must be a simple Postgres identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    return value


def _connect(args: argparse.Namespace) -> object:
    connection = create_connection(args.database_url)
    if args.schema_name is not None:
        connection.execute(
            f"SET search_path TO {_require_schema_identifier(args.schema_name)}, public"
        )
    return connection


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cmd_review(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = _connect(args)
    try:
        result = get_review_queue(
            exam_instance_id=args.exam_instance_id,
            connection=connection,
        )
    finally:
        connection.close()

    queue_path = output_dir / "review-queue.json"
    summary_path = output_dir / "review-summary.txt"

    _write_json(queue_path, result["review_queue"])

    lines = [
        f"exam_instance_id: {result['exam_instance_id']}",
        f"mc_scan_session_id: {result['mc_scan_session_id']}",
        f"unresolved_review_required: {result['summary'].get('unresolved_review_required', 0)}",
        "",
    ]
    if not result["review_queue"]:
        lines.append("No questions require review.")
    else:
        lines.append("Questions requiring review:")
        for item in result["review_queue"]:
            lines.append(
                f"  {item['question_id']} ({item['machine_status']}) "
                f"on {item['scan_id']} page {item['page_number']} "
                f"marked: {item.get('marked_bubble_labels', [])}"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(queue_path)
    print(summary_path)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = _connect(args)
    try:
        result = ingest_and_persist_from_scan_dir(
            artifact_json_path=args.artifact_json,
            scan_dir=args.scan_dir,
            exam_instance_id=args.exam_instance_id,
            output_dir=output_dir,
            connection=connection,
        )
    finally:
        connection.close()

    ingest_result_path = output_dir / "ingest-result.json"
    review_queue_path = output_dir / "review-queue.json"
    summary_path = output_dir / "review-summary.txt"

    _write_json(ingest_result_path, result)
    _write_json(review_queue_path, result["review_queue"])

    lines = [
        f"exam_instance_id: {result['exam_instance_id']}",
        f"mc_scan_session_id: {result['mc_scan_session_id']}",
        f"manifest_path: {result['manifest_path']}",
        f"unresolved_review_required: {result['summary'].get('unresolved_review_required', 0)}",
        "",
    ]
    if not result["review_queue"]:
        lines.append("No questions require review.")
    else:
        lines.append("Questions requiring review:")
        for item in result["review_queue"]:
            lines.append(
                f"  {item['question_id']} ({item['machine_status']}) "
                f"on {item['scan_id']} page {item['page_number']} "
                f"marked: {item.get('marked_bubble_labels', [])}"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(ingest_result_path)
    print(review_queue_path)
    print(summary_path)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolutions_path = Path(args.resolutions_json)
    if not resolutions_path.is_file():
        print(f"Error: {resolutions_path} does not exist.", file=sys.stderr)
        return 1

    simple_resolutions = json.loads(
        resolutions_path.read_text(encoding="utf-8")
    )
    if not isinstance(simple_resolutions, dict):
        print(
            "Error: resolutions JSON must be an object mapping question_id "
            'to bubble label or null. Example: {"mc-1": "B", "mc-3": null}',
            file=sys.stderr,
        )
        return 1

    connection = _connect(args)
    try:
        result = resolve_and_persist(
            exam_instance_id=args.exam_instance_id,
            simple_resolutions=simple_resolutions,
            connection=connection,
        )
    finally:
        connection.close()

    result_path = output_dir / "resolve-result.json"
    _write_json(result_path, result)
    print(result_path)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = _connect(args)
    try:
        exported = export_results(
            exam_instance_id=args.exam_instance_id,
            connection=connection,
        )
    finally:
        connection.close()

    fmt = args.format

    if fmt in ("json", "all"):
        json_path = output_dir / "mc-results.json"
        _write_json(json_path, exported)
        print(json_path)

    if fmt in ("csv", "all"):
        csv_path = output_dir / "mc-results.csv"
        csv_text = render_results_csv(exported)
        csv_path.write_text(csv_text, encoding="utf-8")
        print(csv_path)

    if fmt in ("txt", "all"):
        txt_path = output_dir / "mc-results-summary.txt"
        txt_path.write_text(
            _render_summary_text(exported), encoding="utf-8",
        )
        print(txt_path)

    return 0


def _render_summary_text(exported: dict) -> str:
    """Render a human-readable text summary of the export."""
    lines = [
        f"exam_instance_id: {exported['exam_instance_id']}",
        f"mc_scan_session_id: {exported['mc_scan_session_id']}",
        f"session_ordinal: {exported['session_ordinal']}",
    ]
    summary = exported["summary"]
    for key in (
        "matched", "unmatched", "ambiguous", "unresolved_review_required",
        "resolved_by_review", "correct", "incorrect", "blank", "question_count",
    ):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")

    lines.append("")
    lines.append("questions:")
    for question in exported["questions"]:
        machine_status = question["machine_status"]
        final_status = question["status"]
        line = f"- {question['question_id']}: {machine_status} -> {final_status}"
        if question["resolved_bubble_label"] is not None:
            line += f" [{question['resolved_bubble_label']}]"
        if question["source"] == "review_resolution":
            line += " via review"
        lines.append(line)

    if exported["review_queue"]:
        lines.append("")
        lines.append("review_queue:")
        for item in exported["review_queue"]:
            lines.append(
                f"- {item['question_id']} ({item['machine_status']}) "
                f"on {item['scan_id']} page {item['page_number']}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Professor-facing MC workflow for scan grading, review, and export.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a directory of scan images into the DB-backed MC/OpenCV workflow.",
    )
    _add_common_args(ingest_parser)
    ingest_parser.add_argument(
        "--artifact-json",
        required=True,
        help="Path to the exam artifact JSON used to interpret the scans.",
    )
    ingest_parser.add_argument(
        "--scan-dir",
        required=True,
        help="Directory containing scan images (.png, .jpg, .jpeg).",
    )

    # review
    review_parser = subparsers.add_parser(
        "review", help="Show which MC questions need human review.",
    )
    _add_common_args(review_parser)

    # resolve
    resolve_parser = subparsers.add_parser(
        "resolve", help="Submit review resolutions for flagged questions.",
    )
    _add_common_args(resolve_parser)
    resolve_parser.add_argument(
        "--resolutions-json", required=True,
        help=(
            'Path to a JSON file mapping question_id to resolved bubble label '
            '(or null for blank). Example: {"mc-1": "B", "mc-3": null}'
        ),
    )

    # export
    export_parser = subparsers.add_parser(
        "export", help="Export current-final MC results.",
    )
    _add_common_args(export_parser)
    export_parser.add_argument(
        "--format", default="json", choices=["json", "csv", "txt", "all"],
        help="Output format (default: json).",
    )

    args = parser.parse_args()
    commands = {
        "ingest": cmd_ingest,
        "review": cmd_review,
        "resolve": cmd_resolve,
        "export": cmd_export,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
