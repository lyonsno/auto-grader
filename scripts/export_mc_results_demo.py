#!/usr/bin/env python
"""Export the current-final DB-backed MC results into a compact demo bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from auto_grader.db import create_connection
from auto_grader.mc_results_demo_export import build_mc_results_demo_export


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the current-final MC truth for one exam instance from the DB "
            "and write a compact demo bundle."
        )
    )
    parser.add_argument(
        "--exam-instance-id",
        type=int,
        required=True,
        help="Exam instance id to export.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Explicit Postgres URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional Postgres schema name to prepend to search_path before reading results.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the demo JSON and summary text should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = create_connection(args.database_url)
    try:
        if args.schema_name is not None:
            connection.execute(
                f"SET search_path TO {_require_schema_identifier(args.schema_name)}, public"
            )
        exported = build_mc_results_demo_export(
            exam_instance_id=args.exam_instance_id,
            connection=connection,
        )
    finally:
        connection.close()

    json_path = output_dir / "mc-results-export.json"
    summary_path = output_dir / "mc-results-summary.txt"
    json_path.write_text(
        json.dumps(exported, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        _render_summary_text(exported),
        encoding="utf-8",
    )

    print(json_path)
    print(summary_path)
    return 0


def _render_summary_text(exported: dict) -> str:
    lines = [
        f"exam_instance_id: {exported['exam_instance_id']}",
        f"mc_scan_session_id: {exported['mc_scan_session_id']}",
        f"session_ordinal: {exported['session_ordinal']}",
    ]
    summary = exported["summary"]
    for key in (
        "matched",
        "unmatched",
        "ambiguous",
        "unresolved_review_required",
        "resolved_by_review",
        "correct",
        "incorrect",
        "blank",
        "question_count",
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


def _require_schema_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            "--schema-name must be a simple Postgres identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
