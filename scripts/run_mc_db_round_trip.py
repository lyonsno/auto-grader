#!/usr/bin/env python
"""Run the DB-backed MC/OpenCV round trip from a persisted scan manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from auto_grader.db import create_connection
from auto_grader.mc_db_round_trip import run_mc_db_round_trip


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist one MC/OpenCV scan manifest into the DB, optionally persist "
            "human review resolutions, and emit the authoritative current-final "
            "MC truth for the exam instance."
        )
    )
    parser.add_argument(
        "--manifest-json",
        required=True,
        help="Path to the persisted MC scan-session manifest JSON.",
    )
    parser.add_argument(
        "--exam-instance-id",
        type=int,
        required=True,
        help="Exam instance id to persist/read against.",
    )
    parser.add_argument(
        "--review-resolutions-json",
        default=None,
        help="Optional JSON mapping scan_id -> resolved question outcomes.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Explicit Postgres URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional Postgres schema name to prepend to search_path before running.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the round-trip JSON bundle should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    review_resolutions_by_scan_id = {}
    if args.review_resolutions_json is not None:
        review_resolutions_by_scan_id = json.loads(
            Path(args.review_resolutions_json).read_text(encoding="utf-8")
        )

    connection = create_connection(args.database_url)
    try:
        if args.schema_name is not None:
            connection.execute(
                f"SET search_path TO {_require_schema_identifier(args.schema_name)}, public"
            )
        result = run_mc_db_round_trip(
            manifest=manifest,
            exam_instance_id=args.exam_instance_id,
            review_resolutions_by_scan_id=review_resolutions_by_scan_id,
            connection=connection,
        )
    finally:
        connection.close()

    output_path = output_dir / "mc-db-round-trip.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


def _require_schema_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            "--schema-name must be a simple Postgres identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
