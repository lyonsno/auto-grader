#!/usr/bin/env python
"""Launch the professor-facing MC workflow GUI."""

from __future__ import annotations

import argparse
import os
import sys

from auto_grader.mc_workflow_gui import GuiState, serve_mc_workflow_gui


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a thin local web GUI over the professor-facing MC workflow "
            "for ingest, review, resolve, and export."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    parser.add_argument("--open-browser", action="store_true", help="Open the GUI in the default browser.")
    parser.add_argument("--database-url", default="", help="Optional initial database URL.")
    parser.add_argument("--schema-name", default="", help="Optional initial schema name.")
    parser.add_argument("--exam-instance-id", default="", help="Optional initial exam instance id.")
    parser.add_argument("--artifact-json", default="", help="Optional initial artifact JSON path.")
    parser.add_argument("--scan-dir", default="", help="Optional initial scan directory.")
    parser.add_argument("--output-dir", default="", help="Optional initial output directory.")
    return parser.parse_args()


def _resolve_database_url(args: argparse.Namespace) -> str:
    """Return the database URL from --database-url or DATABASE_URL.

    Exits with a clear operator-facing message if neither is set.
    """
    url = args.database_url or os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    print(
        "No database connection configured.\n"
        "\n"
        "The GUI needs a running Postgres database to store assessments and\n"
        "grading results. Set the DATABASE_URL environment variable before\n"
        "launching:\n"
        "\n"
        "  export DATABASE_URL=\"postgresql:///postgres\"\n"
        "  python scripts/launch_mc_workflow_gui.py --open-browser\n"
        "\n"
        "Or pass it directly:\n"
        "\n"
        "  python scripts/launch_mc_workflow_gui.py --database-url postgresql:///postgres\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    args = _parse_args()
    database_url = _resolve_database_url(args)
    initial_state = GuiState(
        config={
            "database_url": database_url,
            "schema_name": args.schema_name,
            "exam_instance_id": str(args.exam_instance_id),
            "artifact_json": args.artifact_json,
            "scan_dir": args.scan_dir,
            "output_dir": args.output_dir,
        }
    )
    return serve_mc_workflow_gui(
        host=args.host,
        port=args.port,
        initial_state=initial_state,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
