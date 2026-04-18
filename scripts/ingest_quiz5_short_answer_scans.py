from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from auto_grader.db import create_connection
from auto_grader.quiz5_short_answer_scan_db import (
    _get_exam_instance_id_for_opaque_instance_code,
    persist_quiz5_short_answer_scan_session_manifest_to_db,
)
from auto_grader.quiz5_short_answer_scan_session import (
    persist_quiz5_short_answer_scan_session,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package Quiz #5 short-answer scans into a durable ingest manifest."
    )
    parser.add_argument(
        "--artifact-json",
        required=True,
        help="Path to a staged Quiz #5 short-answer artifact JSON file.",
    )
    parser.add_argument(
        "--scan-dir",
        required=True,
        help="Directory containing page scan images for one staged Quiz #5 packet.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the ingest manifest and normalized images should be written.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional Postgres URL. When provided, persist the returned scan-session manifest into the durable DB model too.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact_path = Path(args.artifact_json)
    scan_dir = Path(args.scan_dir)
    output_dir = Path(args.output_dir)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scan_images = _load_scan_images(scan_dir)
    result = persist_quiz5_short_answer_scan_session(
        scan_images=scan_images,
        artifact=artifact,
        output_dir=str(output_dir),
    )
    if args.database_url:
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        connection = create_connection(args.database_url)
        try:
            exam_instance_id = _get_exam_instance_id_for_opaque_instance_code(
                opaque_instance_code=artifact["opaque_instance_code"],
                connection=connection,
            )
            result["db"] = persist_quiz5_short_answer_scan_session_manifest_to_db(
                manifest=manifest,
                exam_instance_id=exam_instance_id,
                connection=connection,
            )
        finally:
            connection.close()
    print(json.dumps(result, indent=2))
    return 0


def _load_scan_images(scan_dir: Path) -> dict[str, object]:
    if not scan_dir.is_dir():
        raise ValueError(f"scan-dir must be an existing directory: {scan_dir}")

    images: dict[str, object] = {}
    for path in sorted(scan_dir.iterdir()):
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images[path.name] = image

    if not images:
        raise ValueError(f"No readable scan images were found in {scan_dir}")
    return images


if __name__ == "__main__":
    raise SystemExit(main())
