from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_grader.quiz5_short_answer_trial_prep import (
    prepare_quiz5_short_answer_trial_crops,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Quiz #5 per-response-box crops from a staged artifact and ingest output."
    )
    parser.add_argument(
        "--artifact-json",
        required=True,
        help="Path to a staged Quiz #5 short-answer artifact JSON file.",
    )
    parser.add_argument(
        "--ingest-output-dir",
        required=True,
        help="Directory containing a Quiz #5 ingest manifest and normalized_images/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trial-prep manifest and crops should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact_path = Path(args.artifact_json)
    ingest_output_dir = Path(args.ingest_output_dir)
    output_dir = Path(args.output_dir)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest_path = ingest_output_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = prepare_quiz5_short_answer_trial_crops(
        artifact=artifact,
        manifest=manifest,
        normalized_dir=ingest_output_dir / "normalized_images",
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
