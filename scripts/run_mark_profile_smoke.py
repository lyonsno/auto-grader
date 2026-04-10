from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_grader.mark_profile_smoke import run_mark_profile_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and evaluate a synthetic student-like bubble mark profile matrix."
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/mc-mark-profile-smoke",
        help="Directory where rendered specimens and summary.json should be written.",
    )
    args = parser.parse_args()

    report = run_mark_profile_smoke(output_dir=Path(args.output_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
