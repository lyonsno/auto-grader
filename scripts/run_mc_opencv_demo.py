#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from auto_grader.mc_opencv_demo import run_mc_opencv_demo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the landed MC/OpenCV pipeline over one directory of scans.",
    )
    parser.add_argument("--artifact-json", required=True, help="Path to the exam artifact JSON.")
    parser.add_argument("--scan-dir", required=True, help="Directory containing scanned page images.")
    parser.add_argument("--output-dir", required=True, help="Directory where the demo bundle should be written.")
    args = parser.parse_args()

    result = run_mc_opencv_demo(
        artifact_json_path=args.artifact_json,
        scan_dir=args.scan_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
