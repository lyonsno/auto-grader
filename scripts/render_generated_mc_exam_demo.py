from __future__ import annotations

import argparse
import json

from auto_grader.generated_exam_demo import write_generated_mc_exam_demo_packet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a printable generated-MC exam demo packet from a real template."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--template-path",
        default="templates/chm141-final-fall2023.yaml",
        help="Repo-local exam template YAML to generate from.",
    )
    parser.add_argument("--student-id", default="demo-001")
    parser.add_argument("--student-name", default="Demo Student")
    parser.add_argument("--attempt-number", type=int, default=1)
    parser.add_argument("--seed", default="17")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_generated_mc_exam_demo_packet(
        output_dir=args.output_dir,
        template_path=args.template_path,
        student_id=args.student_id,
        student_name=args.student_name,
        attempt_number=args.attempt_number,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
