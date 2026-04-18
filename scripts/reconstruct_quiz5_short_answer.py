from __future__ import annotations

import argparse
import json

from auto_grader.quiz5_short_answer_reconstruction import (
    write_reconstructed_short_answer_quiz_family,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the Quiz #5 short-answer family from legacy PDF variants."
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        action="append",
        required=True,
        help="Path to one legacy quiz PDF variant. Pass multiple times for A/B/etc.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the reconstructed family JSON should be written.",
    )
    parser.add_argument(
        "--generate-variant",
        dest="generate_variant_ids",
        action="append",
        default=[],
        help="Optional sibling variant id to generate as a reviewable JSON packet (e.g. C).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_reconstructed_short_answer_quiz_family(
        pdf_paths=args.pdf_paths,
        output_dir=args.output_dir,
        generate_variant_ids=args.generate_variant_ids,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
