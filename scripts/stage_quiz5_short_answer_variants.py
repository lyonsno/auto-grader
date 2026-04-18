from __future__ import annotations

import argparse
import json

from auto_grader.quiz5_short_answer_packets import (
    SUPPORTED_GENERATED_VARIANT_IDS,
    write_quiz5_short_answer_variant_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and optionally DB-register Quiz #5 short-answer variants."
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        action="append",
        required=True,
        help="Path to one legacy Quiz #5 PDF variant. Pass multiple times for A/B/etc.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the rendered Quiz #5 artifacts should be written.",
    )
    parser.add_argument(
        "--generate-variant",
        dest="generate_variant_ids",
        action="append",
        default=[],
        choices=sorted(SUPPORTED_GENERATED_VARIANT_IDS),
        help="Optional sibling variant id to generate in addition to the observed variants.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional Postgres URL. When provided, stage the rendered variants into the durable DB model too.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_quiz5_short_answer_variant_bundle(
        pdf_paths=args.pdf_paths,
        output_dir=args.output_dir,
        generated_variant_ids=args.generate_variant_ids,
        database_url=args.database_url,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
