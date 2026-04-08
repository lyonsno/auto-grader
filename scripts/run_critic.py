"""Run the post-hoc critic over a grader run.

Usage:
    uv run python scripts/run_critic.py runs/<ts>-<model>/
    uv run python scripts/run_critic.py runs/<ts>-<model>/predictions.jsonl

Reads predictions.jsonl from the run directory (or accepts a direct
path), applies the v1 deterministic consistency-rule critic, writes
critic.jsonl alongside, and prints a summary to stdout.

The critic is post-hoc and idempotent — running it twice on the same
predictions.jsonl produces the same critic.jsonl. It is safe to run
after every smoke and treat critic.jsonl as the corrected score
source-of-truth for downstream analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_grader.critic import (
    critique_run,
    summarize,
    write_critic_jsonl,
)


def _resolve_predictions_path(arg: str) -> Path:
    """Accept either a run dir or a direct predictions.jsonl path."""
    p = Path(arg)
    if p.is_dir():
        candidate = p / "predictions.jsonl"
        if not candidate.is_file():
            print(
                f"error: {p} has no predictions.jsonl",
                file=sys.stderr,
            )
            sys.exit(1)
        return candidate
    if p.is_file():
        return p
    print(f"error: {p} is not a file or directory", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the v1 deterministic consistency-rule critic to a "
            "grader run. Reads predictions.jsonl, writes critic.jsonl."
        )
    )
    parser.add_argument(
        "run",
        help="Run directory (containing predictions.jsonl) or direct path to predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for critic.jsonl (default: alongside predictions.jsonl)",
    )
    args = parser.parse_args()

    predictions_path = _resolve_predictions_path(args.run)
    out_path = (
        Path(args.out) if args.out else predictions_path.parent / "critic.jsonl"
    )

    deltas = critique_run(predictions_path)
    write_critic_jsonl(deltas, out_path)

    print(summarize(deltas))
    print()
    print(f"Wrote {out_path}")

    overrides = [d for d in deltas if d.action == "override"]
    return 0 if not overrides else 0  # exit 0 even with overrides — they're not errors


if __name__ == "__main__":
    raise SystemExit(main())
