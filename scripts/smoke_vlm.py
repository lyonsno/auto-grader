"""Smoke test: run a small subset of eval items through the VLM pipeline.

Usage:
    uv run python scripts/smoke_vlm.py [--model MODEL] [--items N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from contextlib import nullcontext

from auto_grader.eval_harness import load_ground_truth, score_predictions
from auto_grader.narrator import BonsaiNarrator
from auto_grader.vlm_inference import ServerConfig, grade_all_items


_GROUND_TRUTH = Path(__file__).resolve().parent.parent / "eval" / "ground_truth.yaml"
_SCANS_DIR = Path.home() / "dev" / "auto-grader-assets" / "scans"
_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "templates"
    / "chm141-final-fall2023.yaml"
)


def _progress(i: int, total: int, item, pred):
    mark = "=" if pred.model_score == item.professor_score else "X"
    print(
        f"  [{i}/{total}] {item.exam_id}/{item.question_id} "
        f"({item.answer_type}) "
        f"prof={item.professor_score}/{item.max_points} "
        f"model={pred.model_score} conf={pred.model_confidence:.2f} [{mark}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3p5-35B-A3B")
    parser.add_argument("--items", type=int, default=8,
                        help="Number of items to grade (from first exam)")
    parser.add_argument("--base-url", default="http://192.168.68.128:8001")
    parser.add_argument("--all", action="store_true",
                        help="Grade all items (overrides --items)")
    parser.add_argument("--narrate", action="store_true",
                        help="Stream Project Paint Dry bonsai narration to stderr")
    parser.add_argument("--narrator-url", default="http://localhost:8001",
                        help="Bonsai narrator OMLX server URL")
    parser.add_argument("--narrator-model", default="Bonsai-8B-mlx-1bit")
    args = parser.parse_args()

    gt = load_ground_truth(_GROUND_TRUTH)
    subset = gt if args.all else gt[: args.items]

    config = ServerConfig(
        base_url=args.base_url,
        model=args.model,
    )

    print(f"Model: {config.model}")
    print(f"Items: {len(subset)} of {len(gt)}")
    print(f"Server: {config.base_url}")
    print(f"Scans: {_SCANS_DIR}")
    print()

    narrator_cm = (
        BonsaiNarrator(
            base_url=args.narrator_url,
            model=args.narrator_model,
        )
        if args.narrate
        else nullcontext()
    )

    t0 = time.time()
    with narrator_cm as narrator:
        predictions = grade_all_items(
            subset, _SCANS_DIR, config,
            template_path=_TEMPLATE,
            progress_callback=_progress,
            narrator=narrator if args.narrate else None,
        )
    elapsed = time.time() - t0

    print(f"\nInference: {elapsed:.1f}s ({elapsed / len(subset):.1f}s/item)")

    report = score_predictions(subset, predictions)

    print(f"\n{'='*60}")
    print(f"EVAL REPORT -- {config.model}")
    print(f"{'='*60}")
    print(f"Items scored:      {report.total_scored}")
    print(f"Unclear excluded:  {report.unclear_excluded}")
    print(f"Exact accuracy:    {report.overall_exact_accuracy:.1%}")
    print(f"+/-1 pt accuracy:  {report.overall_tolerance_accuracy:.1%}")
    print(f"False positives:   {report.false_positives}")
    print(f"False negatives:   {report.false_negatives}")
    print(f"Points possible:   {report.total_points_possible}")
    print(f"Points (professor):{report.total_points_professor}")

    print(f"\nPer answer type (exact):")
    for atype, acc in sorted(report.per_answer_type_exact.items()):
        print(f"  {atype:25s} {acc:.1%}")

    if report.calibration_bins:
        print(f"\nCalibration:")
        for b in report.calibration_bins:
            bar = "#" * int(b.accuracy * 20)
            print(
                f"  [{b.bin_start:.1f}-{b.bin_end:.1f}] "
                f"n={b.count:3d} conf={b.avg_confidence:.2f} "
                f"acc={b.accuracy:.2f} {bar}"
            )


if __name__ == "__main__":
    main()
