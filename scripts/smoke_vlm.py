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
from datetime import datetime

from auto_grader.eval_harness import load_ground_truth, score_predictions
from auto_grader.narrator_sink import NarratorSink, SinkConfig
from auto_grader.thinking_narrator import ThinkingNarrator
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
                        help="Enable Project Paint Dry bonsai narrator (rich Terminal window + log files)")
    parser.add_argument("--narrate-stderr", action="store_true",
                        help="Plain-text narrator output to stderr (no Terminal window) — for dev")
    parser.add_argument("--narrator-url", default="http://localhost:8001",
                        help="Bonsai narrator OMLX server URL")
    parser.add_argument("--narrator-model", default="Bonsai-8B-mlx-1bit")
    parser.add_argument("--run-dir", default=None,
                        help="Directory to write narrator.jsonl/.txt logs (default: runs/<ts>-<model>/)")
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

    narrator_enabled = args.narrate or args.narrate_stderr

    if narrator_enabled:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_model = config.model.replace("/", "_")
            run_dir = (
                Path(__file__).resolve().parent.parent
                / "runs" / f"{ts}-{safe_model}"
            )
        sink_config = SinkConfig(
            spawn_terminal=args.narrate,
            log_dir=run_dir,
            fallback_stream=sys.stderr,
        )
        sink_cm = NarratorSink(sink_config)
        print(f"Narrator log dir: {run_dir}")
    else:
        sink_cm = nullcontext()

    t0 = time.time()
    narrator_stats = None
    predictions: list = []
    interrupted = False
    with sink_cm as sink:
        narrator = (
            ThinkingNarrator(
                sink,
                base_url=args.narrator_url,
                model=args.narrator_model,
            )
            if narrator_enabled
            else None
        )
        try:
            predictions = grade_all_items(
                subset, _SCANS_DIR, config,
                template_path=_TEMPLATE,
                progress_callback=_progress,
                narrator=narrator,
                sink=sink,
            )
        except KeyboardInterrupt:
            interrupted = True
            print(
                f"\n\n[interrupted] Caught Ctrl-C — sent close to OMLX, "
                f"completed {len(predictions)} of {len(subset)} items.",
                file=sys.stderr,
            )
        if narrator is not None:
            narrator_stats = narrator.stats()
    elapsed = time.time() - t0

    if interrupted and not predictions:
        print(
            "\nNo items completed before interrupt. Exiting without report.",
            file=sys.stderr,
        )
        return 130  # standard exit code for SIGINT

    # Trim subset to whatever actually completed (so the report aligns)
    if interrupted:
        subset = subset[: len(predictions)]

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

    if narrator_stats is not None:
        print(f"\nNarrator stats:")
        print(f"  items started:      {narrator_stats['items_started']}")
        print(f"  dispatches total:   {narrator_stats['dispatches_total']}")
        print(f"  summaries emitted:  {narrator_stats['summaries_emitted']}")
        print(f"  drops (dedup):      {narrator_stats['drops_dedup']}")
        print(f"  drops (empty):      {narrator_stats['drops_empty']}")
        print(f"  items hit cap:      {narrator_stats['drops_cap']}")
        total_drops = (
            narrator_stats['drops_dedup'] + narrator_stats['drops_empty']
        )
        if narrator_stats['dispatches_total']:
            drop_rate = total_drops / narrator_stats['dispatches_total']
            print(f"  drop rate:          {drop_rate:.1%}")

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
