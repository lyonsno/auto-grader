"""Smoke test: run a small subset of eval items through the VLM pipeline.

Usage:
    uv run python scripts/smoke_vlm.py [--model MODEL] [--items N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from contextlib import nullcontext
from datetime import datetime

from auto_grader.eval_harness import (
    EvalItem,
    Prediction,
    load_ground_truth,
    score_predictions,
)
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

_DEFAULT_NARRATOR_URL = "http://nlmb2p.local:8002"

_TRICKY_PICKS = [
    ("15-blue", "fr-1"),    # easy warmup (numeric, density)
    ("15-blue", "fr-3"),    # FORMAT: full molecular vs net ionic, prof 0/4
    ("15-blue", "fr-5b"),   # CHARITY: consistent-with-wrong-premise
    ("15-blue", "fr-10a"),  # PARTIAL: prof gave 1.5/3 fractional
    ("15-blue", "fr-11a"),  # ELECTRON CONFIG: orbital boxes, visual
    ("15-blue", "fr-12a"),  # LEWIS: visual + partial credit
]

_TRICKY_PLUS_PICKS = [
    ("27-blue-2023", "fr-3"),    # clean correct net ionic
    ("27-blue-2023", "fr-5b"),   # clean correct stoichiometry numeric
    ("27-blue-2023", "fr-12a"),  # clean correct Lewis structure
    ("39-blue-redacted", "fr-10a"),  # clean correct frequency numeric
    ("34-blue", "fr-8"),         # partial numeric with confused work
    ("34-blue", "fr-12a"),       # Lewis partial with setup credit
    *_TRICKY_PICKS,
]


def _progress(i: int, total: int, item, pred):
    mark = "=" if pred.model_score == item.professor_score else "X"
    print(
        f"  [{i}/{total}] {item.exam_id}/{item.question_id} "
        f"({item.answer_type}) "
        f"prof={item.professor_score}/{item.max_points} "
        f"model={pred.model_score} conf={pred.model_confidence:.2f} [{mark}]"
    )


def _validate_narrator_model(model: str) -> str:
    normalized = model.strip()
    if normalized.rstrip("/").endswith("/snapshots"):
        raise ValueError(
            "--narrator-model must be the full snapshot model path, not the bare snapshots/ directory."
        )
    return normalized


def _scorebug_session_meta(
    *,
    args: argparse.Namespace,
    model: str,
    subset_count: int,
) -> dict[str, object]:
    if args.tricky:
        set_label = "TRICKY"
    elif args.tricky_plus:
        set_label = "TRICKY+"
    elif args.all:
        set_label = "ALL"
    elif args.pick:
        set_label = "PICK"
    else:
        set_label = f"FIRST {subset_count}"
    return {
        "model": model,
        "set_label": set_label,
        "subset_count": subset_count,
    }


class _PredictionWriter:
    """Append-only JSONL writer for grader predictions.

    One JSON object per line, written as each item finishes so an
    interrupt or crash mid-run still leaves a usable partial file. The
    object includes the parsed Prediction fields plus the verbatim
    reasoning trace for the post-hoc critic pass, and per-item ground
    truth so the file is self-contained for offline analysis.
    """

    def __init__(self, path: Path, *, model: str, run_dir: Path):
        self.path = path
        self._model = model
        self._run_dir = run_dir
        self._fh = None
        self._count = 0
        # Failure counter is part of the writer's public surface so
        # main() can check it after the with-block closes and surface
        # a loud banner + non-zero exit. Silent stderr noise was rated
        # material in the predictions-jsonl-persistence anaphora — for
        # a load-bearing file, persistence failures must reach the
        # operator, not vanish into the scrollback.
        self.failures = 0
        self.last_failure_msg: str | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", buffering=1)  # line-buffered
        # Header line — distinguishable by absence of "exam_id".
        header = {
            "type": "header",
            "model": self._model,
            "run_dir": str(self._run_dir),
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self._fh.write(json.dumps(header) + "\n")
        return self

    def write_one(self, item: EvalItem, pred: Prediction) -> None:
        if self._fh is None:
            self._record_failure("writer is closed")
            return
        try:
            record = {
                "type": "prediction",
                "exam_id": pred.exam_id,
                "question_id": pred.question_id,
                "answer_type": item.answer_type,
                "max_points": item.max_points,
                "professor_score": item.professor_score,
                "professor_mark": item.professor_mark,
                "student_answer": item.student_answer,
                "model_score": pred.model_score,
                "model_confidence": pred.model_confidence,
                "is_obviously_fully_correct": pred.is_obviously_fully_correct,
                "is_obviously_wrong": pred.is_obviously_wrong,
                "model_read": pred.model_read,
                "model_reasoning": pred.model_reasoning,
                "upstream_dependency": pred.upstream_dependency,
                "if_dependent_then_consistent": pred.if_dependent_then_consistent,
                "raw_assistant": pred.raw_assistant,
                "raw_reasoning": pred.raw_reasoning,
            }
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1
        except Exception as e:
            self._record_failure(str(e))

    def _record_failure(self, msg: str) -> None:
        self.failures += 1
        self.last_failure_msg = msg
        print(
            f"[predictions.jsonl write failed #{self.failures}] {msg}",
            file=sys.stderr,
        )

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                self._fh.write(
                    json.dumps(
                        {
                            "type": "footer",
                            "ended": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                            "count": self._count,
                            "interrupted": exc_type is KeyboardInterrupt,
                        }
                    )
                    + "\n"
                )
            finally:
                self._fh.close()
                self._fh = None
        return False  # never swallow exceptions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3p5-35B-A3B")
    parser.add_argument("--items", type=int, default=8,
                        help="Number of items to grade (from first exam)")
    parser.add_argument(
        "--base-url",
        default="http://macbook-pro-2.local:8001",
        help=(
            "OpenAI-compatible grader server. Defaults to mDNS so it "
            "follows the M4 Max's current LAN IP across DHCP renewals "
            "instead of pinning a stale dotted-quad."
        ),
    )
    parser.add_argument("--all", action="store_true",
                        help="Grade all items (overrides --items)")
    parser.add_argument(
        "--pick",
        default=None,
        help=(
            "Comma-separated <exam_id>:<question_id> pairs to grade "
            "(e.g. 15-blue:fr-1,15-blue:fr-5b). Overrides --items/--all."
        ),
    )
    parser.add_argument(
        "--tricky",
        action="store_true",
        help=(
            "Grade a curated set of known-tricky items: easy warmup + "
            "consistent-with-wrong-premise charity test + fractional "
            "partial credit + Lewis structure partial. Overrides --items."
        ),
    )
    parser.add_argument(
        "--tricky-plus",
        action="store_true",
        help=(
            "Grade the tricky regression sentinel plus a small expansion "
            "of clean-correct and partial-credit calibration items. "
            "Overrides --items."
        ),
    )
    parser.add_argument("--narrate", action="store_true",
                        help="Enable Project Paint Dry bonsai narrator (rich Terminal window + log files)")
    parser.add_argument("--narrate-stderr", action="store_true",
                        help="Plain-text narrator output to stderr (no Terminal window) — for dev")
    parser.add_argument("--narrator-url", default=_DEFAULT_NARRATOR_URL,
                        help="Bonsai narrator OMLX server URL")
    parser.add_argument("--narrator-model", default="Bonsai-8B-mlx-1bit")
    parser.add_argument(
        "--wrap-up-url",
        default=None,
        help=(
            "OpenAI-compatible server for the end-of-run wrap-up. "
            "Defaults to --base-url (the grader server) since the grader "
            "model is free by the time wrap-up fires and produces a more "
            "grounded post-game read than the small narrator model."
        ),
    )
    parser.add_argument(
        "--wrap-up-model",
        default=None,
        help="Model name for the wrap-up call. Defaults to --model.",
    )
    parser.add_argument("--run-dir", default=None,
                        help="Directory to write narrator.jsonl/.txt logs (default: runs/<ts>-<model>/)")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.narrate or args.narrate_stderr:
        args.narrator_model = _validate_narrator_model(args.narrator_model)

    gt = load_ground_truth(_GROUND_TRUTH)

    if args.pick:
        wanted = []
        for token in args.pick.split(","):
            token = token.strip()
            if not token or ":" not in token:
                continue
            exam_id, qid = token.split(":", 1)
            wanted.append((exam_id.strip(), qid.strip()))
        gt_index = {(item.exam_id, item.question_id): item for item in gt}
        subset = []
        missing = []
        for key in wanted:
            if key in gt_index:
                subset.append(gt_index[key])
            else:
                missing.append(key)
        if missing:
            print(
                f"WARNING: --pick missing in ground truth: {missing}",
                file=sys.stderr,
            )
    elif args.tricky:
        gt_index = {(item.exam_id, item.question_id): item for item in gt}
        subset = [gt_index[k] for k in _TRICKY_PICKS if k in gt_index]
    elif args.tricky_plus:
        gt_index = {(item.exam_id, item.question_id): item for item in gt}
        subset = [gt_index[k] for k in _TRICKY_PLUS_PICKS if k in gt_index]
    elif args.all:
        subset = gt
    else:
        subset = gt[: args.items]

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

    # Run dir is always created (even without narrator) so predictions
    # persist for the post-hoc critic and cross-run comparison reports.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model = config.model.replace("/", "_")
        run_dir = (
            Path(__file__).resolve().parent.parent
            / "runs" / f"{ts}-{safe_model}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    if narrator_enabled:
        sink_config = SinkConfig(
            spawn_terminal=args.narrate,
            log_dir=run_dir,
            fallback_stream=sys.stderr,
            session_meta=_scorebug_session_meta(
                args=args,
                model=config.model,
                subset_count=len(subset),
            ),
        )
        sink_cm = NarratorSink(sink_config)
    else:
        sink_cm = nullcontext()

    predictions_path = run_dir / "predictions.jsonl"
    pred_writer_cm = _PredictionWriter(
        predictions_path, model=config.model, run_dir=run_dir
    )

    t0 = time.time()
    narrator_stats = None
    predictions: list = []
    interrupted = False
    with sink_cm as sink, pred_writer_cm as pred_writer:
        # Wrap the progress callback so each completed prediction is
        # written to predictions.jsonl as it lands. Crash-safe: an
        # interrupt mid-loop still leaves the partial file usable.
        # Per-item failures are tracked on the writer (not raised) so
        # one bad item doesn't kill the whole grading run, but the
        # final pred_writer.failures count is checked after the
        # with-block and surfaced as a loud banner + non-zero exit.
        def _on_item(i, total, item, pred):
            pred_writer.write_one(item, pred)
            _progress(i, total, item, pred)

        narrator = (
            ThinkingNarrator(
                sink,
                base_url=args.narrator_url,
                model=args.narrator_model,
                wrap_up_base_url=args.wrap_up_url or args.base_url,
                wrap_up_model=args.wrap_up_model or args.model,
            )
            if narrator_enabled
            else None
        )
        try:
            predictions = grade_all_items(
                subset, _SCANS_DIR, config,
                template_path=_TEMPLATE,
                progress_callback=_on_item,
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

        # End-of-run wrap-up commentary from bonsai. Only fire if we have
        # at least one prediction and we weren't interrupted during
        # grading. Wrap-up itself is also interruptible via Ctrl-C —
        # if the user bails out of the wrap-up call, we skip it and
        # still print the eval report and tear down the sink cleanly.
        elapsed_for_wrap = time.time() - t0
        if (
            narrator is not None
            and predictions
            and not interrupted
        ):
            try:
                wrap_subset = subset[: len(predictions)]
                wrap_report = score_predictions(wrap_subset, predictions)
                narrator.wrap_up(
                    wrap_report,
                    model_name=config.model,
                    item_count=len(predictions),
                    elapsed_seconds=elapsed_for_wrap,
                )
            except KeyboardInterrupt:
                print(
                    "\n[wrap_up interrupted] skipped post-game commentary.",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[wrap_up failed] {e}", file=sys.stderr)

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
        print(f"  items started:        {narrator_stats['items_started']}")
        print(f"  dispatches total:     {narrator_stats['dispatches_total']}")
        print(f"  summaries emitted:    {narrator_stats['summaries_emitted']}")
        print(f"  drops (dedup):        {narrator_stats['drops_dedup']}")
        print(f"  drops (empty):        {narrator_stats['drops_empty']}")
        print(f"  max disp / one item:  {narrator_stats['max_dispatches_one_item']}")
        total_drops = (
            narrator_stats['drops_dedup'] + narrator_stats['drops_empty']
        )
        if narrator_stats['dispatches_total']:
            drop_rate = total_drops / narrator_stats['dispatches_total']
            print(f"  drop rate:            {drop_rate:.1%}")

    # Loud banner + non-zero exit if predictions.jsonl persistence had
    # any failures. The file is load-bearing for the post-hoc critic;
    # silent failure was rated material in the predictions-jsonl-
    # persistence anaphora and would let the critic operate on
    # incomplete data without anyone noticing.
    if pred_writer_cm.failures > 0:
        print(
            f"\n{'!' * 60}\n"
            f"PREDICTIONS PERSISTENCE FAILED on {pred_writer_cm.failures} item(s)\n"
            f"  file:        {pred_writer_cm.path}\n"
            f"  last error:  {pred_writer_cm.last_failure_msg}\n"
            f"  written:     {pred_writer_cm._count} of "
            f"{len(predictions)} predictions\n"
            f"The downstream critic pass will see incomplete data.\n"
            f"{'!' * 60}",
            file=sys.stderr,
        )
        return 2

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
