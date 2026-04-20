"""Smoke test: run a small subset of eval items through the VLM pipeline.

Usage:
    uv run python scripts/smoke_vlm.py [--model MODEL] [--items N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from contextlib import nullcontext
from datetime import datetime

from auto_grader.eval_harness import (
    EvalItem,
    FocusRegion,
    Prediction,
    load_ground_truth,
    resolve_focus_region,
    score_predictions,
)
from auto_grader.focus_preview import render_focus_preview
from auto_grader.focus_regions import (
    DEFAULT_FOCUS_REGIONS_PATH,
    load_focus_regions,
)
from auto_grader.narrator_sink import NarratorSink, SinkConfig
from auto_grader.thinking_narrator import ThinkingNarrator
from auto_grader.vlm_inference import (
    DESCRIBE_ONLY_PROMPT,
    ServerConfig,
    _EXAM_PDF_MAP,
    apply_model_sampling_preset,
    describe_prompt_metadata,
    extract_page_image,
    grade_all_items,
    grading_prompt_metadata,
    known_model_families,
    resolve_model_family,
    stream_vision_completion,
)
import yaml


_GROUND_TRUTH = Path(__file__).resolve().parent.parent / "eval" / "ground_truth.yaml"
_SCANS_DIR = Path.home() / "dev" / "auto-grader-assets" / "scans"
_DEFAULT_RUNS_ROOT = Path.home() / "dev" / "auto-grader-runs"
_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "templates"
    / "chm141-final-fall2023.yaml"
)

_DEFAULT_NARRATOR_URL = "http://nlm2pr.local:8002"

_TRICKY_PICKS = [
    ("15-blue", "fr-1"),    # easy warmup (numeric, density)
    ("15-blue", "fr-3"),    # FORMAT: full molecular vs net ionic, prof 0/4
    ("15-blue", "fr-5b"),   # CHARITY: consistent-with-wrong-premise
    ("15-blue", "fr-10a"),  # PARTIAL: prof gave 1.5/3 fractional
    ("15-blue", "fr-11a"),  # ELECTRON CONFIG: orbital boxes, visual
    ("15-blue", "fr-12a"),  # LEWIS: visual + partial credit
]
_TRICKY_TEST_SET_ID = "tricky-v1"

_TRICKY_PLUS_PICKS = [
    ("27-blue-2023", "fr-3"),    # clean correct net ionic
    ("27-blue-2023", "fr-5b"),   # clean correct stoichiometry numeric
    ("27-blue-2023", "fr-12a"),  # clean correct Lewis structure
    ("39-blue-redacted", "fr-10a"),  # clean correct frequency numeric
    ("34-blue", "fr-8"),         # partial numeric with confused work
    ("34-blue", "fr-12a"),       # Lewis partial with setup credit
    *_TRICKY_PICKS,
]
_TRICKY_PLUS_TEST_SET_ID = "tricky-plus-v1"

_TRICKY_PLUS_PLUS_PICKS = [
    ("15-blue", "fr-10b"),   # FOLLOW-ON: tiny numeric continuation after partial fr-10a
    ("15-blue", "fr-11c"),   # EXACTNESS: small orbital-box count / visual exact-match
    ("15-blue", "fr-12b"),   # RESONANCE: Lewis follow-on beyond basic structure
    *_TRICKY_PLUS_PICKS,
]
_TRICKY_PLUS_PLUS_TEST_SET_ID = "tricky-plus-plus-v1"


def _default_run_dir(model: str, *, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace("/", "_")
    return _DEFAULT_RUNS_ROOT / f"{stamp}-{safe_model}"


def _is_openrouter_base_url(base_url: str) -> bool:
    return "openrouter.ai" in base_url.lower()


def _resolve_api_key(base_url: str) -> str:
    if _is_openrouter_base_url(base_url):
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "--base-url openrouter.ai requires OPENROUTER_API_KEY in the "
                "environment. Set it and re-run."
            )
        return key
    return "1234"


def _describe_only_extra_body(base_url: str) -> dict[str, object] | None:
    if _is_openrouter_base_url(base_url):
        return {"reasoning": {"enabled": True}}
    return None


def _progress(i: int, total: int, item, pred):
    # Mark against truth_score (corrected when present, else professor_score)
    # so the live display agrees with score_predictions for corrected items.
    # Comparing against professor_score would make the live mark contradict
    # the eval report for items where a human-investigated correction has
    # been recorded.
    truth = item.truth_score
    # Truncated / unparseable rows are non-predictions — the mark is
    # neither "=" (match) nor "X" (miss); use "—" so the live display
    # stops lying about the model having scored zero. See Operation
    # Zilch Reaper (forward lane).
    if pred.truncated:
        mark = "—"
    else:
        mark = "=" if pred.model_score == truth else "X"
    # Display the truth baseline; when a correction is present, also show
    # the historical professor_score in parens so operators can see both
    # the original prof mark and the corrected value at a glance.
    if item.corrected_score is not None and item.corrected_score != item.professor_score:
        baseline = f"truth={truth}/{item.max_points} (prof={item.professor_score})"
    else:
        baseline = f"prof={item.professor_score}/{item.max_points}"
    if pred.truncated:
        model_str = "model=— conf=— (truncated)"
    else:
        model_str = (
            f"model={pred.model_score} conf={pred.model_confidence:.2f}"
        )
    print(
        f"  [{i}/{total}] {item.exam_id}/{item.question_id} "
        f"({item.answer_type}) "
        f"{baseline} "
        f"{model_str} [{mark}]"
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
    elif args.tricky_plus_plus:
        set_label = "TRICKY++"
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
        "scans_dir": str(_SCANS_DIR),
        "focus_regions_path": str(args.focus_regions or DEFAULT_FOCUS_REGIONS_PATH),
    }


def _load_template_document(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_preview_focus_region(
    item: EvalItem,
    *,
    template_document: dict | None,
    focus_region_overrides: dict[tuple[str, str], FocusRegion],
) -> FocusRegion | None:
    resolved = resolve_focus_region(item, template_document)
    if resolved is not None:
        return resolved
    return focus_region_overrides.get((item.exam_id, item.question_id))


def _emit_focus_preview(
    sink: NarratorSink,
    *,
    item: EvalItem,
    page_image: bytes,
    template_document: dict | None,
    focus_region_overrides: dict[tuple[str, str], FocusRegion],
) -> None:
    focus_region = _resolve_preview_focus_region(
        item,
        template_document=template_document,
        focus_region_overrides=focus_region_overrides,
    )
    if focus_region is None:
        return
    sink.write_focus_preview(
        render_focus_preview(page_image, focus_region),
        label=f"{item.exam_id}/{item.question_id}",
        source=focus_region.source,
    )


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
                # corrected_score and correction_reason preserve the
                # human-investigated truth alongside the historical prof
                # mark, so the prediction file is genuinely self-contained
                # for offline analysis (downstream tools like
                # compare_runs.py can compute truth_score without needing
                # the original ground_truth.yaml). Uncorrected items emit
                # explicit null / empty string rather than omitting the
                # fields, so readers have an unambiguous signal.
                "corrected_score": item.corrected_score,
                "correction_reason": item.correction_reason,
                # Acceptable-band telemetry is a second analysis surface,
                # not the primary truth target. Persist it alongside the
                # record so offline drift analysis can flag "within band"
                # vs. "too generous" / "too harsh" without reopening the
                # original ground_truth.yaml file.
                "acceptable_score_floor": item.acceptable_score_floor,
                "acceptable_score_ceiling": item.acceptable_score_ceiling,
                "acceptable_score_reason": item.acceptable_score_reason,
                "professor_mark": item.professor_mark,
                "student_answer": item.student_answer,
                # model_score and model_confidence serialize to JSON null
                # on truncated / unparseable rows (pred.model_score is
                # None). Operation Zilch Reaper (forward lane) — the old
                # "0.0 + magic substring in model_reasoning" shape is
                # retired. Consumers should check `truncated` (or
                # equivalently `model_score is None`) to distinguish
                # non-predictions from confident zeros.
                "model_score": pred.model_score,
                "model_confidence": pred.model_confidence,
                "truncated": pred.truncated,
                "is_obviously_fully_correct": pred.is_obviously_fully_correct,
                "is_obviously_wrong": pred.is_obviously_wrong,
                "model_read": pred.model_read,
                "score_basis": pred.score_basis,
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


class _RunManifest:
    """Persist run identity + lifecycle for later comparison tooling."""

    def __init__(self, path: Path, initial_data: dict[str, object]):
        self.path = path
        self._data = dict(initial_data)

    def write_status(
        self,
        status: str,
        *,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        self._data["status"] = status
        if finished_at is not None:
            self._data["finished_at"] = finished_at
        if error is not None:
            self._data["error"] = error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True) + "\n"
        )
        tmp_path.replace(self.path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _run_identity(model: str, run_dir_override: str | None) -> tuple[str, Path]:
    if run_dir_override:
        run_dir = Path(run_dir_override)
        return run_dir.name, run_dir

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace("/", "_")
    run_id = f"{ts}-{safe_model}"
    run_dir = _DEFAULT_RUNS_ROOT / run_id
    return run_id, run_dir


def _select_subset(
    args: argparse.Namespace,
    ground_truth: list[EvalItem],
) -> tuple[list[EvalItem], str]:
    if args.pick:
        wanted = []
        for token in args.pick.split(","):
            token = token.strip()
            if not token or ":" not in token:
                continue
            exam_id, qid = token.split(":", 1)
            wanted.append((exam_id.strip(), qid.strip()))
        gt_index = {
            (item.exam_id, item.question_id): item for item in ground_truth
        }
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
        normalized = ",".join(f"{exam}:{qid}" for exam, qid in wanted)
        return subset, f"pick-v1:{normalized}"

    gt_index = {
        (item.exam_id, item.question_id): item for item in ground_truth
    }
    if args.tricky_plus_plus:
        subset = [gt_index[k] for k in _TRICKY_PLUS_PLUS_PICKS if k in gt_index]
        return subset, _TRICKY_PLUS_PLUS_TEST_SET_ID
    if args.tricky_plus:
        subset = [gt_index[k] for k in _TRICKY_PLUS_PICKS if k in gt_index]
        return subset, _TRICKY_PLUS_TEST_SET_ID
    if args.tricky:
        subset = [gt_index[k] for k in _TRICKY_PICKS if k in gt_index]
        return subset, _TRICKY_TEST_SET_ID
    if args.all:
        return ground_truth, "all-v1"
    return ground_truth[: args.items], f"first-{args.items}-v1"


def _build_manifest(
    *,
    run_id: str,
    run_dir: Path,
    config: ServerConfig,
    args: argparse.Namespace,
    test_set_id: str,
    item_count: int,
) -> _RunManifest:
    prompt_meta = grading_prompt_metadata()
    manifest = _RunManifest(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "running",
            "started_at": _iso_now(),
            "finished_at": None,
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "model": config.model,
            "base_url": config.base_url,
            "prompt_version": prompt_meta["version"],
            "prompt_content_hash": prompt_meta["content_hash"],
            "test_set_id": test_set_id,
            "item_count": item_count,
            "narrator_url": (
                args.narrator_url
                if (args.narrate or args.narrate_stderr)
                else None
            ),
            "narrator_model": (
                args.narrator_model
                if (args.narrate or args.narrate_stderr)
                else None
            ),
        },
    )
    manifest.write_status("running")
    return manifest


def run_describe_only_mode(
    args: argparse.Namespace,
    subset: list[EvalItem],
    config: ServerConfig,
    *,
    model_family: str,
) -> dict[str, object]:
    """Describe selected pages without invoking the grading pipeline."""
    run_id, run_dir = _run_identity(config.model, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    probe_path = run_dir / "probe.jsonl"
    prompt_meta = describe_prompt_metadata()

    with open(probe_path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "header",
                    "mode": "describe-only",
                    "model": config.model,
                    "model_family": model_family,
                    "base_url": config.base_url,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "prompt_version": prompt_meta["version"],
                    "prompt_content_hash": prompt_meta["content_hash"],
                    "prompt": DESCRIBE_ONLY_PROMPT,
                    "started": _iso_now(),
                }
            )
            + "\n"
        )

    page_cache: dict[tuple[str, int], bytes] = {}
    n_ok = 0
    n_err = 0
    for i, item in enumerate(subset, start=1):
        cache_key = (item.exam_id, item.page)
        if cache_key not in page_cache:
            pdf_name = _EXAM_PDF_MAP.get(item.exam_id)
            if not pdf_name:
                msg = f"No PDF mapping for exam_id: {item.exam_id}"
                print(f"[{i}/{len(subset)}] {msg}", file=sys.stderr)
                n_err += 1
                continue
            pdf_path = _SCANS_DIR / pdf_name
            if not pdf_path.exists():
                msg = f"Scan PDF not found: {pdf_path}"
                print(f"[{i}/{len(subset)}] {msg}", file=sys.stderr)
                n_err += 1
                continue
            page_cache[cache_key] = extract_page_image(pdf_path, item.page)

        print(
            f"[{i}/{len(subset)}] {item.exam_id}:{item.question_id} "
            f"(page {item.page}) ...",
            flush=True,
        )
        row: dict[str, object] = {
            "type": "probe",
            "mode": "describe-only",
            "exam_id": item.exam_id,
            "question_id": item.question_id,
            "page": item.page,
            "student_answer_for_reference": item.student_answer,
            "model": config.model,
            "model_family": model_family,
        }
        try:
            t0 = time.time()
            content, reasoning = stream_vision_completion(
                config=config,
                prompt_text=DESCRIBE_ONLY_PROMPT,
                page_image=page_cache[cache_key],
                extra_body=_describe_only_extra_body(config.base_url),
                failure_context=f"{item.exam_id}/{item.question_id}",
            )
            elapsed = time.time() - t0
            row["description"] = content
            row["reasoning"] = reasoning
            row["elapsed_sec"] = round(elapsed, 2)
            row["error"] = None
            n_ok += 1
            preview = (content or "").strip().splitlines()[0:3]
            for line in preview:
                print(f"    {line[:140]}")
            print(
                f"    ({elapsed:.1f}s, {len(content)} chars content, "
                f"{len(reasoning)} chars reasoning)"
            )
        except Exception as e:
            row["description"] = ""
            row["reasoning"] = ""
            row["elapsed_sec"] = None
            row["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
            print(f"    {type(e).__name__}: {e}", file=sys.stderr)

        with open(probe_path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(probe_path, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "footer",
                    "ended": _iso_now(),
                    "count_ok": n_ok,
                    "count_err": n_err,
                }
            )
            + "\n"
        )

    print(f"Wrote {probe_path}")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "records_path": probe_path,
        "count_ok": n_ok,
        "count_err": n_err,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3p5-35B-A3B")
    parser.add_argument(
        "--model-family",
        choices=known_model_families(),
        default=None,
        help=(
            "Explicit sampling family override for unregistered models. "
            "Required when --model does not match a known family prefix."
        ),
    )
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
        "--describe-only",
        action="store_true",
        help=(
            "Run a bare perception smoke against the selected pages instead "
            "of the grading pipeline. Writes probe.jsonl in the run dir."
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
    parser.add_argument(
        "--tricky-plus-plus",
        action="store_true",
        help=(
            "Grade TRICKY+ plus three more 15-blue stress items up front "
            "for a denser 15-blue-heavy smoke. Overrides --items."
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
                        help="Directory to write narrator.jsonl/.txt logs (default: ~/dev/auto-grader-runs/<ts>-<model>/)")
    parser.add_argument(
        "--focus-regions",
        default=None,
        help=(
            "Path to a focus-regions YAML file. Default: "
            "eval/focus_regions.yaml. Override with a per-session path so "
            "concurrent sessions can work against independent copies "
            "without stomping each other."
        ),
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.narrate or args.narrate_stderr:
        args.narrator_model = _validate_narrator_model(args.narrator_model)

    gt = load_ground_truth(_GROUND_TRUTH)
    subset, test_set_id = _select_subset(args, gt)
    resolved_family = resolve_model_family(args.model, args.model_family)
    task = "describe" if args.describe_only else "grading"
    template_document = _load_template_document(_TEMPLATE)
    focus_region_overrides = load_focus_regions(
        args.focus_regions or DEFAULT_FOCUS_REGIONS_PATH
    )

    config = ServerConfig(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.base_url),
        model=args.model,
    )
    config = apply_model_sampling_preset(
        config,
        family=resolved_family,
        task=task,
    )

    print(f"Model: {config.model}")
    print(f"Family: {resolved_family}")
    print(f"Items: {len(subset)} of {len(gt)}")
    print(f"Server: {config.base_url}")
    print(
        f"Sampling: temp={config.temperature} top_p={config.top_p} "
        f"top_k={config.top_k} min_p={config.min_p} "
        f"presence={config.presence_penalty} rep={config.repetition_penalty}"
    )
    print(f"Scans: {_SCANS_DIR}")
    print()

    if args.describe_only:
        print("Mode: describe-only")
        result = run_describe_only_mode(
            args,
            subset,
            config,
            model_family=resolved_family,
        )
        return 0 if result["count_err"] == 0 else 1

    narrator_enabled = args.narrate or args.narrate_stderr

    # Run dir is always created (even without narrator) so predictions
    # persist for the post-hoc critic and cross-run comparison reports.
    run_id, run_dir = _run_identity(config.model, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    manifest = _build_manifest(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        args=args,
        test_set_id=test_set_id,
        item_count=len(subset),
    )

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
    try:
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
            focus_preview_callback = None
            if narrator_enabled and sink is not None:
                def focus_preview_callback(*, item, page_image, template_question=None):
                    _emit_focus_preview(
                        sink,
                        item=item,
                        page_image=page_image,
                        template_document=template_document,
                        focus_region_overrides=focus_region_overrides,
                    )
            try:
                predictions = grade_all_items(
                    subset, _SCANS_DIR, config,
                    template_path=_TEMPLATE,
                    progress_callback=_on_item,
                    narrator=narrator,
                    sink=sink,
                    focus_preview_callback=focus_preview_callback,
                )
            except KeyboardInterrupt:
                interrupted = True
                print(
                    f"\n\n[interrupted] Caught Ctrl-C — sent close to OMLX, "
                    f"completed {len(predictions)} of {len(subset)} items.",
                    file=sys.stderr,
                )

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
    except Exception as e:
        manifest.write_status(
            "failed",
            finished_at=_iso_now(),
            error=f"{type(e).__name__}: {e}",
        )
        raise
    elapsed = time.time() - t0

    if interrupted and not predictions:
        manifest.write_status("interrupted", finished_at=_iso_now())
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
    print(f"Points (truth):    {report.total_points_truth}")

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
        manifest.write_status(
            "failed",
            finished_at=_iso_now(),
            error=pred_writer_cm.last_failure_msg,
        )
        return 2

    manifest.write_status(
        "interrupted" if interrupted else "completed",
        finished_at=_iso_now(),
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
