"""Compare grader runs item-by-item across prompts or models.

Usage:
    uv run python scripts/compare_runs.py runs/run-a runs/run-b
    uv run python scripts/compare_runs.py --label old-qwen runs/qwen-old --label new-qwen runs/qwen-new
    uv run python scripts/compare_runs.py \
        --query model=gemma-4,prompt_version=2026-04-08-condensed-v1,test_set_id=tricky-v1 \
        --query model=qwen3p5-35B-A3B,prompt_version=2026-04-08-condensed-v1,test_set_id=tricky-v1

Writes a CSV with one row per (exam_id, question_id), plus per-run
columns for score, critic-adjusted score, confidence, declared
dependency, reasoning length, and narrator elapsed time when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_TOPIC_TIME_RE = re.compile(r"^(?P<seconds>\d+)s\s*·\s*Grader:")


@dataclass(frozen=True)
class RunRecord:
    model: str
    exam_id: str
    question_id: str
    professor_mark: str
    professor_score: float
    corrected_score: float | None
    acceptable_score_floor: float | None
    acceptable_score_ceiling: float | None
    acceptable_score_reason: str
    max_points: float
    answer_type: str
    # None on truncated / unparseable rows — see Operation Zilch Reaper
    # (forward lane). Legacy predictions.jsonl files written before the
    # sentinel contract still emit 0.0 here; the loader preserves that
    # historical shape as-is.
    model_score: float | None
    critic_score: float | None
    model_confidence: float | None
    upstream_dependency: str
    if_dependent_then_consistent: bool | None
    reasoning_chars: int
    elapsed_s: int | None
    # Truncation flag. False for complete rows AND for legacy rows
    # written before the sentinel contract existed — legacy files
    # simply did not carry this information. The historical rewriter
    # (Operation Zilch Reaper, historical lane) is the place where
    # legacy rows get their real truncation status restored.
    truncated: bool = False

    @property
    def truth_score(self) -> float:
        """The score we believe is correct after any human investigation.

        Mirrors `EvalItem.truth_score` from `auto_grader.eval_harness`:
        returns `corrected_score` when set (human-investigated prof
        grading error), otherwise falls back to `professor_score`. Use
        this everywhere the comparison surface needs the corrected
        baseline, so that the same run viewed through compare_runs.py
        and through eval_harness.py reports the same numbers.
        """
        return (
            self.corrected_score
            if self.corrected_score is not None
            else self.professor_score
        )


@dataclass(frozen=True)
class RunManifest:
    run_dir: Path
    run_id: str
    status: str
    started_at: str
    model: str
    prompt_version: str
    test_set_id: str
    raw: dict[str, object]


def _has_value(value: object) -> bool:
    return value is not None and value != ""


def _resolve_shared_row_value(
    *,
    exam_id: str,
    question_id: str,
    field_name: str,
    records: list[RunRecord],
) -> object:
    values = [
        getattr(record, field_name)
        for record in records
        if _has_value(getattr(record, field_name))
    ]
    if not values:
        return getattr(records[0], field_name)
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError(
                f"{exam_id}/{question_id}: conflicting {field_name} values "
                f"across compared runs ({first!r} vs {value!r})"
            )
    return first


def _resolve_predictions_path(run_dir: Path) -> Path:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no predictions.jsonl")
    return predictions_path


def _load_elapsed_times(run_dir: Path) -> dict[tuple[str, str], int]:
    narrator_path = run_dir / "narrator.jsonl"
    if not narrator_path.is_file():
        return {}

    current_item: tuple[str, str] | None = None
    elapsed: dict[tuple[str, str], int] = {}
    with open(narrator_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            typ = obj.get("type")
            text = str(obj.get("text", ""))
            if typ == "header":
                # Example: [item 2/6] 15-blue/fr-3 (balanced_equation, 4.0 pts)
                if "] " in text and " (" in text:
                    after = text.split("] ", 1)[1]
                    item_id = after.split(" (", 1)[0]
                    if "/" in item_id:
                        exam_id, question_id = item_id.split("/", 1)
                        current_item = (exam_id, question_id)
            elif typ == "topic" and current_item is not None:
                match = _TOPIC_TIME_RE.match(text)
                if match:
                    elapsed[current_item] = int(match.group("seconds"))
    return elapsed


def _load_critic_scores(run_dir: Path) -> dict[tuple[str, str], float]:
    critic_path = run_dir / "critic.jsonl"
    if not critic_path.is_file():
        return {}

    scores: dict[tuple[str, str], float] = {}
    with open(critic_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "delta":
                continue
            key = (obj["exam_id"], obj["question_id"])
            scores[key] = float(obj["new_score"])
    return scores


def load_run_records(run_dir: Path) -> dict[tuple[str, str], RunRecord]:
    run_dir = Path(run_dir)
    predictions_path = _resolve_predictions_path(run_dir)
    critic_scores = _load_critic_scores(run_dir)
    elapsed_times = _load_elapsed_times(run_dir)

    model = ""
    records: dict[tuple[str, str], RunRecord] = {}
    with open(predictions_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            typ = obj.get("type")
            if typ == "header":
                model = str(obj.get("model", ""))
                continue
            if typ != "prediction":
                continue
            key = (obj["exam_id"], obj["question_id"])
            raw_reasoning = str(obj.get("raw_reasoning", ""))
            # corrected_score defaults to None for backwards compat with
            # prediction files written before the self-contained-record
            # change. Old files simply produce truth_score == professor_score
            # via the RunRecord.truth_score property fallback.
            corrected_raw = obj.get("corrected_score")
            corrected_score = (
                float(corrected_raw) if corrected_raw is not None else None
            )
            acceptable_floor_raw = obj.get("acceptable_score_floor")
            acceptable_ceiling_raw = obj.get("acceptable_score_ceiling")
            # Truncation sentinel fields. model_score and model_confidence
            # may be JSON null on predictions.jsonl files written by the
            # post-Zilch-Reaper grader. Legacy files from before the
            # sentinel contract still emit numeric 0.0 and lack the
            # truncated key entirely — those round-trip as truncated=False.
            raw_model_score = obj.get("model_score")
            raw_model_confidence = obj.get("model_confidence")
            records[key] = RunRecord(
                model=model,
                exam_id=obj["exam_id"],
                question_id=obj["question_id"],
                professor_mark=str(obj.get("professor_mark", "")),
                professor_score=float(obj["professor_score"]),
                corrected_score=corrected_score,
                acceptable_score_floor=(
                    float(acceptable_floor_raw)
                    if acceptable_floor_raw is not None
                    else None
                ),
                acceptable_score_ceiling=(
                    float(acceptable_ceiling_raw)
                    if acceptable_ceiling_raw is not None
                    else None
                ),
                acceptable_score_reason=str(
                    obj.get("acceptable_score_reason", "")
                ),
                max_points=float(obj["max_points"]),
                answer_type=str(obj["answer_type"]),
                model_score=(
                    float(raw_model_score)
                    if raw_model_score is not None
                    else None
                ),
                critic_score=critic_scores.get(key),
                model_confidence=(
                    float(raw_model_confidence)
                    if raw_model_confidence is not None
                    else None
                ),
                upstream_dependency=str(obj.get("upstream_dependency", "none")),
                if_dependent_then_consistent=obj.get(
                    "if_dependent_then_consistent", None
                ),
                reasoning_chars=len(raw_reasoning),
                elapsed_s=elapsed_times.get(key),
                truncated=bool(obj.get("truncated", False)),
            )
    return records


def build_comparison_rows(
    labeled_runs: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    loaded = [(label, load_run_records(path)) for label, path in labeled_runs]
    all_keys = sorted({key for _, records in loaded for key in records})

    rows: list[dict[str, object]] = []
    for exam_id, question_id in all_keys:
        present_records = [
            records[(exam_id, question_id)]
            for _, records in loaded
            if (exam_id, question_id) in records
        ]
        row: dict[str, object] = {
            "exam_id": exam_id,
            "question_id": question_id,
            "answer_type": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="answer_type",
                records=present_records,
            ),
            "professor_mark": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="professor_mark",
                records=present_records,
            ),
            "max_points": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="max_points",
                records=present_records,
            ),
            # professor_score is the historical prof mark; truth_score
            # is the corrected baseline (equal to professor_score when
            # no correction is recorded). Both columns are emitted so
            # operators can see the historical distinction and still
            # get the corrected accuracy baseline in the same CSV.
            "professor_score": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="professor_score",
                records=present_records,
            ),
            "truth_score": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="truth_score",
                records=present_records,
            ),
            "acceptable_score_floor": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="acceptable_score_floor",
                records=present_records,
            ),
            "acceptable_score_ceiling": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="acceptable_score_ceiling",
                records=present_records,
            ),
            "acceptable_score_reason": _resolve_shared_row_value(
                exam_id=exam_id,
                question_id=question_id,
                field_name="acceptable_score_reason",
                records=present_records,
            ),
        }
        for label, records in loaded:
            record = records.get((exam_id, question_id))
            prefix = f"{label}__"
            if record is None:
                row[prefix + "present"] = False
                continue
            row[prefix + "present"] = True
            row[prefix + "model"] = record.model
            row[prefix + "score"] = record.model_score
            row[prefix + "truncated"] = record.truncated
            row[prefix + "critic_score"] = (
                record.critic_score
                if record.critic_score is not None
                else record.model_score
            )
            row[prefix + "confidence"] = record.model_confidence
            row[prefix + "upstream_dependency"] = record.upstream_dependency
            row[prefix + "if_dependent_then_consistent"] = (
                record.if_dependent_then_consistent
            )
            row[prefix + "reasoning_chars"] = record.reasoning_chars
            row[prefix + "elapsed_s"] = record.elapsed_s
        rows.append(row)
    return rows


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _default_out_path(labels: list[str]) -> Path:
    joined = "__vs__".join(labels)
    return Path("runs") / "comparisons" / f"{joined}.csv"


def _default_runs_root() -> Path:
    return Path.home() / "dev" / "auto-grader-runs"


def _parse_query(text: str) -> dict[str, str]:
    selector: dict[str, str] = {}
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"invalid query selector {part!r}; expected key=value"
            )
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"invalid query selector {part!r}; expected non-empty key=value"
            )
        selector[key] = value
    if not selector:
        raise ValueError("query selector must contain at least one key=value pair")
    return selector


def _load_manifest(manifest_path: Path) -> RunManifest:
    raw = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    return RunManifest(
        run_dir=run_dir,
        run_id=str(raw.get("run_id", run_dir.name)),
        status=str(raw.get("status", "unknown")),
        started_at=str(raw.get("started_at", "")),
        model=str(raw.get("model", "")),
        prompt_version=str(raw.get("prompt_version", "")),
        test_set_id=str(raw.get("test_set_id", "")),
        raw=raw,
    )


def _discover_manifests(runs_root: Path) -> list[RunManifest]:
    manifests: list[RunManifest] = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifests.append(_load_manifest(manifest_path))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path} is not valid JSON") from exc
    return manifests


def _matches_query(manifest: RunManifest, selector: dict[str, str]) -> bool:
    for key, value in selector.items():
        candidate = manifest.raw.get(key)
        if candidate is None:
            return False
        if str(candidate) != value:
            return False
    return True


def _selector_label(selector: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in selector.items())


def resolve_query_run(
    *,
    runs_root: Path,
    query: str,
) -> Path:
    selector = _parse_query(query)
    matches = [
        manifest
        for manifest in _discover_manifests(runs_root)
        if manifest.status == "completed" and _matches_query(manifest, selector)
    ]
    if not matches:
        raise FileNotFoundError(
            f"no completed run matched query {query!r} under {runs_root}"
        )
    latest = max(
        matches,
        key=lambda manifest: (manifest.started_at, manifest.run_dir.name),
    )
    return latest.run_dir


def resolve_labeled_runs(
    *,
    run_args: list[str],
    query_args: list[str],
    label_args: list[str],
    runs_root: Path,
) -> list[tuple[str, Path]]:
    if run_args and query_args:
        raise ValueError(
            "cannot mix direct run paths with --query until the CLI preserves operator order"
        )

    resolved_paths = [Path(arg) for arg in run_args]
    resolved_paths.extend(
        resolve_query_run(runs_root=runs_root, query=query)
        for query in query_args
    )
    if not resolved_paths:
        raise ValueError("must provide at least one run path or --query selector")

    if label_args and len(label_args) != len(resolved_paths):
        raise ValueError(
            "--label must be provided exactly once per run or query, in order"
        )

    default_labels = [path.name for path in map(Path, run_args)]
    default_labels.extend(
        _selector_label(_parse_query(query)) for query in query_args
    )
    labels = label_args or default_labels
    return list(zip(labels, resolved_paths, strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare run artifacts item-by-item across prompts or models. "
            "Each positional arg is a run directory containing predictions.jsonl, "
            "or use --query to resolve the latest completed run by manifest metadata."
        )
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run directories to compare (cannot be mixed with --query in the same invocation)",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help=(
            "Select the latest completed run whose manifest matches a "
            "comma-separated key=value selector, e.g. "
            "model=gemma-4,prompt_version=2026-04-08-condensed-v1,test_set_id=tricky-v1. "
            "Cannot be mixed with direct run-path args in the same invocation."
        ),
    )
    parser.add_argument(
        "--runs-root",
        default=str(_default_runs_root()),
        help=(
            "Durable runs root to scan for manifest-backed query resolution "
            f"(default: {_default_runs_root()})"
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help=(
            "Optional label for a run. Supply once per positional run or query, "
            "in the same order. Defaults to the run directory basename or query selector."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: runs/comparisons/<labels>.csv)",
    )
    args = parser.parse_args()

    try:
        labeled_runs = resolve_labeled_runs(
            run_args=args.runs,
            query_args=args.query,
            label_args=args.label,
            runs_root=Path(args.runs_root),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    labels = [label for label, _ in labeled_runs]
    rows = build_comparison_rows(
        [(label, path) for label, path in labeled_runs]
    )
    out_path = Path(args.out) if args.out else _default_out_path(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    print(f"Rows: {len(rows)}")
    for label, path in labeled_runs:
        records = load_run_records(path)
        elapsed_values = [
            record.elapsed_s for record in records.values() if record.elapsed_s is not None
        ]
        avg_elapsed = (
            sum(elapsed_values) / len(elapsed_values)
            if elapsed_values
            else None
        )
        if avg_elapsed is None:
            print(f"  {label}: {len(records)} items")
        else:
            print(
                f"  {label}: {len(records)} items, avg elapsed {avg_elapsed:.1f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
