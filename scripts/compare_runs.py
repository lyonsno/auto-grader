"""Compare grader runs item-by-item across prompts or models.

Usage:
    uv run python scripts/compare_runs.py runs/run-a runs/run-b
    uv run python scripts/compare_runs.py --label old-qwen runs/qwen-old --label new-qwen runs/qwen-new

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
    professor_score: float
    max_points: float
    answer_type: str
    model_score: float
    critic_score: float | None
    model_confidence: float
    is_obviously_fully_correct: bool | None
    is_obviously_wrong: bool | None
    upstream_dependency: str
    if_dependent_then_consistent: bool | None
    reasoning_chars: int
    elapsed_s: int | None


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
            records[key] = RunRecord(
                model=model,
                exam_id=obj["exam_id"],
                question_id=obj["question_id"],
                professor_score=float(obj["professor_score"]),
                max_points=float(obj["max_points"]),
                answer_type=str(obj["answer_type"]),
                model_score=float(obj["model_score"]),
                critic_score=critic_scores.get(key),
                model_confidence=float(obj.get("model_confidence", 0.0)),
                is_obviously_fully_correct=obj.get(
                    "is_obviously_fully_correct", None
                ),
                is_obviously_wrong=obj.get("is_obviously_wrong", None),
                upstream_dependency=str(obj.get("upstream_dependency", "none")),
                if_dependent_then_consistent=obj.get(
                    "if_dependent_then_consistent", None
                ),
                reasoning_chars=len(raw_reasoning),
                elapsed_s=elapsed_times.get(key),
            )
    return records


def build_comparison_rows(
    labeled_runs: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    loaded = [(label, load_run_records(path)) for label, path in labeled_runs]
    all_keys = sorted({key for _, records in loaded for key in records})

    rows: list[dict[str, object]] = []
    for exam_id, question_id in all_keys:
        exemplar = next(
            records[(exam_id, question_id)]
            for _, records in loaded
            if (exam_id, question_id) in records
        )
        row: dict[str, object] = {
            "exam_id": exam_id,
            "question_id": question_id,
            "answer_type": exemplar.answer_type,
            "max_points": exemplar.max_points,
            "professor_score": exemplar.professor_score,
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
            row[prefix + "critic_score"] = (
                record.critic_score
                if record.critic_score is not None
                else record.model_score
            )
            row[prefix + "confidence"] = record.model_confidence
            row[prefix + "is_obviously_fully_correct"] = (
                record.is_obviously_fully_correct
            )
            row[prefix + "is_obviously_wrong"] = record.is_obviously_wrong
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare run artifacts item-by-item across prompts or models. "
            "Each positional arg is a run directory containing predictions.jsonl."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories to compare",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help=(
            "Optional label for a run. Supply once per positional run, "
            "in the same order. Defaults to the run directory basename."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: runs/comparisons/<labels>.csv)",
    )
    args = parser.parse_args()

    run_paths = [Path(arg) for arg in args.runs]
    if args.label and len(args.label) != len(run_paths):
        print(
            "error: --label must be provided exactly once per run, in order",
            file=sys.stderr,
        )
        return 1

    labels = args.label or [path.name for path in run_paths]
    labeled_runs = list(zip(labels, run_paths, strict=True))
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
