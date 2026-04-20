"""Backfill corrected truth metadata into historical predictions archives.

Archived ``predictions.jsonl`` files are self-contained: comparison and
analysis surfaces read ``corrected_score`` / ``correction_reason`` from each
run artifact rather than reopening current ``eval/ground_truth.yaml``. When
human investigation later discovers a professor grading error, new runs pick up
that corrected truth automatically, but old archived runs stay stale until the
archive is rewritten.

This script performs that historical repair pass.

Safety posture mirrors the historical Zilch Reaper truncation cull:

* Dry-run by default.
* ``--commit`` required to touch disk.
* Commit mode writes a ``.bak`` sibling before rewriting.
* Files without a ``type: "footer"`` row are skipped defensively.
* Rewrites are idempotent and preserve all unrelated data verbatim.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger("backfill_correction_history")


class GroundTruthLoadError(ValueError):
    """Raised when the supplied ground-truth YAML cannot be loaded safely."""


@dataclass
class BackfillReport:
    path: Path
    rewritten: int = 0
    preserved: int = 0
    skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    committed: bool = False
    backup_path: Path | None = None


def load_corrections(
    yaml_path: Path,
) -> dict[tuple[str, str], dict[str, float | str]]:
    """Load only the items with human-verified corrected truth."""

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise GroundTruthLoadError(
            f"ground truth file does not exist: {yaml_path}"
        )
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GroundTruthLoadError(
            f"ground truth YAML is malformed: {yaml_path}"
        ) from exc
    except OSError as exc:
        raise GroundTruthLoadError(
            f"ground truth file could not be read: {yaml_path} ({exc})"
        ) from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise GroundTruthLoadError(
            f"ground truth root must be a mapping: {yaml_path}"
        )
    corrections: dict[tuple[str, str], dict[str, float | str]] = {}
    for exam in raw.get("exams", []):
        exam_id = str(exam["exam_id"])
        for item in exam.get("items", []):
            corrected_raw = item.get("corrected_score")
            if corrected_raw is None:
                continue
            corrections[(exam_id, str(item["question_id"]))] = {
                "corrected_score": float(corrected_raw),
                "correction_reason": str(item.get("correction_reason", "")),
            }
    return corrections


def rewrite_prediction_row(
    row: dict[str, Any],
    corrections: dict[tuple[str, str], dict[str, float | str]],
) -> dict[str, Any]:
    """Return a corrected copy of a prediction row.

    Pure function. If the row's ``(exam_id, question_id)`` is not present in
    the supplied corrections map, the returned copy is semantically identical
    to the input.
    """

    rewritten = dict(row)
    key = (str(row.get("exam_id", "")), str(row.get("question_id", "")))
    correction = corrections.get(key)
    if correction is None:
        return rewritten
    rewritten["corrected_score"] = correction["corrected_score"]
    rewritten["correction_reason"] = correction["correction_reason"]
    return rewritten


def _process_line(
    lineno: int,
    raw: str,
    *,
    corrections: dict[tuple[str, str], dict[str, float | str]],
    report: BackfillReport,
) -> tuple[str, bool]:
    stripped = raw.rstrip("\n")
    if not stripped.strip():
        report.preserved += 1
        return stripped, False

    try:
        row = json.loads(stripped)
    except json.JSONDecodeError as exc:
        report.skipped += 1
        report.skip_reasons.append(f"line {lineno}: malformed JSON ({exc.msg})")
        return stripped, False

    if not isinstance(row, dict):
        report.skipped += 1
        report.skip_reasons.append(
            f"line {lineno}: top-level JSON value is not an object"
        )
        return stripped, False

    row_type = row.get("type")
    if row_type == "footer":
        report.preserved += 1
        return stripped, True
    if row_type == "header":
        report.preserved += 1
        return stripped, False
    if row_type != "prediction":
        report.skipped += 1
        report.skip_reasons.append(f"line {lineno}: unknown row type {row_type!r}")
        return stripped, False

    rewritten = rewrite_prediction_row(row, corrections)
    if rewritten != row:
        report.rewritten += 1
        return json.dumps(rewritten), False

    report.preserved += 1
    return stripped, False


def _classify_and_process(
    rows: list[tuple[int, str]],
    *,
    corrections: dict[tuple[str, str], dict[str, float | str]],
) -> tuple[list[str], BackfillReport, bool]:
    report = BackfillReport(path=Path())
    out_lines: list[str] = []
    saw_footer = False

    for lineno, raw in rows:
        rendered, line_has_footer = _process_line(
            lineno,
            raw,
            corrections=corrections,
            report=report,
        )
        saw_footer = saw_footer or line_has_footer
        out_lines.append(rendered)

    return out_lines, report, saw_footer


def backfill_file(
    path: Path,
    *,
    corrections: dict[tuple[str, str], dict[str, float | str]],
    commit: bool,
) -> BackfillReport:
    """Backfill a single ``predictions.jsonl`` file."""

    path = Path(path)
    report = BackfillReport(path=path)
    saw_footer = False
    tmp_path = path.with_name(path.name + ".tmp")

    def _consume(writer: Any | None) -> None:
        nonlocal saw_footer
        with path.open("r", encoding="utf-8") as src:
            for lineno, raw in enumerate(src, start=1):
                rendered, line_has_footer = _process_line(
                    lineno,
                    raw,
                    corrections=corrections,
                    report=report,
                )
                saw_footer = saw_footer or line_has_footer
                if writer is None:
                    continue
                writer.write(rendered)
                if raw.endswith("\n"):
                    writer.write("\n")

    try:
        if commit:
            with tmp_path.open("w", encoding="utf-8") as tmp:
                _consume(tmp)
        else:
            _consume(None)
    finally:
        if (
            not commit
            and tmp_path.exists()
        ):
            tmp_path.unlink()

    if not saw_footer:
        report.skipped += 1
        report.skip_reasons.append(
            "file has no type: footer row (mid-flight run or crashed before close); refusing to rewrite"
        )
        report.rewritten = 0
        report.committed = False
        if tmp_path.exists():
            tmp_path.unlink()
        for reason in report.skip_reasons:
            logger.warning("%s: %s", path, reason)
        return report

    for reason in report.skip_reasons:
        logger.warning("%s: %s", path, reason)

    if report.rewritten == 0 or not commit:
        report.committed = False
        if tmp_path.exists():
            tmp_path.unlink()
        return report

    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copyfile(path, backup)
    report.backup_path = backup

    os.replace(tmp_path, path)
    report.committed = True
    return report


def backfill_archive(
    root: Path,
    *,
    corrections: dict[tuple[str, str], dict[str, float | str]],
    commit: bool,
) -> list[BackfillReport]:
    """Walk a runs root and backfill every ``predictions.jsonl`` found."""

    reports: list[BackfillReport] = []
    root = Path(root)
    if not root.exists():
        logger.warning("archive root does not exist: %s", root)
        return reports

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        predictions = run_dir / "predictions.jsonl"
        if not predictions.exists():
            continue
        reports.append(
            backfill_file(predictions, corrections=corrections, commit=commit)
        )
    return reports


def _format_report_summary(
    reports: list[BackfillReport],
    *,
    commit: bool,
) -> str:
    header = "COMMITTED" if commit else "DRY-RUN"
    total_files = len(reports)
    affected_files = sum(1 for r in reports if r.rewritten > 0)
    total_rewritten = sum(r.rewritten for r in reports)
    total_preserved = sum(r.preserved for r in reports)
    total_skipped = sum(r.skipped for r in reports)

    lines = [
        f"Historical correction backfill — {header}",
        f"  files inspected: {total_files}",
        f"  files with updates: {affected_files}",
        f"  rows rewritten: {total_rewritten}",
        f"  rows preserved: {total_preserved}",
        f"  rows skipped (logged):    {total_skipped}",
    ]
    if affected_files:
        lines.append("")
        lines.append("Per-file breakdown (affected files only):")
        for report in reports:
            if report.rewritten == 0:
                continue
            lines.append(
                f"  {report.path}: +{report.rewritten} rewritten"
                + (
                    f" -> backup at {report.backup_path}"
                    if commit and report.backup_path is not None
                    else ""
                )
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill current corrected truth metadata into historical "
            "auto-grader run archives."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="~/dev/auto-grader-runs",
        help=(
            "Archive root to walk. Each immediate subdirectory is treated "
            "as a run directory containing predictions.jsonl. "
            "Default: ~/dev/auto-grader-runs."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        default=str(
            Path(__file__).resolve().parent.parent / "eval" / "ground_truth.yaml"
        ),
        help=(
            "Ground truth YAML to read corrections from. Default: repo eval/ground_truth.yaml."
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually rewrite files. Default is dry-run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level diagnostics as well as the summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    try:
        corrections = load_corrections(Path(args.ground_truth).expanduser())
    except GroundTruthLoadError as exc:
        logger.error("could not load ground truth corrections: %s", exc)
        return 2
    reports = backfill_archive(
        Path(args.root).expanduser(),
        corrections=corrections,
        commit=args.commit,
    )
    print(_format_report_summary(reports, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
