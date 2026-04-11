"""Operation Zilch Reaper — historical lane.

Rewrite truncation-as-zero rows in the existing ``auto-grader-runs/``
archive into the settled sentinel shape::

    model_score: null
    model_confidence: null
    truncated: true

The sentinel shape was decided by the human on 2026-04-11 and is jointly
owned with the companion forward-fix lane (``Zilch Reaper Forward``). The
attractor at
``attractors/auto-grader_zilch-reaper-historical_cull-truncation-as-zero-rows-from-existing-run-archive_2026-04-11.md``
has the full narrative.

## Archive safety

The archive at ``~/dev/auto-grader-runs/`` is irreplaceable — each run
represents ~15-20 minutes of eval time plus manual kick-off overhead.
This script defaults to dry-run. ``--commit`` is required to write
anything back, and committing writes a ``.bak`` sibling alongside each
rewritten file so the cull is reversible.

## Idempotency

The rewriter detects "already migrated" rows by the presence of the
``truncated: true`` flag, not by the absence of the old sentinel string
on ``model_reasoning``. This is intentional: the companion forward-fix
lane may start emitting the new shape during the overlap window before
it lands on main, and we want this rewriter to be a no-op on those rows
rather than re-stamping them.

## Non-goals

- Does NOT re-grade truncated items. We only fix the record to say "no
  prediction" instead of falsely saying "prediction was 0."
- Does NOT touch ``model_reasoning`` on non-truncated rows. In
  particular, the fabricated-rule citation rows flagged by the earlier
  sub-agent audit are *correct records* of a real failure mode and must
  be preserved verbatim.
- Does NOT modify any forward-path grader code. That work lives on the
  ``Zilch Reaper Forward`` lane.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger("cull_zilch_reaper")


# The exact sentinel string the grader has been writing for the entire
# archive's lifetime. Detection tolerates case drift and missing trailing
# punctuation but otherwise requires a direct substring match against this
# normalized form, so we do not accidentally rewrite rows whose
# ``model_reasoning`` merely contains the word "truncated" in some other
# context.
_TRUNCATION_SENTINEL = (
    "Grader output was truncated at the max token limit "
    "before it finished the required JSON."
)


def _normalize_reasoning(value: Any) -> str:
    """Normalize ``model_reasoning`` for tolerant sentinel matching.

    Lowercases, strips surrounding whitespace, and collapses internal
    whitespace runs (including newlines) so that a sentinel that drifted
    across a line break or picked up extra spaces still matches.
    Non-string inputs return an empty string.
    """

    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


_NORMALIZED_SENTINEL = _normalize_reasoning(_TRUNCATION_SENTINEL)
# Sentinel with its trailing period stripped, so a drifted version that
# lost the final punctuation still matches.
_NORMALIZED_SENTINEL_NO_TRAILING_PUNCT = _NORMALIZED_SENTINEL.rstrip(".")


def is_truncated_row(row: dict) -> bool:
    """Return True iff this row is a truncation-as-zero corruption.

    A row is considered truncated iff all of the following hold:

    1. It is a prediction row (``type == "prediction"``).
    2. It does NOT already carry ``truncated: true`` (so already-migrated
       rows are not re-flagged, which is what makes the rewriter
       idempotent).
    3. Its ``model_reasoning`` field, normalized for case and
       whitespace, is **exactly** the canonical truncation sentinel,
       with or without the trailing period. Substring containment is
       intentionally NOT accepted — a legitimate prediction row whose
       reasoning happens to quote the sentinel inside longer prose
       must not be flagged. The parser-default truncation row that
       this rewriter is fixing always has ``model_reasoning`` set to
       exactly the sentinel and nothing else; a full-archive audit on
       2026-04-11 confirmed 71/71 real truncation rows are the exact
       string. Tightening to equality closes a false-positive path
       without dropping any real row.

    Non-prediction rows (headers, footers) and prediction rows that
    happen to score zero for legitimate reasons are not flagged.
    """

    if not isinstance(row, dict):
        return False
    if row.get("type") != "prediction":
        return False
    if row.get("truncated") is True:
        return False

    normalized = _normalize_reasoning(row.get("model_reasoning"))
    if not normalized:
        return False

    return (
        normalized == _NORMALIZED_SENTINEL
        or normalized == _NORMALIZED_SENTINEL_NO_TRAILING_PUNCT
    )


def rewrite_truncated_row(row: dict) -> dict:
    """Return a new row in the settled sentinel shape.

    Pure: does not mutate the input. Preserves every field of the input
    verbatim except for ``model_score`` and ``model_confidence``, which
    are forced to ``None``, and adds a ``truncated: True`` flag.

    Callers should gate this on ``is_truncated_row`` — the function does
    not itself check whether the input looks truncated, so it can be
    reused for unit tests and for repair passes that already know they
    are handling a corrupted row.
    """

    rewritten = dict(row)
    rewritten["model_score"] = None
    rewritten["model_confidence"] = None
    rewritten["truncated"] = True
    return rewritten


@dataclass
class CullReport:
    """Summary of a single-file cull pass.

    ``rewritten`` counts prediction rows that were actually in the old
    corrupted shape and were rewritten (or would be, in dry-run mode).
    ``preserved`` counts rows that passed through untouched because they
    were either non-prediction rows, already-migrated prediction rows,
    or legitimate non-truncated prediction rows. ``skipped`` counts rows
    that could not be classified — typically malformed JSON or
    prediction rows in an unknown shape — and were preserved defensively
    rather than destroyed.
    """

    path: Path
    rewritten: int = 0
    preserved: int = 0
    skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    committed: bool = False
    backup_path: Path | None = None

    def __add__(self, other: "CullReport") -> "CullReport":
        merged = CullReport(path=self.path)
        merged.rewritten = self.rewritten + other.rewritten
        merged.preserved = self.preserved + other.preserved
        merged.skipped = self.skipped + other.skipped
        merged.skip_reasons = list(self.skip_reasons) + list(other.skip_reasons)
        return merged


def _classify_and_process(
    rows: Iterable[tuple[int, str]],
) -> tuple[list[str], CullReport, bool]:
    """Walk rows and produce the post-cull JSONL lines + a report.

    Each input is a ``(lineno, raw_line)`` pair. Output lines are JSONL
    strings WITHOUT trailing newlines — the caller is responsible for
    joining them back. This separation lets ``cull_file`` handle
    dry-run vs commit without re-doing classification.

    The returned tuple is ``(out_lines, report, saw_footer)``. The
    third element tells the caller whether the file ever reached a
    ``type: "footer"`` row — see ``cull_file`` for the mid-flight
    guard that consumes this signal.
    """

    report = CullReport(path=Path())  # path is set by the caller
    out_lines: list[str] = []
    saw_footer = False

    for lineno, raw in rows:
        stripped = raw.rstrip("\n")
        if not stripped.strip():
            # Preserve blank/whitespace lines verbatim rather than
            # dropping them. Blank lines in the middle of a JSONL file
            # are unusual but we will not silently collapse them.
            out_lines.append(stripped)
            report.preserved += 1
            continue

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            report.skipped += 1
            report.skip_reasons.append(
                f"line {lineno}: malformed JSON ({exc.msg})"
            )
            out_lines.append(stripped)
            continue

        if not isinstance(row, dict):
            report.skipped += 1
            report.skip_reasons.append(
                f"line {lineno}: top-level JSON value is not an object"
            )
            out_lines.append(stripped)
            continue

        row_type = row.get("type")
        if row_type == "footer":
            saw_footer = True
            out_lines.append(stripped)
            report.preserved += 1
            continue
        if row_type == "header":
            out_lines.append(stripped)
            report.preserved += 1
            continue

        if row_type != "prediction":
            # Unknown row type. Preserve defensively and log.
            report.skipped += 1
            report.skip_reasons.append(
                f"line {lineno}: unknown row type {row_type!r}"
            )
            out_lines.append(stripped)
            continue

        if is_truncated_row(row):
            new_row = rewrite_truncated_row(row)
            out_lines.append(json.dumps(new_row))
            report.rewritten += 1
            continue

        # Prediction row that is either already migrated or a real
        # non-truncated prediction. Check that we recognize its shape
        # before preserving — a prediction row missing ``model_score``
        # entirely is suspicious and should be logged rather than
        # silently preserved.
        if "model_score" not in row and row.get("truncated") is not True:
            report.skipped += 1
            report.skip_reasons.append(
                f"line {lineno}: prediction row is missing model_score "
                "and is not already migrated"
            )
            out_lines.append(stripped)
            continue

        out_lines.append(stripped)
        report.preserved += 1

    return out_lines, report, saw_footer


def cull_file(path: Path, *, commit: bool) -> CullReport:
    """Cull a single ``predictions.jsonl`` file.

    In dry-run mode (``commit=False``, the default) the file on disk
    is not modified and no ``.bak`` sibling is written. The returned
    ``CullReport`` still accurately reports how many rows would be
    rewritten / preserved / skipped if the caller were to commit.

    ``cull_file`` refuses to touch a file that does not contain a
    ``type: "footer"`` row. A predictions file without a footer is
    either mid-flight (``smoke_vlm.py`` is still appending to it) or
    crashed before it could close cleanly; either way, rewriting it
    would race against the live writer or destroy forensically useful
    state. In that case, the file is reported as skipped with a
    reason naming the footer, no row-level rewrite is reported, no
    ``.bak`` is written, and the file on disk is left untouched. This
    is the structural guard for the concurrent-write race; operators
    should still treat the archive as quiescent before running
    ``--commit``.

    In commit mode (``commit=True``) the original file contents are
    copied to ``<path>.bak`` (a sibling with the ``.bak`` suffix
    appended) BEFORE the rewritten content is written. If the
    ``.bak`` already exists from a prior committed pass it is left
    alone — we never overwrite an existing backup, to avoid
    destroying the earliest recoverable snapshot of the archive.

    The rewrite itself is atomic: new content is written to a
    ``<path>.tmp`` sibling in the same directory, then ``os.replace``
    is used to rename it onto ``path``. This means a crash during the
    write leaves either the original content (if the crash happens
    before the rename) or the fully rewritten content (if after),
    never a partially written ``predictions.jsonl``.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        original_text = fh.read()

    lines = original_text.splitlines()
    numbered = list(enumerate(lines, start=1))
    out_lines, report, saw_footer = _classify_and_process(numbered)
    report.path = path

    if not saw_footer:
        # Mid-flight or crashed run. Refuse to touch it.
        report.skipped += 1
        report.skip_reasons.append(
            "file has no type: footer row (mid-flight run or crashed "
            "before close); refusing to rewrite to avoid racing with "
            "an active writer"
        )
        report.rewritten = 0
        report.committed = False
        for reason in report.skip_reasons:
            logger.warning("%s: %s", path, reason)
        return report

    for reason in report.skip_reasons:
        logger.warning("%s: %s", path, reason)

    if report.rewritten == 0 or not commit:
        report.committed = False
        return report

    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_text(original_text, encoding="utf-8")
        report.backup_path = backup
    else:
        # Earlier backup exists — don't overwrite, but record where it
        # is so the caller can reason about recoverability.
        report.backup_path = backup

    # Reassemble in the same line-ending style we read. The source
    # files are LF-terminated JSONL; we end with a trailing newline to
    # match.
    new_text = "\n".join(out_lines)
    if original_text.endswith("\n"):
        new_text += "\n"

    # Atomic write: tmp in same directory, then os.replace. Same-
    # directory is required for rename to be a cheap same-filesystem
    # operation. The tmp name is derived from the final name so a
    # leftover file after a crash is identifiable.
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(new_text, encoding="utf-8")
    os.replace(tmp_path, path)
    report.committed = True
    return report


def cull_archive(root: Path, *, commit: bool) -> list[CullReport]:
    """Walk a runs root and cull every ``predictions.jsonl`` found.

    Returns one ``CullReport`` per file that was inspected, whether or
    not any rows in that file needed rewriting. Does not recurse
    beyond one level of run directory — the archive layout is
    ``<root>/<run_id>/predictions.jsonl`` — but tolerates runs without a
    ``predictions.jsonl`` file by simply skipping them.
    """

    reports: list[CullReport] = []
    root = Path(root)
    if not root.exists():
        logger.warning("archive root does not exist: %s", root)
        return reports

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        predictions = run_dir / "predictions.jsonl"
        if not predictions.exists():
            continue
        reports.append(cull_file(predictions, commit=commit))
    return reports


def _format_report_summary(reports: list[CullReport], *, commit: bool) -> str:
    """Human-readable summary of a multi-file cull pass."""

    total_files = len(reports)
    affected_files = sum(1 for r in reports if r.rewritten > 0)
    total_rewritten = sum(r.rewritten for r in reports)
    total_preserved = sum(r.preserved for r in reports)
    total_skipped = sum(r.skipped for r in reports)

    header = "COMMITTED" if commit else "DRY-RUN"
    lines = [
        f"Zilch Reaper Historical — {header}",
        f"  files inspected: {total_files}",
        f"  files with truncated rows: {affected_files}",
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
            "Rewrite truncation-as-zero rows in the historical "
            "auto-grader-runs archive into the settled sentinel shape."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="~/dev/auto-grader-runs",
        help=(
            "Archive root to walk. Each immediate subdirectory is treated "
            "as a run directory containing a predictions.jsonl. "
            "Default: ~/dev/auto-grader-runs."
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Actually rewrite files. Without this flag the script runs "
            "in dry-run mode and reports what would change without "
            "touching anything on disk."
        ),
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

    root = Path(args.root).expanduser()
    reports = cull_archive(root, commit=args.commit)
    print(_format_report_summary(reports, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
