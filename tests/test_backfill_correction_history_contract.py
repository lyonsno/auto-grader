"""Fail-first contract tests for historical correction backfill.

This lane repairs archived ``predictions.jsonl`` files so they carry the
current human-verified ``corrected_score`` / ``correction_reason`` values
for items where the professor's original page mark is now believed to be
wrong. Unlike the truncation-focused Zilch Reaper historical pass, this
tool does not reinterpret model output; it only updates the persisted
truth metadata on matching archived prediction rows.

The tool under test does not exist yet when these tests are introduced.
Running this file before implementation MUST fail at import time. That is
the fail-first signal.

Contract surface:

* Load corrections from a supplied ground-truth YAML file, filtering to
  rows where ``corrected_score`` is present.
* Rewrite only matching archived prediction rows identified by
  ``(exam_id, question_id)``.
* Preserve every other field verbatim, especially ``professor_score`` and
  the model's own outputs.
* Leave unrelated rows untouched.
* Idempotent: files already carrying the current correction metadata are
  not rewritten on a second pass.
* Dry-run by default. ``--commit`` is required to touch disk.
* Commit mode writes a ``.bak`` sibling and uses the same footer guard as
  the historical truncation cull: files missing a footer are skipped
  defensively rather than rewritten mid-flight.
"""

from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "backfill_correction_history.py"
    )
    spec = importlib.util.spec_from_file_location(
        "backfill_correction_history", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _header_row() -> dict:
    return {
        "type": "header",
        "run_id": "20260419-000000-test-model",
        "model": "test-model",
        "prompt_version": "test-prompt-v1",
        "test_set_id": "test-set",
        "started_at": "2026-04-19T00:00:00Z",
    }


def _footer_row() -> dict:
    return {
        "type": "footer",
        "completed_at": "2026-04-19T00:10:00Z",
        "interrupted": False,
    }


def _prediction_row(
    *,
    exam_id: str,
    question_id: str,
    professor_score: float,
    corrected_score: float | None = None,
    correction_reason: str = "",
) -> dict:
    return {
        "type": "prediction",
        "exam_id": exam_id,
        "question_id": question_id,
        "answer_type": "numeric",
        "max_points": 4.0,
        "professor_score": professor_score,
        "corrected_score": corrected_score,
        "correction_reason": correction_reason,
        "professor_mark": "partial",
        "student_answer": "-186.2 kJ",
        "model_score": 4.0,
        "model_confidence": 0.96,
        "is_obviously_fully_correct": True,
        "is_obviously_wrong": False,
        "model_read": "-186.2 kJ",
        "score_basis": "Correct Hess's Law setup and arithmetic.",
        "model_reasoning": "Student reversed reaction (1), added reaction (2), and canceled shared species correctly.",
        "upstream_dependency": "none",
        "if_dependent_then_consistent": None,
        "raw_assistant": '{"model_score": 4.0}',
        "raw_reasoning": "The student performed the Hess's Law combination correctly.",
        "truncated": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_ground_truth(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            exams:
              - exam_id: 34-blue
                items:
                  - question_id: fr-8
                    answer_type: numeric
                    page: 2
                    professor_score: 2
                    max_points: 4
                    professor_mark: partial
                    student_answer: "-186.2 kJ"
                    notes: "Partial. Correct answer but confused intermediate work."
                    corrected_score: 4
                    correction_reason: "Verified Hess's Law work is correct."
              - exam_id: 15-blue
                items:
                  - question_id: fr-11c
                    answer_type: exact_match
                    page: 3
                    professor_score: 0
                    max_points: 0.5
                    professor_mark: x
                    student_answer: "2"
                    notes: "X. Should be 1."
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


class BackfillCorrectionHistoryContract(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.truth_path = self.root / "ground_truth.yaml"
        _write_ground_truth(self.truth_path)

    def test_load_corrections_filters_to_corrected_items_only(self) -> None:
        corrections = self.module.load_corrections(self.truth_path)
        self.assertEqual(
            corrections,
            {
                ("34-blue", "fr-8"): {
                    "corrected_score": 4.0,
                    "correction_reason": "Verified Hess's Law work is correct.",
                }
            },
        )

    def test_rewrite_row_stamps_current_correction_metadata(self) -> None:
        row = _prediction_row(
            exam_id="34-blue",
            question_id="fr-8",
            professor_score=2.0,
        )
        corrections = self.module.load_corrections(self.truth_path)

        rewritten = self.module.rewrite_prediction_row(row, corrections)

        self.assertEqual(rewritten["corrected_score"], 4.0)
        self.assertEqual(
            rewritten["correction_reason"],
            "Verified Hess's Law work is correct.",
        )
        self.assertEqual(
            rewritten["professor_score"],
            2.0,
            "historical professor score must be preserved",
        )
        self.assertEqual(
            rewritten["model_score"],
            4.0,
            "backfill must not rewrite model output",
        )

    def test_rewrite_row_is_pure(self) -> None:
        row = _prediction_row(
            exam_id="34-blue",
            question_id="fr-8",
            professor_score=2.0,
        )
        snapshot = json.loads(json.dumps(row))
        corrections = self.module.load_corrections(self.truth_path)

        self.module.rewrite_prediction_row(row, corrections)

        self.assertEqual(row, snapshot)

    def test_dry_run_reports_updates_without_touching_disk(self) -> None:
        path = self.root / "run-a" / "predictions.jsonl"
        rows = [
            _header_row(),
            _prediction_row(
                exam_id="34-blue",
                question_id="fr-8",
                professor_score=2.0,
            ),
            _prediction_row(
                exam_id="15-blue",
                question_id="fr-11c",
                professor_score=0.0,
            ),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        before = path.read_bytes()
        corrections = self.module.load_corrections(self.truth_path)

        report = self.module.backfill_file(path, corrections=corrections, commit=False)

        self.assertEqual(report.rewritten, 1)
        self.assertEqual(report.preserved, 3)
        self.assertEqual(report.skipped, 0)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(path.with_suffix(path.suffix + ".bak").exists())

    def test_commit_rewrites_only_matching_rows_and_writes_backup(self) -> None:
        path = self.root / "run-b" / "predictions.jsonl"
        rows = [
            _header_row(),
            _prediction_row(
                exam_id="34-blue",
                question_id="fr-8",
                professor_score=2.0,
            ),
            _prediction_row(
                exam_id="15-blue",
                question_id="fr-11c",
                professor_score=0.0,
            ),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        original = path.read_bytes()
        corrections = self.module.load_corrections(self.truth_path)

        report = self.module.backfill_file(path, corrections=corrections, commit=True)

        self.assertEqual(report.rewritten, 1)
        bak = path.with_suffix(path.suffix + ".bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_bytes(), original)

        after = _read_jsonl(path)
        corrected = after[1]
        unrelated = after[2]
        self.assertEqual(corrected["corrected_score"], 4.0)
        self.assertEqual(
            corrected["correction_reason"],
            "Verified Hess's Law work is correct.",
        )
        self.assertIsNone(
            unrelated["corrected_score"],
            "unrelated rows must stay untouched",
        )
        self.assertEqual(unrelated["correction_reason"], "")

    def test_backfill_is_idempotent_when_file_already_matches_truth(self) -> None:
        path = self.root / "run-c" / "predictions.jsonl"
        rows = [
            _header_row(),
            _prediction_row(
                exam_id="34-blue",
                question_id="fr-8",
                professor_score=2.0,
                corrected_score=4.0,
                correction_reason="Verified Hess's Law work is correct.",
            ),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        before = path.read_bytes()
        corrections = self.module.load_corrections(self.truth_path)

        report = self.module.backfill_file(path, corrections=corrections, commit=True)

        self.assertEqual(report.rewritten, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_missing_footer_file_is_skipped_and_not_rewritten(self) -> None:
        path = self.root / "run-d" / "predictions.jsonl"
        rows = [
            _header_row(),
            _prediction_row(
                exam_id="34-blue",
                question_id="fr-8",
                professor_score=2.0,
            ),
        ]
        _write_jsonl(path, rows)
        before = path.read_bytes()
        corrections = self.module.load_corrections(self.truth_path)

        report = self.module.backfill_file(path, corrections=corrections, commit=True)

        self.assertEqual(report.rewritten, 0)
        self.assertGreaterEqual(report.skipped, 1)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(path.with_suffix(path.suffix + ".bak").exists())

    def test_backfill_archive_walks_run_directories(self) -> None:
        corrections = self.module.load_corrections(self.truth_path)
        _write_jsonl(
            self.root / "run-1" / "predictions.jsonl",
            [
                _header_row(),
                _prediction_row(
                    exam_id="34-blue",
                    question_id="fr-8",
                    professor_score=2.0,
                ),
                _footer_row(),
            ],
        )
        _write_jsonl(
            self.root / "run-2" / "predictions.jsonl",
            [
                _header_row(),
                _prediction_row(
                    exam_id="15-blue",
                    question_id="fr-11c",
                    professor_score=0.0,
                ),
                _footer_row(),
            ],
        )

        reports = self.module.backfill_archive(
            self.root,
            corrections=corrections,
            commit=False,
        )

        self.assertEqual(len(reports), 2)
        self.assertEqual(sum(r.rewritten for r in reports), 1)

    def test_commit_streams_archive_instead_of_slurping_with_read_text(self) -> None:
        path = self.root / "run-stream" / "predictions.jsonl"
        _write_jsonl(
            path,
            [
                _header_row(),
                _prediction_row(
                    exam_id="34-blue",
                    question_id="fr-8",
                    professor_score=2.0,
                ),
                _footer_row(),
            ],
        )
        corrections = self.module.load_corrections(self.truth_path)

        with mock.patch(
            "pathlib.Path.read_text",
            side_effect=AssertionError(
                "backfill_file should stream archive input instead of read_text slurping"
            ),
        ):
            report = self.module.backfill_file(
                path, corrections=corrections, commit=True
            )

        self.assertEqual(report.rewritten, 1)
        self.assertTrue(report.committed)
        after = _read_jsonl(path)
        self.assertEqual(after[1]["corrected_score"], 4.0)

    def test_main_returns_nonzero_for_missing_ground_truth_file(self) -> None:
        missing = self.root / "missing-ground-truth.yaml"

        with self.assertLogs("backfill_correction_history", level="ERROR") as logs:
            rc = self.module.main(
                ["--ground-truth", str(missing), str(self.root)]
            )

        self.assertNotEqual(rc, 0)
        self.assertIn("ground truth", "\n".join(logs.output).lower())
        self.assertIn("does not exist", "\n".join(logs.output).lower())

    def test_main_returns_nonzero_for_malformed_ground_truth_file(self) -> None:
        broken = self.root / "broken-ground-truth.yaml"
        broken.write_text("exams: [\n", encoding="utf-8")

        with self.assertLogs("backfill_correction_history", level="ERROR") as logs:
            rc = self.module.main(
                ["--ground-truth", str(broken), str(self.root)]
            )

        self.assertNotEqual(rc, 0)
        self.assertIn("ground truth", "\n".join(logs.output).lower())
        self.assertIn("malformed", "\n".join(logs.output).lower())

    def test_preserves_malformed_json_lines_and_logs_skip_reason(self) -> None:
        path = self.root / "run-malformed-json" / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    json.dumps(_header_row()),
                    '{"type":"prediction","exam_id":"34-blue"',
                    json.dumps(_footer_row()),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        corrections = self.module.load_corrections(self.truth_path)

        with self.assertLogs("backfill_correction_history", level="WARNING") as logs:
            report = self.module.backfill_file(
                path, corrections=corrections, commit=False
            )

        self.assertEqual(report.rewritten, 0)
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertIn("malformed json", "\n".join(logs.output).lower())

    def test_preserves_non_object_json_rows_and_logs_skip_reason(self) -> None:
        path = self.root / "run-non-object" / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    json.dumps(_header_row()),
                    json.dumps(["not", "an", "object"]),
                    json.dumps(_footer_row()),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        corrections = self.module.load_corrections(self.truth_path)

        with self.assertLogs("backfill_correction_history", level="WARNING") as logs:
            report = self.module.backfill_file(
                path, corrections=corrections, commit=False
            )

        self.assertEqual(report.rewritten, 0)
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertIn("not an object", "\n".join(logs.output).lower())

    def test_preserves_unknown_row_types_and_logs_skip_reason(self) -> None:
        path = self.root / "run-unknown-type" / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    json.dumps(_header_row()),
                    json.dumps({"type": "mystery", "value": 7}),
                    json.dumps(_footer_row()),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        corrections = self.module.load_corrections(self.truth_path)

        with self.assertLogs("backfill_correction_history", level="WARNING") as logs:
            report = self.module.backfill_file(
                path, corrections=corrections, commit=False
            )

        self.assertEqual(report.rewritten, 0)
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertIn("unknown row type", "\n".join(logs.output).lower())

    def test_backfill_archive_returns_empty_for_missing_root(self) -> None:
        missing_root = self.root / "missing-archive-root"
        corrections = self.module.load_corrections(self.truth_path)

        with self.assertLogs("backfill_correction_history", level="WARNING") as logs:
            reports = self.module.backfill_archive(
                missing_root,
                corrections=corrections,
                commit=False,
            )

        self.assertEqual(reports, [])
        self.assertIn("archive root does not exist", "\n".join(logs.output))

    def test_main_dry_run_prints_summary_and_returns_zero(self) -> None:
        _write_jsonl(
            self.root / "run-main" / "predictions.jsonl",
            [
                _header_row(),
                _prediction_row(
                    exam_id="34-blue",
                    question_id="fr-8",
                    professor_score=2.0,
                ),
                _footer_row(),
            ],
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            rc = self.module.main(
                ["--ground-truth", str(self.truth_path), str(self.root)]
            )

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("Historical correction backfill", output)
        self.assertIn("files with updates: 1", output)


if __name__ == "__main__":
    unittest.main()
