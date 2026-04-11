"""Fail-first contract tests for scripts/cull_zilch_reaper.py.

Operation Zilch Reaper, historical lane. These tests pin the contract for
the historical rewriter that walks ``~/dev/auto-grader-runs/`` and rewrites
truncation-as-zero rows into the settled sentinel shape:

    model_score: null
    model_confidence: null
    truncated: true

The sentinel decision is settled by the human on 2026-04-11 and owned by
``Zilch Reaper Forward``. This lane treats the shape as fixed.

Tests are written before any implementation or dry-run code exists. Running
this file against a repository without ``scripts/cull_zilch_reaper.py``
MUST fail at import time — that is the fail-first signal.

Tests cover the contract surface that is reachable without the forward fix:

* **Detection**: exact-sentinel match plus tolerance for case and trailing
  punctuation drift on ``model_reasoning``.
* **Rewrite shape**: truncated rows emit ``model_score=None``,
  ``model_confidence=None``, ``truncated=True``.
* **Preservation invariant**: every other field of a truncated row is
  preserved verbatim — especially ``raw_reasoning``, ``raw_assistant``,
  ``professor_score``, ``corrected_score``, ground-truth and metadata
  fields. The rewriter is not allowed to discard data.
* **Non-prediction rows untouched**: ``type: "header"`` and
  ``type: "footer"`` rows pass through unchanged.
* **Non-truncated prediction rows untouched**: real ``model_score: 0.0``
  rows (grader committed to a zero) stay intact. We must not rewrite
  zeros the model actually produced.
* **Idempotency**: a file already in the new shape is left alone. Running
  the rewriter twice produces the same result as running it once.
* **Skip-and-log**: rows matching neither the old nor the new shape (e.g.
  corrupted or from an unknown future format) are logged and passed
  through unchanged, not destroyed.
* **Archive safety**: committing the rewrite creates a ``.bak`` file
  alongside the original and leaves it in place, so the cull is
  reversible.
* **Dry-run default**: without ``--commit``, nothing on disk is modified
  even when the in-memory rewrite report says rows would be rewritten.

The tests deliberately do NOT exercise the forward-fix grader paths, the
live archive at ``~/dev/auto-grader-runs/``, or network/model behavior.
They operate entirely against synthetic JSONL fixtures in a tempdir so
they can run in CI and on arbitrary checkouts without touching
irreplaceable data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_SENTINEL_MODEL_REASONING = (
    "Grader output was truncated at the max token limit "
    "before it finished the required JSON."
)


def _load_cull_module():
    """Import ``scripts/cull_zilch_reaper.py`` by path.

    Matches the loader pattern used by ``test_compare_runs_contract.py``.
    A missing implementation file is the fail-first signal — every test
    in this file MUST error at setUp when the script does not yet exist.
    """

    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "cull_zilch_reaper.py"
    )
    spec = importlib.util.spec_from_file_location("cull_zilch_reaper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Row fixtures
# ---------------------------------------------------------------------------


def _header_row() -> dict:
    return {
        "type": "header",
        "run_id": "20260411-000000-test-model",
        "model": "test-model",
        "prompt_version": "test-prompt-v1",
        "test_set_id": "test-set",
        "started_at": "2026-04-11T00:00:00Z",
    }


def _footer_row() -> dict:
    return {
        "type": "footer",
        "completed_at": "2026-04-11T00:10:00Z",
        "interrupted": False,
    }


def _truncated_prediction_row(
    *,
    exam_id: str = "15-blue",
    question_id: str = "fr-10b",
    model_reasoning: str = _SENTINEL_MODEL_REASONING,
    raw_reasoning: str = (
        "\nThe user wants me to grade question 10(b). "
        "**Question 10(b):** What is the energy of one mole of..."
    ),
) -> dict:
    """A prediction row in the old corrupted shape.

    Modeled on a real row from
    ``~/dev/auto-grader-runs/20260411-145605-qwen3p5-35B-A3B/predictions.jsonl``
    as observed on 2026-04-11. Keep this shape in sync with the real
    archive if it ever drifts.
    """

    return {
        "type": "prediction",
        "exam_id": exam_id,
        "question_id": question_id,
        "answer_type": "numeric",
        "max_points": 1.0,
        "professor_score": 1.0,
        "corrected_score": None,
        "correction_reason": "",
        "professor_mark": "check",
        "student_answer": "-2.415",
        "model_score": 0.0,
        "model_confidence": 0.0,
        "is_obviously_fully_correct": None,
        "is_obviously_wrong": None,
        "model_read": "",
        "score_basis": "",
        "model_reasoning": model_reasoning,
        "upstream_dependency": "none",
        "if_dependent_then_consistent": None,
        "raw_assistant": "",
        "raw_reasoning": raw_reasoning,
    }


def _complete_zero_prediction_row() -> dict:
    """A prediction row where the model really did commit to a zero.

    The rewriter must NOT touch this row: ``model_score: 0.0`` alone is
    not evidence of truncation.
    """

    return {
        "type": "prediction",
        "exam_id": "15-blue",
        "question_id": "fr-11c",
        "answer_type": "numeric",
        "max_points": 1.0,
        "professor_score": 0.0,
        "corrected_score": None,
        "correction_reason": "",
        "professor_mark": "x",
        "student_answer": "42",
        "model_score": 0.0,
        "model_confidence": 0.85,
        "is_obviously_fully_correct": False,
        "is_obviously_wrong": True,
        "model_read": "42",
        "score_basis": (
            "Student wrote 42 but the correct answer is the rate constant "
            "k = 2.3e-3; the value is off by orders of magnitude."
        ),
        "model_reasoning": (
            "The student gave 42 with no units and no setup. "
            "This is clearly wrong, so the score is 0."
        ),
        "upstream_dependency": "none",
        "if_dependent_then_consistent": None,
        "raw_assistant": '{"model_score": 0.0, "model_confidence": 0.85}',
        "raw_reasoning": "The student wrote 42. The correct answer is ...",
    }


def _already_rewritten_prediction_row() -> dict:
    """A prediction row already in the new (post-cull) shape.

    The rewriter must leave this row alone. Detection of "already in the
    new shape" is carried by the presence of the ``truncated: true``
    flag, not by the absence of the old sentinel string, because
    ``Zilch Reaper Forward`` may also emit this shape during the overlap
    window.
    """

    row = _truncated_prediction_row()
    row["model_score"] = None
    row["model_confidence"] = None
    row["truncated"] = True
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CullZilchReaperContract(unittest.TestCase):
    """Contract surface for the historical rewriter.

    These tests operate on synthetic JSONL files in a tempdir. They do not
    touch the real archive at ``~/dev/auto-grader-runs/``.
    """

    def setUp(self) -> None:
        self.module = _load_cull_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Row-level classification
    # ------------------------------------------------------------------

    def test_detects_exact_sentinel_truncated_row(self) -> None:
        row = _truncated_prediction_row()
        self.assertTrue(self.module.is_truncated_row(row))

    def test_tolerates_case_drift_on_sentinel(self) -> None:
        row = _truncated_prediction_row(
            model_reasoning=_SENTINEL_MODEL_REASONING.upper()
        )
        self.assertTrue(
            self.module.is_truncated_row(row),
            "case drift on the sentinel must still be recognized",
        )

    def test_tolerates_trailing_punctuation_drift_on_sentinel(self) -> None:
        row = _truncated_prediction_row(
            model_reasoning=_SENTINEL_MODEL_REASONING.rstrip(".")
        )
        self.assertTrue(
            self.module.is_truncated_row(row),
            "missing trailing period must still be recognized",
        )

    def test_does_not_flag_complete_zero_row_as_truncated(self) -> None:
        self.assertFalse(
            self.module.is_truncated_row(_complete_zero_prediction_row())
        )

    def test_does_not_flag_header_or_footer_as_truncated(self) -> None:
        self.assertFalse(self.module.is_truncated_row(_header_row()))
        self.assertFalse(self.module.is_truncated_row(_footer_row()))

    def test_does_not_flag_already_rewritten_row_as_truncated(self) -> None:
        self.assertFalse(
            self.module.is_truncated_row(_already_rewritten_prediction_row()),
            "rows already carrying truncated=True must not be re-flagged",
        )

    # ------------------------------------------------------------------
    # Rewrite shape
    # ------------------------------------------------------------------

    def test_rewrite_sets_sentinel_fields_to_none_and_flag_to_true(self) -> None:
        row = _truncated_prediction_row()
        rewritten = self.module.rewrite_truncated_row(row)
        self.assertIsNone(rewritten["model_score"])
        self.assertIsNone(rewritten["model_confidence"])
        self.assertIs(rewritten["truncated"], True)

    def test_rewrite_preserves_raw_reasoning_verbatim(self) -> None:
        raw = "The model's full real reasoning trace, which is the only trustworthy part."
        row = _truncated_prediction_row(raw_reasoning=raw)
        rewritten = self.module.rewrite_truncated_row(row)
        self.assertEqual(rewritten["raw_reasoning"], raw)

    def test_rewrite_preserves_all_non_score_fields(self) -> None:
        row = _truncated_prediction_row()
        rewritten = self.module.rewrite_truncated_row(row)
        preserved_keys = [
            "type",
            "exam_id",
            "question_id",
            "answer_type",
            "max_points",
            "professor_score",
            "corrected_score",
            "correction_reason",
            "professor_mark",
            "student_answer",
            "model_read",
            "score_basis",
            "model_reasoning",
            "upstream_dependency",
            "if_dependent_then_consistent",
            "raw_assistant",
            "raw_reasoning",
        ]
        for key in preserved_keys:
            with self.subTest(key=key):
                self.assertEqual(rewritten[key], row[key])

    def test_rewrite_does_not_mutate_input_row(self) -> None:
        row = _truncated_prediction_row()
        snapshot = json.loads(json.dumps(row))
        self.module.rewrite_truncated_row(row)
        self.assertEqual(row, snapshot, "rewrite_truncated_row must be pure")

    # ------------------------------------------------------------------
    # File-level dry run
    # ------------------------------------------------------------------

    def test_dry_run_reports_rewrite_counts_without_touching_disk(self) -> None:
        path = self.root / "run-a" / "predictions.jsonl"
        rows = [
            _header_row(),
            _truncated_prediction_row(question_id="fr-10b"),
            _complete_zero_prediction_row(),
            _truncated_prediction_row(question_id="fr-5b"),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        before = path.read_bytes()

        report = self.module.cull_file(path, commit=False)

        self.assertEqual(report.rewritten, 2)
        self.assertEqual(report.preserved, 3)
        self.assertEqual(report.skipped, 0)
        self.assertEqual(
            path.read_bytes(),
            before,
            "dry-run must not modify the file on disk",
        )
        self.assertFalse(
            path.with_suffix(path.suffix + ".bak").exists(),
            "dry-run must not create a .bak file",
        )

    def test_commit_rewrites_file_and_writes_bak(self) -> None:
        path = self.root / "run-b" / "predictions.jsonl"
        rows = [
            _header_row(),
            _truncated_prediction_row(),
            _complete_zero_prediction_row(),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        original_bytes = path.read_bytes()

        report = self.module.cull_file(path, commit=True)
        self.assertEqual(report.rewritten, 1)

        bak = path.with_suffix(path.suffix + ".bak")
        self.assertTrue(bak.exists(), "commit must write a .bak sibling")
        self.assertEqual(
            bak.read_bytes(),
            original_bytes,
            ".bak must contain the original file bytes",
        )

        rewritten = _read_jsonl(path)
        self.assertEqual(rewritten[0]["type"], "header")
        self.assertEqual(rewritten[-1]["type"], "footer")

        truncated = [r for r in rewritten if r.get("truncated") is True]
        self.assertEqual(len(truncated), 1)
        self.assertIsNone(truncated[0]["model_score"])
        self.assertIsNone(truncated[0]["model_confidence"])

        complete = [
            r
            for r in rewritten
            if r.get("type") == "prediction" and "truncated" not in r
        ]
        self.assertEqual(
            len(complete),
            1,
            "the real-zero row must stay untouched and unflagged",
        )
        self.assertEqual(complete[0]["model_score"], 0.0)
        self.assertEqual(complete[0]["model_confidence"], 0.85)

    def test_commit_preserves_header_and_footer_bytes_exactly(self) -> None:
        path = self.root / "run-c" / "predictions.jsonl"
        header = _header_row()
        footer = _footer_row()
        _write_jsonl(path, [header, _truncated_prediction_row(), footer])

        self.module.cull_file(path, commit=True)
        rows = _read_jsonl(path)

        self.assertEqual(rows[0], header)
        self.assertEqual(rows[-1], footer)

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def test_rewriter_is_idempotent_on_mixed_file(self) -> None:
        path = self.root / "run-d" / "predictions.jsonl"
        rows = [
            _header_row(),
            _truncated_prediction_row(question_id="fr-10b"),
            _already_rewritten_prediction_row(),
            _complete_zero_prediction_row(),
            _footer_row(),
        ]
        _write_jsonl(path, rows)

        first = self.module.cull_file(path, commit=True)
        after_first = path.read_bytes()

        # Second pass: nothing should change on disk and no row should be
        # reported as newly rewritten.
        second = self.module.cull_file(path, commit=True)
        after_second = path.read_bytes()

        self.assertEqual(first.rewritten, 1)
        self.assertEqual(second.rewritten, 0)
        self.assertEqual(
            after_first,
            after_second,
            "second pass must be a no-op on disk (idempotent)",
        )

    def test_already_rewritten_file_is_unchanged_by_cull(self) -> None:
        path = self.root / "run-e" / "predictions.jsonl"
        rows = [
            _header_row(),
            _already_rewritten_prediction_row(),
            _already_rewritten_prediction_row(),
            _footer_row(),
        ]
        _write_jsonl(path, rows)
        before = path.read_bytes()

        report = self.module.cull_file(path, commit=True)

        self.assertEqual(report.rewritten, 0)
        self.assertEqual(
            path.read_bytes(),
            before,
            "fully-migrated file must be left byte-identical",
        )

    # ------------------------------------------------------------------
    # Skip-and-log on unrecognized shapes
    # ------------------------------------------------------------------

    def test_unknown_shape_row_is_skipped_not_destroyed(self) -> None:
        path = self.root / "run-f" / "predictions.jsonl"
        weird_row = {
            "type": "prediction",
            "exam_id": "15-blue",
            "question_id": "fr-99z",
            "some_future_field": "we don't know what this means",
        }
        rows = [_header_row(), weird_row, _footer_row()]
        _write_jsonl(path, rows)

        report = self.module.cull_file(path, commit=True)

        self.assertEqual(report.rewritten, 0)
        self.assertGreaterEqual(report.skipped, 1)

        after = _read_jsonl(path)
        self.assertEqual(
            after[1],
            weird_row,
            "unknown-shape rows must pass through unchanged",
        )


if __name__ == "__main__":
    unittest.main()
