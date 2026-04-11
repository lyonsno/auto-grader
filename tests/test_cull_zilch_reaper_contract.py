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


# ---------------------------------------------------------------------------
# Forward-lane row fixtures
#
# These shapes did not exist in the legacy archive. They are emitted by
# ``Zilch Reaper Forward`` (landed 2026-04-11 at origin/cc/zilch-reaper-
# forward-main tip ``3d03320``) and the historical rewriter must handle
# them correctly when they appear alongside legacy rows in the same file.
# ---------------------------------------------------------------------------


# Message strings Forward sets on ``model_reasoning`` for its two failure
# paths. These are the literal strings from
# ``auto_grader/vlm_inference.py`` on ``cc/zilch-reaper-forward-main`` as
# of ``3d03320``. They are **not** the legacy truncation sentinel — the
# historical rewriter's detection predicate must not fire on them, and
# the rewriter must instead rely on the ``truncated`` flag to leave these
# rows alone via the already-migrated idempotency path.
_FORWARD_SENTINEL_PARSE_FAIL = (
    "Grader output could not be parsed as the required JSON "
    "(truncated or malformed)."
)
_FORWARD_SENTINEL_MISSING_SCORE = (
    "Grader emitted parseable JSON but did not include a model_score field."
)


def _forward_complete_prediction_row(
    *,
    question_id: str = "fr-7a",
    model_score: float = 1.5,
    model_confidence: float = 0.82,
) -> dict:
    """A prediction row written by Forward for a successfully graded item.

    Forward always emits an explicit ``truncated: False`` on complete
    rows. Every other field is real content; ``model_reasoning`` is the
    model's actual reasoning, not a sentinel. The historical rewriter
    must preserve this shape byte-exact — it is neither a truncation
    corruption nor a pre-cull "already migrated" shape, it is the
    Forward lane's steady-state write.
    """

    return {
        "type": "prediction",
        "exam_id": "15-blue",
        "question_id": question_id,
        "answer_type": "numeric",
        "max_points": 2.0,
        "professor_score": 2.0,
        "corrected_score": None,
        "correction_reason": "",
        "professor_mark": "check",
        "student_answer": "0.0423",
        "model_score": model_score,
        "model_confidence": model_confidence,
        "is_obviously_fully_correct": True,
        "is_obviously_wrong": False,
        "model_read": "0.0423 mol",
        "score_basis": (
            "Student computed moles from the given mass and molar mass "
            "correctly; answer matches the expected value within rounding."
        ),
        "model_reasoning": (
            "The student used mass / molar mass to find moles. "
            "Computation checks out: 1.234 / 29.2 ≈ 0.0423. "
            "Full credit."
        ),
        "upstream_dependency": "none",
        "if_dependent_then_consistent": None,
        "raw_assistant": (
            '{"model_score": 1.5, "model_confidence": 0.82, '
            '"model_read": "0.0423 mol"}'
        ),
        "raw_reasoning": (
            "The student's setup is correct and the arithmetic is clean. "
            "I see 1.234 / 29.2 = 0.0423. This matches the expected "
            "answer. Full credit."
        ),
        "truncated": False,
    }


def _forward_complete_zero_row() -> dict:
    """A Forward-written row where the model legitimately scored zero.

    ``model_score: 0.0`` with ``truncated: False`` must NOT be flagged —
    this is a real-zero row from a Forward-emitted file, not a parser
    default. The detection predicate must distinguish it from the
    legacy truncation-as-zero shape by checking the truncated flag and
    the model_reasoning field, not by the score value alone.
    """

    row = _forward_complete_prediction_row(
        question_id="fr-11c",
        model_score=0.0,
        model_confidence=0.91,
    )
    row["is_obviously_fully_correct"] = False
    row["is_obviously_wrong"] = True
    row["model_read"] = "42"
    row["score_basis"] = (
        "Student wrote 42 but the correct answer is the rate constant "
        "k = 2.3e-3; the value is off by orders of magnitude."
    )
    row["model_reasoning"] = (
        "The student gave 42 with no units and no setup. "
        "This is clearly wrong, so the score is 0."
    )
    return row


def _forward_truncated_row_new_sentinel() -> dict:
    """A Forward-written truncated row carrying the new sentinel text.

    Forward sets ``model_reasoning`` to one of two human-readable
    strings explaining the failure (``_FORWARD_SENTINEL_PARSE_FAIL`` or
    ``_FORWARD_SENTINEL_MISSING_SCORE``) and flags the row with
    ``truncated: True``. Neither of these strings matches the legacy
    truncation sentinel, so the historical rewriter's detection
    predicate does not fire on them. Instead, the idempotency check on
    the ``truncated`` flag catches them on the already-migrated path
    and the row is preserved untouched.
    """

    return {
        "type": "prediction",
        "exam_id": "15-blue",
        "question_id": "fr-10b",
        "answer_type": "numeric",
        "max_points": 1.0,
        "professor_score": 1.0,
        "corrected_score": None,
        "correction_reason": "",
        "professor_mark": "check",
        "student_answer": "-2.415",
        "model_score": None,
        "model_confidence": None,
        "is_obviously_fully_correct": None,
        "is_obviously_wrong": None,
        "model_read": "",
        "score_basis": "",
        "model_reasoning": _FORWARD_SENTINEL_PARSE_FAIL,
        "upstream_dependency": "none",
        "if_dependent_then_consistent": None,
        "raw_assistant": "<partial assistant content>",
        "raw_reasoning": "<full reasoning trace the model produced>",
        "truncated": True,
    }


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

    # ------------------------------------------------------------------
    # Review-hardening (regression lens, findings 1 / 2 / 3)
    # ------------------------------------------------------------------

    def test_detection_does_not_fire_on_quoted_sentinel_in_longer_reasoning(
        self,
    ) -> None:
        """Finding 1: false-positive on substring-contains detection.

        A legitimate prediction row whose ``model_reasoning`` merely
        *quotes* the truncation sentinel inside a longer paragraph must
        not be flagged. Real truncation rows are *exactly* the sentinel
        and nothing else — that's the parser default shape. A row whose
        reasoning is longer than the sentinel is not a parser default
        and must be preserved.
        """

        quoted_row = _complete_zero_prediction_row()
        quoted_row["model_reasoning"] = (
            "The grader previously emitted "
            f'"{_SENTINEL_MODEL_REASONING}" on this item, but the '
            "student answer is in fact present and clearly wrong."
        )
        self.assertFalse(
            self.module.is_truncated_row(quoted_row),
            "a row that merely quotes the sentinel inside longer prose "
            "must not be classified as truncated",
        )

    def test_detection_accepts_exact_sentinel_with_no_other_text(self) -> None:
        """Companion to the false-positive test.

        The canonical shape every real truncation row carries is
        ``model_reasoning`` equal to exactly the sentinel string. The
        detection predicate must still fire on this shape.
        """

        row = _truncated_prediction_row(
            model_reasoning=_SENTINEL_MODEL_REASONING
        )
        self.assertTrue(self.module.is_truncated_row(row))

    def test_commit_is_atomic_via_tmp_plus_rename(self) -> None:
        """Finding 2: non-atomic file overwrite.

        The file-level commit path must not expose a mid-write window
        in which ``predictions.jsonl`` is partially written. The
        operation is specified as:

        1. Write ``<path>.bak`` (original bytes)
        2. Write the new content to a temporary sibling
        3. Atomically rename the temporary sibling onto ``<path>``

        We verify the contract by mocking ``os.replace`` to observe
        that the rename happens and that the source path for the
        rename is a sibling of ``<path>``, not ``<path>`` itself.
        """

        import os as _os
        from unittest import mock

        path = self.root / "run-atomic" / "predictions.jsonl"
        _write_jsonl(
            path,
            [_header_row(), _truncated_prediction_row(), _footer_row()],
        )

        real_replace = _os.replace
        observed: list[tuple[str, str]] = []

        def spy_replace(src, dst, *a, **kw):
            observed.append((str(src), str(dst)))
            return real_replace(src, dst, *a, **kw)

        with mock.patch.object(self.module.os, "replace", side_effect=spy_replace):
            self.module.cull_file(path, commit=True)

        self.assertEqual(
            len(observed),
            1,
            "commit must call os.replace exactly once to land the rewrite",
        )
        src, dst = observed[0]
        self.assertEqual(dst, str(path), "rename target must be the final path")
        self.assertNotEqual(
            src,
            str(path),
            "rename source must be a temporary sibling, not the final path",
        )
        self.assertEqual(
            Path(src).parent,
            path.parent,
            "temporary file must live in the same directory as the final "
            "path so the rename is a same-filesystem atomic operation",
        )

    def test_commit_leaves_no_tmp_file_behind_on_success(self) -> None:
        """After a successful commit, only the rewritten file and its
        ``.bak`` sibling should exist in the run directory. The
        temporary file used for the atomic rename must have been
        consumed by the rename."""

        path = self.root / "run-no-tmp" / "predictions.jsonl"
        _write_jsonl(
            path,
            [_header_row(), _truncated_prediction_row(), _footer_row()],
        )

        self.module.cull_file(path, commit=True)

        siblings = sorted(p.name for p in path.parent.iterdir())
        self.assertEqual(
            siblings,
            ["predictions.jsonl", "predictions.jsonl.bak"],
            "no leftover .tmp or partial files should remain after commit",
        )

    def test_file_without_footer_is_skipped_as_in_flight(self) -> None:
        """Finding 3: concurrent-write guard against mid-flight runs.

        A ``predictions.jsonl`` that does not contain a ``type:
        "footer"`` row is either mid-flight (``smoke_vlm.py`` is still
        writing to it) or crashed before it could close cleanly. In
        either case, the rewriter must refuse to touch it, report the
        file as skipped with a reason, and leave both the file on
        disk and the ``.bak`` sibling absent.
        """

        path = self.root / "run-inflight" / "predictions.jsonl"
        _write_jsonl(
            path,
            [_header_row(), _truncated_prediction_row()],  # no footer
        )
        original = path.read_bytes()

        report = self.module.cull_file(path, commit=True)

        self.assertEqual(
            report.rewritten,
            0,
            "a footerless file must not have any rows rewritten",
        )
        self.assertGreaterEqual(
            report.skipped,
            1,
            "the footerless file itself must be reported as skipped",
        )
        self.assertTrue(
            any("footer" in r.lower() for r in report.skip_reasons),
            "skip reason must name the footer as the trigger; "
            f"got {report.skip_reasons}",
        )
        self.assertEqual(
            path.read_bytes(),
            original,
            "the file on disk must be unchanged after a footerless "
            "skip",
        )
        self.assertFalse(
            path.with_name(path.name + ".bak").exists(),
            "no .bak should be written when the file was skipped",
        )

    def test_file_with_footer_is_culled_normally(self) -> None:
        """Companion to the footer-skip test.

        A file carrying a proper footer row is a completed run and
        must be culled normally — this pins that the footer check
        does not accidentally block healthy files.
        """

        path = self.root / "run-complete" / "predictions.jsonl"
        _write_jsonl(
            path,
            [_header_row(), _truncated_prediction_row(), _footer_row()],
        )

        report = self.module.cull_file(path, commit=True)

        self.assertEqual(report.rewritten, 1)
        self.assertEqual(report.skipped, 0)

    # ------------------------------------------------------------------
    # Forward-lane coordination (Zilch Reaper Forward landed 2026-04-11)
    # ------------------------------------------------------------------

    def test_forward_written_complete_row_is_not_flagged(self) -> None:
        """Forward-lane complete rows carry ``truncated: False``.

        These rows must not be flagged as truncated by the historical
        rewriter. They have real ``model_score``, real
        ``model_confidence``, real ``model_reasoning``, and an explicit
        ``truncated: False`` flag. The detection predicate must
        distinguish them from both the legacy truncation-as-zero shape
        and the already-migrated shape.
        """

        row = _forward_complete_prediction_row()
        self.assertFalse(self.module.is_truncated_row(row))

    def test_forward_written_complete_zero_row_is_not_flagged(self) -> None:
        """A Forward-written row that legitimately scored the item 0.

        ``model_score: 0.0`` is not evidence of truncation when the row
        carries ``truncated: False`` and has real reasoning. This pins
        that the detection predicate does not confuse a real-zero
        Forward row with a legacy truncation row.
        """

        row = _forward_complete_zero_row()
        self.assertFalse(self.module.is_truncated_row(row))

    def test_forward_written_truncated_row_new_sentinel_is_not_reflagged(
        self,
    ) -> None:
        """Forward-lane truncated rows use a new sentinel string.

        Forward sets ``model_reasoning`` to a different string than the
        legacy truncation sentinel, but also flags the row with
        ``truncated: True``. The idempotency check on the ``truncated``
        flag must catch these rows and leave them alone, rather than
        matching on the legacy sentinel string — because the legacy
        sentinel string is not what Forward writes.
        """

        row = _forward_truncated_row_new_sentinel()
        self.assertFalse(
            self.module.is_truncated_row(row),
            "Forward-written truncated rows must be caught by the "
            "already-migrated idempotency path, not re-flagged",
        )

    def test_mixed_forward_and_legacy_file_is_culled_correctly(self) -> None:
        """A file can contain both legacy and Forward-written rows.

        During the overlap window, a single ``predictions.jsonl`` may
        contain a mix of legacy truncation-as-zero rows (which must be
        rewritten), Forward-written truncated rows (which must be
        preserved as already-migrated), and Forward-written complete
        rows (which must be preserved as real predictions). The
        rewriter must handle the mixed case without touching any
        non-legacy row.
        """

        path = self.root / "run-mixed-forward" / "predictions.jsonl"
        legacy_truncated = _truncated_prediction_row(question_id="fr-10b")
        forward_truncated = _forward_truncated_row_new_sentinel()
        forward_complete = _forward_complete_prediction_row()
        forward_complete_zero = _forward_complete_zero_row()
        rows = [
            _header_row(),
            legacy_truncated,
            forward_truncated,
            forward_complete,
            forward_complete_zero,
            _footer_row(),
        ]
        _write_jsonl(path, rows)

        report = self.module.cull_file(path, commit=True)

        self.assertEqual(
            report.rewritten,
            1,
            "only the legacy truncated row should be rewritten",
        )
        self.assertEqual(report.skipped, 0)

        rewritten_rows = _read_jsonl(path)

        # Legacy row is now in the new sentinel shape.
        rewritten_legacy = rewritten_rows[1]
        self.assertEqual(
            rewritten_legacy["question_id"], legacy_truncated["question_id"]
        )
        self.assertIsNone(rewritten_legacy["model_score"])
        self.assertIsNone(rewritten_legacy["model_confidence"])
        self.assertIs(rewritten_legacy["truncated"], True)

        # Forward-written rows are byte-exact preserved.
        self.assertEqual(rewritten_rows[2], forward_truncated)
        self.assertEqual(rewritten_rows[3], forward_complete)
        self.assertEqual(rewritten_rows[4], forward_complete_zero)


if __name__ == "__main__":
    unittest.main()
