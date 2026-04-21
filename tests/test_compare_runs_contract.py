from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_compare_runs():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "compare_runs.py"
    )
    spec = importlib.util.spec_from_file_location("compare_runs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompareRunsContract(unittest.TestCase):
    def _write_run(
        self,
        base: Path,
        *,
        run_name: str,
        model: str,
        prompt_version: str,
        test_set_id: str,
        started_at: str,
        score: float,
        corrected_score: float | None = None,
        correction_reason: str = "",
        acceptable_score_floor: float | None = None,
        acceptable_score_ceiling: float | None = None,
        acceptable_score_reason: str = "",
    ) -> Path:
        run_dir = base / run_name
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_name,
                    "run_dir": str(run_dir),
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": started_at,
                    "git_commit": "deadbeef",
                    "git_branch": "cc/test",
                    "model": model,
                    "base_url": "http://127.0.0.1:8001",
                    "prompt_version": prompt_version,
                    "prompt_content_hash": "a" * 64,
                    "test_set_id": test_set_id,
                    "item_count": 1,
                    "narrator_url": None,
                    "narrator_model": None,
                }
            )
            + "\n"
        )
        prediction_record: dict[str, object] = {
            "type": "prediction",
            "exam_id": "15-blue",
            "question_id": "fr-1",
            "answer_type": "numeric",
            "max_points": 2,
            "professor_score": 2,
            "professor_mark": "check",
            "student_answer": "foo",
            "model_score": score,
            "model_confidence": 0.5,
            "model_read": "foo",
            "model_reasoning": "bar",
            "raw_assistant": "{}",
            "raw_reasoning": "abc",
            "upstream_dependency": "none",
            "if_dependent_then_consistent": None,
        }
        if corrected_score is not None:
            prediction_record["corrected_score"] = corrected_score
            prediction_record["correction_reason"] = correction_reason
        if acceptable_score_floor is not None:
            prediction_record["acceptable_score_floor"] = acceptable_score_floor
        if acceptable_score_ceiling is not None:
            prediction_record["acceptable_score_ceiling"] = acceptable_score_ceiling
        if (
            acceptable_score_floor is not None
            or acceptable_score_ceiling is not None
            or acceptable_score_reason
        ):
            prediction_record["acceptable_score_reason"] = acceptable_score_reason
        (run_dir / "predictions.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "header",
                            "model": model,
                            "run_dir": str(run_dir),
                            "started": started_at,
                        }
                    ),
                    json.dumps(prediction_record),
                ]
            )
            + "\n"
        )
        return run_dir

    def _run_main(self, module, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["compare_runs.py", *argv]):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = module.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_load_run_records_extracts_elapsed_and_critic_score(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir()

            (run_dir / "predictions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "header",
                                "model": "gemma-old",
                                "run_dir": str(run_dir),
                                "started": "2026-04-08T20:00:00",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "prediction",
                                "exam_id": "15-blue",
                                "question_id": "fr-1",
                                "answer_type": "numeric",
                                "max_points": 2,
                                "professor_score": 2,
                                "professor_mark": "check",
                                "student_answer": "foo",
                                "model_score": 1,
                                "model_confidence": 0.75,
                                "model_read": "foo",
                                "model_reasoning": "bar",
                                "raw_assistant": "{}",
                                "raw_reasoning": "abcdef",
                                "upstream_dependency": "none",
                                "if_dependent_then_consistent": None,
                            }
                        ),
                    ]
                )
                + "\n"
            )
            (run_dir / "critic.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "header"}),
                        json.dumps(
                            {
                                "type": "delta",
                                "exam_id": "15-blue",
                                "question_id": "fr-1",
                                "new_score": 2,
                            }
                        ),
                    ]
                )
                + "\n"
            )
            (run_dir / "narrator.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "header",
                                "text": "[item 1/6] 15-blue/fr-1 (numeric, 2.0 pts)",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "topic",
                                "text": "147s · Grader: 1/2 (missed conversion). Prof: 2/2.",
                            }
                        ),
                    ]
                )
                + "\n"
            )

            records = module.load_run_records(run_dir)
            record = records[("15-blue", "fr-1")]
            self.assertEqual(record.model, "gemma-old")
            self.assertEqual(record.elapsed_s, 147)
            self.assertEqual(record.critic_score, 2.0)
            self.assertEqual(record.reasoning_chars, 6)

    def test_build_comparison_rows_emits_per_run_columns(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            for label, score in (("old", 1), ("new", 2)):
                run_dir = base / label
                run_dir.mkdir()
                (run_dir / "predictions.jsonl").write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "type": "header",
                                    "model": f"{label}-model",
                                    "run_dir": str(run_dir),
                                    "started": "2026-04-08T20:00:00",
                                }
                            ),
                            json.dumps(
                                {
                                    "type": "prediction",
                                    "exam_id": "15-blue",
                                    "question_id": "fr-1",
                                    "answer_type": "numeric",
                                    "max_points": 2,
                                    "professor_score": 2,
                                    "professor_mark": "check",
                                    "student_answer": "foo",
                                    "model_score": score,
                                    "model_confidence": 0.5,
                                    "model_read": "foo",
                                    "model_reasoning": "bar",
                                    "raw_assistant": "{}",
                                    "raw_reasoning": "abc",
                                    "upstream_dependency": "none",
                                    "if_dependent_then_consistent": None,
                                }
                            ),
                        ]
                    )
                    + "\n"
                )

            rows = module.build_comparison_rows(
                [("old", base / "old"), ("new", base / "new")]
            )
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["exam_id"], "15-blue")
            self.assertEqual(row["old__score"], 1.0)
            self.assertEqual(row["new__score"], 2.0)
            self.assertEqual(row["old__model"], "old-model")
            self.assertEqual(row["new__model"], "new-model")

    def test_resolve_query_run_uses_discovered_manifest_parent_for_copied_runs(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs-root"
            runs_root.mkdir()
            copied_run = self._write_run(
                runs_root,
                run_name="copied-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-08T21:00:00",
                score=2.0,
            )

            manifest_path = copied_run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_dir"] = "/old/box/auto-grader-runs/copied-run"
            manifest_path.write_text(json.dumps(manifest) + "\n")

            resolved = module.resolve_query_run(
                runs_root=runs_root,
                query="model=gemma-test,prompt_version=prompt-v2,test_set_id=tricky-v1",
            )

            self.assertEqual(
                resolved,
                copied_run,
                "query resolution should treat the discovered manifest location as authoritative, not a stale embedded absolute path",
            )

    def test_main_can_resolve_latest_runs_by_manifest_metadata(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runs_root = base / "runs-root"
            runs_root.mkdir()
            out_path = base / "comparison.csv"

            self._write_run(
                runs_root,
                run_name="gemma-old",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-08T20:00:00",
                score=1.0,
            )
            self._write_run(
                runs_root,
                run_name="gemma-latest",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-08T21:00:00",
                score=2.0,
            )
            self._write_run(
                runs_root,
                run_name="qwen-latest",
                model="qwen-test",
                prompt_version="prompt-v1",
                test_set_id="tricky-v1",
                started_at="2026-04-08T21:30:00",
                score=1.0,
            )

            code, _stdout, stderr = self._run_main(
                module,
                [
                    "--runs-root",
                    str(runs_root),
                    "--query",
                    "model=gemma-test,prompt_version=prompt-v2,test_set_id=tricky-v1",
                    "--query",
                    "model=qwen-test,prompt_version=prompt-v1,test_set_id=tricky-v1",
                    "--label",
                    "gemma-v2",
                    "--label",
                    "qwen-v1",
                    "--out",
                    str(out_path),
                ],
            )

            self.assertEqual(
                code,
                0,
                f"query mode should resolve latest matching manifests without raw run-dir inputs; stderr was: {stderr}",
            )
            rows = out_path.read_text().splitlines()
            self.assertEqual(len(rows), 2)
            header, row = rows
            self.assertIn("gemma-v2__score", header)
            self.assertIn("qwen-v1__score", header)
            self.assertIn(",2.0,", f",{row},")
            self.assertIn(",1.0,", f",{row},")

    def test_load_run_records_picks_up_corrected_score_and_exposes_truth_score(self):
        """compare_runs must use the corrected truth_score when present.

        The eval harness's EvalItem exposes truth_score as a property that
        returns corrected_score when set, otherwise professor_score. The
        compare_runs surface has to mirror that contract so the same run
        viewed through both surfaces reports the same accuracy baseline.
        Pinning the invariant on RunRecord directly.
        """
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="corrected-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=0.0,
                corrected_score=0.0,
                correction_reason="prof methodology error",
            )
            records = module.load_run_records(run_dir)
            record = records[("15-blue", "fr-1")]
            self.assertEqual(
                record.professor_score,
                2.0,
                "professor_score field must preserve the historical prof mark",
            )
            self.assertEqual(
                record.corrected_score,
                0.0,
                "corrected_score field must be populated from the prediction record",
            )
            self.assertEqual(
                record.truth_score,
                0.0,
                "truth_score must reflect the corrected value when corrected_score is present",
            )

    def test_load_run_records_picks_up_sparse_acceptable_score_band(self):
        """compare_runs must round-trip acceptable-band telemetry fields."""
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="banded-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=1.0,
                acceptable_score_floor=1.0,
                acceptable_score_ceiling=1.5,
                acceptable_score_reason="sign-only slips may still receive generous setup credit",
            )
            records = module.load_run_records(run_dir)
            record = records[("15-blue", "fr-1")]
            self.assertEqual(record.acceptable_score_floor, 1.0)
            self.assertEqual(record.acceptable_score_ceiling, 1.5)
            self.assertEqual(
                record.acceptable_score_reason,
                "sign-only slips may still receive generous setup credit",
            )

    def test_build_comparison_rows_emits_acceptable_score_band_columns(self):
        """CSV output must expose acceptable-band telemetry when present."""
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="banded-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=1.0,
                acceptable_score_floor=1.0,
                acceptable_score_ceiling=1.5,
                acceptable_score_reason="sign-only slips may still receive generous setup credit",
            )
            rows = module.build_comparison_rows([("only", run_dir)])
            row = rows[0]

        self.assertEqual(row["acceptable_score_floor"], 1.0)
        self.assertEqual(row["acceptable_score_ceiling"], 1.5)
        self.assertEqual(
            row["acceptable_score_reason"],
            "sign-only slips may still receive generous setup credit",
        )

    def test_acceptable_score_band_columns_do_not_depend_on_run_order(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            legacy_run = self._write_run(
                base,
                run_name="legacy-run",
                model="gemma-old",
                prompt_version="prompt-v1",
                test_set_id="tricky-v1",
                started_at="2026-04-09T20:00:00",
                score=1.0,
            )
            banded_run = self._write_run(
                base,
                run_name="banded-run",
                model="gemma-new",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=1.5,
                acceptable_score_floor=1.0,
                acceptable_score_ceiling=1.5,
                acceptable_score_reason="setup credit survives the sign slip",
            )

            legacy_first = module.build_comparison_rows(
                [("legacy", legacy_run), ("banded", banded_run)]
            )[0]
            banded_first = module.build_comparison_rows(
                [("banded", banded_run), ("legacy", legacy_run)]
            )[0]

        for row in (legacy_first, banded_first):
            self.assertEqual(row["acceptable_score_floor"], 1.0)
            self.assertEqual(row["acceptable_score_ceiling"], 1.5)
            self.assertEqual(
                row["acceptable_score_reason"],
                "setup credit survives the sign slip",
            )

    def test_load_run_records_truth_score_falls_back_to_professor_score(self):
        """Without a correction, truth_score mirrors professor_score exactly.

        This is the backwards-compat path: old prediction.jsonl files
        from before corrected_score existed, and new prediction files
        for items that don't need a correction, both must produce the
        same truth_score they would have before this change.
        """
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="uncorrected-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=2.0,
            )
            records = module.load_run_records(run_dir)
            record = records[("15-blue", "fr-1")]
            self.assertEqual(record.professor_score, 2.0)
            self.assertIsNone(
                record.corrected_score,
                "corrected_score must be None when the prediction record carried no correction",
            )
            self.assertEqual(
                record.truth_score,
                2.0,
                "truth_score must fall back to professor_score when no correction is recorded",
            )

    def test_build_comparison_rows_emits_both_professor_and_truth_score_columns(self):
        """CSV output must carry professor_score AND truth_score columns.

        The historical record and the corrected baseline are both useful
        to operators — we keep the distinction so past-prof-error cases
        stay legible. When no correction is present, the two columns
        match; when a correction is present, truth_score reflects it
        while professor_score preserves the original prof mark.
        """
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="corrected-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=0.0,
                corrected_score=0.0,
                correction_reason="prof methodology error",
            )
            rows = module.build_comparison_rows([("only", run_dir)])
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(
                row["professor_score"],
                2.0,
                "professor_score column must preserve the historical prof mark",
            )
            self.assertIn(
                "truth_score",
                row,
                "build_comparison_rows must emit a truth_score column alongside professor_score",
            )
            self.assertEqual(
                row["truth_score"],
                0.0,
                "truth_score column must reflect the corrected value when corrected_score is present",
            )

    def test_build_comparison_rows_carries_professor_mark_for_unclear_history(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="unclear-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=0.0,
            )
            predictions_path = run_dir / "predictions.jsonl"
            rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                if row.get("type") == "prediction":
                    row["professor_mark"] = "unclear"
            predictions_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            comparison_rows = module.build_comparison_rows([("only", run_dir)])
            self.assertEqual(len(comparison_rows), 1)
            row = comparison_rows[0]
            self.assertIn(
                "professor_mark",
                row,
                "comparison rows should surface professor_mark so unclear historical items stop masquerading as score-bearing evidence",
            )
            self.assertEqual(row["professor_mark"], "unclear")

    def test_build_comparison_rows_truth_score_equals_professor_score_when_uncorrected(self):
        """Backwards-compat: no correction means the two columns match."""
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_dir = self._write_run(
                base,
                run_name="uncorrected-run",
                model="gemma-test",
                prompt_version="prompt-v2",
                test_set_id="tricky-v1",
                started_at="2026-04-09T21:00:00",
                score=2.0,
            )
            rows = module.build_comparison_rows([("only", run_dir)])
            row = rows[0]
            self.assertEqual(row["professor_score"], 2.0)
            self.assertEqual(
                row["truth_score"],
                2.0,
                "truth_score must equal professor_score when no correction is recorded",
            )

    def test_main_rejects_mixed_path_and_query_mode_until_order_is_preserved(self):
        module = _load_compare_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runs_root = base / "runs-root"
            runs_root.mkdir()
            path_run = self._write_run(
                base,
                run_name="path-run",
                model="path-model",
                prompt_version="prompt-path",
                test_set_id="manual-v1",
                started_at="2026-04-08T20:00:00",
                score=2.0,
            )
            self._write_run(
                runs_root,
                run_name="query-run",
                model="query-model",
                prompt_version="prompt-query",
                test_set_id="tricky-v1",
                started_at="2026-04-08T21:00:00",
                score=1.0,
            )

            code, _stdout, stderr = self._run_main(
                module,
                [
                    str(path_run),
                    "--runs-root",
                    str(runs_root),
                    "--query",
                    "model=query-model,prompt_version=prompt-query,test_set_id=tricky-v1",
                ],
            )

            self.assertEqual(code, 1)
            self.assertIn("cannot mix direct run paths with --query", stderr)


if __name__ == "__main__":
    unittest.main()
