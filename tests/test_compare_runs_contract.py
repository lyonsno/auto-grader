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
        return run_dir

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

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "compare_runs.py",
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
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = module.main()

            self.assertEqual(
                code,
                0,
                f"query mode should resolve latest matching manifests without raw run-dir inputs; stderr was: {stderr.getvalue()}",
            )
            rows = out_path.read_text().splitlines()
            self.assertEqual(len(rows), 2)
            header, row = rows
            self.assertIn("gemma-v2__score", header)
            self.assertIn("qwen-v1__score", header)
            self.assertIn(",2.0,", f",{row},")
            self.assertIn(",1.0,", f",{row},")


if __name__ == "__main__":
    unittest.main()
