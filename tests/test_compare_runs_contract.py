from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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
                                "is_obviously_fully_correct": True,
                                "is_obviously_wrong": False,
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
            self.assertTrue(record.is_obviously_fully_correct)
            self.assertFalse(record.is_obviously_wrong)

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
                                    "is_obviously_fully_correct": False,
                                    "is_obviously_wrong": True,
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
            self.assertEqual(row["old__is_obviously_wrong"], True)
            self.assertEqual(row["new__is_obviously_wrong"], True)


if __name__ == "__main__":
    unittest.main()
