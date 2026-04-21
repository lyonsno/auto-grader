from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_script(module_name: str):
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / f"{module_name}.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_narrator_reader():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "narrator_reader.py"
    )
    spec = importlib.util.spec_from_file_location("narrator_reader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NarratorReplayContract(unittest.TestCase):
    def test_replay_reconstructs_committed_lines_in_order_without_network_calls(self):
        replay = _load_script("narrator_replay")
        reader = _load_narrator_reader()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            narrator_path = run_dir / "narrator.jsonl"
            narrator_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "ts": 10.0,
                                "type": "header",
                                "text": "[item 1/2] 15-blue/fr-1 (numeric, 2.0 pts)",
                            }
                        ),
                        json.dumps(
                            {"ts": 10.5, "type": "delta", "text": "I'm ", "mode": "thought"}
                        ),
                        json.dumps(
                            {
                                "ts": 10.8,
                                "type": "delta",
                                "text": "checking the setup.",
                                "mode": "thought",
                            }
                        ),
                        json.dumps({"ts": 11.0, "type": "commit", "mode": "thought"}),
                        json.dumps(
                            {
                                "ts": 12.0,
                                "type": "ambiguity",
                                "text": "The final digit could be a 3 or an 8.",
                            }
                        ),
                        json.dumps(
                            {
                                "ts": 12.6,
                                "type": "delta",
                                "text": "I'm rescuing the units.",
                                "mode": "thought",
                            }
                        ),
                        json.dumps({"ts": 12.9, "type": "commit", "mode": "thought"}),
                        json.dumps(
                            {
                                "ts": 13.5,
                                "type": "topic",
                                "text": "19s · Grader: 1/2 (unit rescue). Prof: 2/2.",
                                "verdict": "undershoot",
                                "grader_score": 1.0,
                                "truth_score": 2.0,
                                "max_points": 2.0,
                            }
                        ),
                        json.dumps({"ts": 14.0, "type": "end"}),
                    ]
                )
                + "\n"
            )

            display = reader.PaintDryDisplay()
            committed_lines: list[str] = []
            sleep_calls: list[float] = []

            def _dispatch(msg: dict[str, object]) -> None:
                reader.dispatch_message(display, msg)
                if msg.get("type") == "commit":
                    committed_lines.append(display.frozen_line)

            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("replay must stay offline")):
                count = replay.replay_recorded_messages(
                    replay.load_recorded_messages(narrator_path),
                    dispatch=_dispatch,
                    speed=2.0,
                    sleep_fn=sleep_calls.append,
                )

        self.assertEqual(count, 9)
        self.assertEqual(
            committed_lines,
            [
                "I'm checking the setup.",
                "I'm rescuing the units.",
            ],
        )
        self.assertIn(
            ("ambiguity", "The final digit could be a 3 or an 8.", None),
            display.history,
            "replay should preserve structured legibility rows, not silently drop them",
        )
        self.assertEqual(
            [round(delay, 3) for delay in sleep_calls[:3]],
            [0.25, 0.15, 0.1],
        )

    def test_postmortem_writes_markdown_summary_for_run(self):
        postmortem = _load_script("narrator_postmortem")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "run_dir": str(run_dir),
                        "status": "completed",
                        "started_at": "2026-04-21T14:00:00",
                        "finished_at": "2026-04-21T14:02:00",
                        "git_commit": "deadbeef",
                        "git_branch": "cc/test",
                        "model": "qwen3p5-35B-A3B",
                        "base_url": "http://127.0.0.1:8001",
                        "prompt_version": "2026-04-21-replay-v1",
                        "prompt_content_hash": "a" * 64,
                        "test_set_id": "fixture-v1",
                        "item_count": 1,
                        "narrator_url": "http://127.0.0.1:8002",
                        "narrator_model": "bonsai-test",
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
                                "model": "qwen3p5-35B-A3B",
                                "run_dir": str(run_dir),
                                "started": "2026-04-21T14:00:00",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "prediction",
                                "exam_id": "15-blue",
                                "question_id": "fr-1",
                                "answer_type": "numeric",
                                "max_points": 2.0,
                                "professor_score": 2.0,
                                "professor_mark": "check",
                                "student_answer": "6.98 mL",
                                "model_score": 1.0,
                                "model_confidence": 0.63,
                                "model_read": "6.98 mL",
                                "model_reasoning": "Method is fine but units drifted.",
                                "score_basis": "1/2 - setup right, units off.",
                                "raw_assistant": "{}",
                                "raw_reasoning": "reasoning excerpt",
                                "upstream_dependency": "none",
                                "if_dependent_then_consistent": None,
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
                                "ts": 1.0,
                                "type": "header",
                                "text": "[item 1/1] 15-blue/fr-1 (numeric, 2.0 pts)",
                            }
                        ),
                        json.dumps(
                            {"ts": 1.2, "type": "delta", "text": "I'm ", "mode": "thought"}
                        ),
                        json.dumps(
                            {
                                "ts": 1.4,
                                "type": "delta",
                                "text": "checking the units.",
                                "mode": "thought",
                            }
                        ),
                        json.dumps({"ts": 1.5, "type": "commit", "mode": "thought"}),
                        json.dumps(
                            {
                                "ts": 2.0,
                                "type": "topic",
                                "text": "12s · Grader: 1/2 (units). Prof: 2/2.",
                                "verdict": "undershoot",
                                "grader_score": 1.0,
                                "truth_score": 2.0,
                                "max_points": 2.0,
                            }
                        ),
                        json.dumps({"ts": 2.5, "type": "end"}),
                    ]
                )
                + "\n"
            )

            output_path = postmortem.write_postmortem(
                run_dir,
                model="retro-bonsai",
                completion_fn=lambda **_: "# Findings\n\nThe grader missed recoverable unit credit.\n",
            )
            self.assertTrue(output_path.exists())
            text = output_path.read_text()
            self.assertIn("# Findings", text)
            self.assertIn("retro-bonsai", text)
            self.assertIn("15-blue/fr-1", text)


if __name__ == "__main__":
    unittest.main()
