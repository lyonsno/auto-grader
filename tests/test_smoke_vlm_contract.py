from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_grader.eval_harness import EvalItem, Prediction


def _load_smoke_vlm():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "smoke_vlm.py"
    )
    spec = importlib.util.spec_from_file_location("smoke_vlm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _NullSink:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NarratorStub:
    def __init__(self, *args, **kwargs):
        self._stats = {
            "items_started": 1,
            "dispatches_total": 0,
            "summaries_emitted": 0,
            "drops_dedup": 0,
            "drops_empty": 0,
            "max_dispatches_one_item": 0,
        }

    def wrap_up(self, *args, **kwargs):
        return None

    def stats(self):
        return dict(self._stats)


class SmokeVlmContract(unittest.TestCase):
    def setUp(self):
        self.item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="check",
            student_answer="13.6",
            notes="density warmup",
        )
        self.prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.75,
            model_reasoning="correct density calculation",
            model_read="13.6 g/mL",
            raw_assistant='{"model_score": 2}',
            raw_reasoning="checked units",
            upstream_dependency="none",
            if_dependent_then_consistent=None,
        )

    def _run_main(
        self,
        module,
        *,
        argv: list[str],
        fake_home: Path | None = None,
        fake_script_repo: Path | None = None,
    ) -> int:
        def _fake_grade_all_items(subset, *_args, **_kwargs):
            return [self.prediction for _ in subset]

        stdout = io.StringIO()
        stderr = io.StringIO()
        patches = [
            mock.patch.object(module, "load_ground_truth", return_value=[self.item]),
            mock.patch.object(module, "grade_all_items", side_effect=_fake_grade_all_items),
            mock.patch.object(module, "NarratorSink", side_effect=lambda _cfg: _NullSink()),
            mock.patch.object(module, "ThinkingNarrator", _NarratorStub),
            mock.patch.object(sys, "argv", ["smoke_vlm.py", *argv]),
        ]
        if fake_home is not None:
            patches.append(
                mock.patch.dict(os.environ, {"HOME": str(fake_home)}, clear=False)
            )
        if fake_script_repo is not None:
            fake_script_path = fake_script_repo / "scripts" / "smoke_vlm.py"
            patches.append(mock.patch.object(module, "__file__", str(fake_script_path)))

        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return module.main()

    def test_main_defaults_run_dir_to_durable_root_outside_repo(self):
        module = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_home = tmp / "home"
            fake_repo = tmp / "repo"
            fake_home.mkdir()
            (fake_repo / "scripts").mkdir(parents=True)

            code = self._run_main(
                module,
                argv=["--items", "1", "--model", "gemma-test"],
                fake_home=fake_home,
                fake_script_repo=fake_repo,
            )

            self.assertEqual(code, 0)
            durable_root = fake_home / "dev" / "auto-grader-runs"
            self.assertTrue(
                durable_root.is_dir(),
                "default runs root should be a durable path under ~/dev, not a repo-local runs/ dir",
            )
            run_dirs = [path for path in durable_root.iterdir() if path.is_dir()]
            self.assertEqual(
                len(run_dirs),
                1,
                "one run should create exactly one dedicated run directory under the durable root",
            )
            self.assertFalse(
                str(run_dirs[0]).startswith(str(fake_repo)),
                "default run directory must live outside the active worktree",
            )

    def test_main_writes_manifest_with_prompt_model_and_test_set_identity(self):
        module = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "chosen-run"
            code = self._run_main(
                module,
                argv=[
                    "--tricky",
                    "--model",
                    "gemma-test",
                    "--run-dir",
                    str(run_dir),
                    "--narrate-stderr",
                    "--narrator-url",
                    "http://127.0.0.1:8002",
                    "--narrator-model",
                    "bonsai-test",
                ],
            )

            self.assertEqual(code, 0)
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(
                manifest_path.is_file(),
                "every run should persist a machine-readable manifest.json alongside predictions.jsonl",
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["run_id"])
            self.assertTrue(manifest["started_at"])
            self.assertTrue(manifest["finished_at"])
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["git_commit"])
            self.assertTrue(manifest["git_branch"])
            self.assertEqual(manifest["model"], "gemma-test")
            self.assertEqual(manifest["test_set_id"], "tricky-v1")
            self.assertEqual(manifest["item_count"], 1)
            self.assertEqual(manifest["narrator_model"], "bonsai-test")
            self.assertEqual(manifest["narrator_url"], "http://127.0.0.1:8002")
            self.assertRegex(manifest["prompt_version"], r"^\d{4}-\d{2}-\d{2}-")
            self.assertRegex(manifest["prompt_content_hash"], r"^[0-9a-f]{64}$")

    def test_main_respects_explicit_run_dir_override(self):
        module = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "manual-run-dir"
            code = self._run_main(
                module,
                argv=[
                    "--items",
                    "1",
                    "--model",
                    "qwen-test",
                    "--run-dir",
                    str(run_dir),
                ],
            )

            self.assertEqual(code, 0)
            self.assertTrue(
                (run_dir / "predictions.jsonl").is_file(),
                "explicit --run-dir should remain a supported escape hatch",
            )


if __name__ == "__main__":
    unittest.main()
