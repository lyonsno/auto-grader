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
                argv=[
                    "--items",
                    "1",
                    "--model",
                    "gemma-test",
                    "--model-family",
                    "neutral",
                ],
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
                    "--model-family",
                    "neutral",
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
                    "--model-family",
                    "neutral",
                    "--run-dir",
                    str(run_dir),
                ],
            )

            self.assertEqual(code, 0)
            self.assertTrue(
                (run_dir / "predictions.jsonl").is_file(),
                "explicit --run-dir should remain a supported escape hatch",
            )

    def test_describe_only_mode_bypasses_grading_pipeline(self):
        module = _load_smoke_vlm()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "describe-only-run"
            describe_result = {
                "run_dir": run_dir,
                "records_path": run_dir / "probe.jsonl",
                "count_ok": 1,
                "count_err": 0,
            }

            stdout = io.StringIO()
            stderr = io.StringIO()
            describe_mock = None
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        module, "load_ground_truth", return_value=[self.item]
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "grade_all_items",
                        side_effect=AssertionError(
                            "describe-only mode must not invoke the grading pipeline"
                        ),
                    )
                )
                describe_mock = stack.enter_context(
                    mock.patch.object(
                        module,
                        "run_describe_only_mode",
                        return_value=describe_result,
                        create=True,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "smoke_vlm.py",
                            "--describe-only",
                            "--pick",
                            "15-blue:fr-1",
                            "--run-dir",
                            str(run_dir),
                        ],
                    )
                )
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = module.main()

            self.assertEqual(code, 0)
            assert describe_mock is not None
            describe_mock.assert_called_once()
            self.assertIn("describe-only", stdout.getvalue().lower())

    def test_prediction_record_carries_corrected_score_and_reason(self):
        """predictions.jsonl must be self-contained for offline analysis.

        The _PredictionWriter docstring promises this. When an EvalItem
        carries a corrected_score (recording a human-investigated prof
        grading error), the prediction record written for that item must
        include both corrected_score and correction_reason, so downstream
        tools (compare_runs.py, ad-hoc analysis) can compute truth_score
        without needing the original ground_truth.yaml alongside.

        Tested directly against _PredictionWriter.write_one rather than
        the full main() pipeline — the contract belongs to the writer.
        """
        module = _load_smoke_vlm()
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=5,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="check",
            student_answer="14.2031 moles",
            notes="internally consistent with wrong 5a",
            corrected_score=0.0,
            correction_reason=(
                "cannot add moles of N2 and H2 to get moles of NH3; "
                "prof gave 2/2 but methodology is invalid"
            ),
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=0.0,
            model_confidence=0.9,
            model_reasoning="caught the methodology error",
            model_read="14.2031 moles",
            raw_assistant='{"model_score": 0}',
            raw_reasoning="upstream 5a was also wrong",
            upstream_dependency="5(a)",
            if_dependent_then_consistent=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            predictions_path = run_dir / "predictions.jsonl"
            with module._PredictionWriter(
                predictions_path, model="qwen-test", run_dir=run_dir
            ) as writer:
                writer.write_one(item, prediction)

            self.assertTrue(predictions_path.is_file())
            records = [
                json.loads(line)
                for line in predictions_path.read_text().splitlines()
                if line.strip()
            ]
            pred_records = [r for r in records if r.get("type") == "prediction"]
            self.assertEqual(
                len(pred_records),
                1,
                "expected exactly one prediction record written",
            )
            record = pred_records[0]
            self.assertIn(
                "corrected_score",
                record,
                "prediction record must carry corrected_score for self-contained offline analysis",
            )
            self.assertEqual(
                record["corrected_score"],
                0.0,
                "corrected_score in the record must match the EvalItem's corrected_score",
            )
            self.assertIn(
                "correction_reason",
                record,
                "prediction record must carry correction_reason alongside corrected_score",
            )
            self.assertIn(
                "methodology",
                record["correction_reason"].lower(),
                "correction_reason should carry the human-investigator's note verbatim",
            )
            # The historical professor_score must ALSO be preserved, not
            # silently overwritten by the correction. Both fields matter.
            self.assertEqual(
                record["professor_score"],
                2.0,
                "professor_score field must preserve the historical prof mark, not be rewritten by corrected_score",
            )

    def test_prediction_record_corrected_fields_null_when_no_correction(self):
        """Uncorrected items must produce null corrected_score + empty reason.

        Backwards-compat contract: when the EvalItem has no corrected_score
        recorded, the prediction record must explicitly carry
        corrected_score=null (not omit the field), so downstream readers
        have an unambiguous signal rather than having to distinguish
        "field missing" from "field explicitly null". correction_reason
        in that case is the empty string.
        """
        module = _load_smoke_vlm()
        item = EvalItem(
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
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=2.0,
            model_confidence=0.75,
            model_reasoning="ok",
            model_read="13.6",
            raw_assistant='{"model_score": 2}',
            raw_reasoning="checked",
            upstream_dependency="none",
            if_dependent_then_consistent=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            predictions_path = run_dir / "predictions.jsonl"
            with module._PredictionWriter(
                predictions_path, model="qwen-test", run_dir=run_dir
            ) as writer:
                writer.write_one(item, prediction)

            records = [
                json.loads(line)
                for line in predictions_path.read_text().splitlines()
                if line.strip()
            ]
            pred_records = [r for r in records if r.get("type") == "prediction"]
            self.assertEqual(len(pred_records), 1)
            record = pred_records[0]
            self.assertIn("corrected_score", record)
            self.assertIsNone(
                record["corrected_score"],
                "uncorrected items must emit explicit null, not omit the field",
            )
            self.assertIn("correction_reason", record)
            self.assertEqual(record["correction_reason"], "")

    def test_progress_marks_match_against_truth_score_not_professor_score(self):
        """_progress must compare model_score against truth_score, not professor_score.

        When a corrected_score is recorded for an item (human-investigated
        prof grading error), the live console mark shown by _progress must
        reflect whether the model matches the *corrected* truth, not the
        historical prof mark. Otherwise the live display contradicts the
        eval report for the same run — the model could correctly return
        the corrected answer and the live display would show X while
        score_predictions counts it as an exact_match.

        Found by Panopticon auto-review of dfa0eb3.
        """
        module = _load_smoke_vlm()
        # The fr-5b case: prof gave 2/2 but the methodology is invalid,
        # corrected to 0. A model that also returns 0 is actually correct.
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=5,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="check",
            student_answer="14.2031 moles",
            notes="consistent with wrong 5a",
            corrected_score=0.0,
            correction_reason="cannot add moles of N2 and H2 to get moles of NH3",
        )
        pred_matches_truth = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=0.0,  # matches truth_score (corrected), not professor_score
            model_confidence=0.9,
            model_reasoning="caught the methodology error",
            model_read="14.2031 moles",
            raw_assistant='{"model_score": 0}',
            raw_reasoning="",
            upstream_dependency="5(a)",
            if_dependent_then_consistent=False,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            module._progress(1, 1, item, pred_matches_truth)
        output = stdout.getvalue()

        self.assertIn(
            "[=]",
            output,
            "model matching the corrected truth_score (0.0) must render "
            "as '=', not as 'X' against the original professor_score (2.0)",
        )
        self.assertNotIn(
            "[X]",
            output,
            "live mark must not disagree with truth_score when the model is actually correct per the corrected baseline",
        )

    def test_progress_marks_mismatch_when_model_only_matches_original_prof(self):
        """Converse case: model matches the original prof mark but not truth.

        When a corrected_score overrides the prof mark and the model
        returns the (now-incorrect) prof value, _progress must render
        that as a mismatch. Otherwise the live display would reward
        the model for agreeing with a known prof error.
        """
        module = _load_smoke_vlm()
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-5b",
            answer_type="numeric",
            page=5,
            professor_score=2.0,
            max_points=2.0,
            professor_mark="check",
            student_answer="14.2031 moles",
            notes="consistent with wrong 5a",
            corrected_score=0.0,
            correction_reason="cannot add moles of N2 and H2 to get moles of NH3",
        )
        pred_matches_prof_only = Prediction(
            exam_id="15-blue",
            question_id="fr-5b",
            model_score=2.0,  # matches professor_score (pre-correction) but NOT truth
            model_confidence=0.9,
            model_reasoning="accepted the student's premise",
            model_read="14.2031 moles",
            raw_assistant='{"model_score": 2}',
            raw_reasoning="",
            upstream_dependency="5(a)",
            if_dependent_then_consistent=True,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            module._progress(1, 1, item, pred_matches_prof_only)
        output = stdout.getvalue()

        self.assertIn(
            "[X]",
            output,
            "model matching the original professor_score but not the corrected truth must render as 'X'",
        )
        self.assertNotIn(
            "[=]",
            output,
            "live mark must not reward the model for matching a known prof error",
        )


if __name__ == "__main__":
    unittest.main()
