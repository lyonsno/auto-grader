from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from auto_grader.eval_harness import EvalItem, FocusRegion, Prediction
from auto_grader.thinking_narrator import ThinkingNarrator
from scripts import smoke_vlm


def _load_smoke_vlm():
    return smoke_vlm


class _DummySink:
    def write_delta(self, text: str, *, mode: str = "thought") -> None:
        return None

    def rollback_live(self) -> None:
        return None

    def commit_live(self, *, mode: str = "thought") -> None:
        return None

    def write_drop(self, reason: str, text: str) -> None:
        return None

    def write_topic(self, text: str, verdict: str | None = None, **kwargs) -> None:
        return None

    def start_wrap_up(self) -> None:
        return None

    def write_wrap_up(self, text: str) -> None:
        return None


_BONSAI_SNAPSHOT_MODEL = (
    "/Users/noahlyons/.cache/huggingface/hub/"
    "models--prism-ml--bonsai-8b-mlx-1bit/snapshots/"
    "d95a01f5e78184d278e21c4cfd57ff417a60ae22"
)


class SmokeVlmContract(unittest.TestCase):
    def test_smoke_vlm_defaults_narrator_model_to_full_snapshot_path(self) -> None:
        parser = smoke_vlm._build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.narrator_model, _BONSAI_SNAPSHOT_MODEL)

    def test_smoke_vlm_defaults_narrator_to_nlm2pr_bonsai(self) -> None:
        parser = smoke_vlm._build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.narrator_url, "http://nlm2pr.local:8002")

    def test_thinking_narrator_defaults_to_nlm2pr_bonsai(self) -> None:
        narrator = ThinkingNarrator(_DummySink())

        self.assertEqual(narrator._base_url, "http://nlm2pr.local:8002")

    def test_thinking_narrator_defaults_to_full_snapshot_model_path(self) -> None:
        narrator = ThinkingNarrator(_DummySink())

        self.assertEqual(narrator._model, _BONSAI_SNAPSHOT_MODEL)

    def test_validate_narrator_model_rejects_bare_snapshots_directory(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "full snapshot model path",
        ):
            smoke_vlm._validate_narrator_model(
                "/Users/noahlyons/.cache/huggingface/hub/models--prism-ml--bonsai-8b-mlx-1bit/snapshots/"
            )

    def test_validate_narrator_model_accepts_full_snapshot_path(self) -> None:
        model_path = _BONSAI_SNAPSHOT_MODEL

        resolved = smoke_vlm._validate_narrator_model(model_path)

        self.assertEqual(resolved, model_path)

    def test_scorebug_session_meta_labels_tricky_subset(self) -> None:
        parser = smoke_vlm._build_arg_parser()
        args = parser.parse_args(["--model", "gemma-4-26b-a4b-it-bf16", "--tricky"])

        meta = smoke_vlm._scorebug_session_meta(
            args=args,
            model=args.model,
            subset_count=6,
        )

        self.assertEqual(
            meta,
            {
                "model": "gemma-4-26b-a4b-it-bf16",
                "set_label": "TRICKY",
                "subset_count": 6,
            },
        )

    def test_resolve_preview_focus_region_falls_back_to_yaml_overrides(self) -> None:
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-12a",
            answer_type="lewis_structure",
            page=4,
            professor_score=1.0,
            max_points=2.0,
            professor_mark="partial",
            student_answer="O3 Lewis structure drawn",
            notes="Half annotation.",
        )
        overrides = {
            ("15-blue", "fr-12a"): FocusRegion(
                page=4,
                x=0.16,
                y=0.08,
                width=0.48,
                height=0.19,
                source="mock_tricky",
            ),
        }

        focus = smoke_vlm._resolve_preview_focus_region(
            item,
            template_document=None,
            focus_region_overrides=overrides,
        )

        self.assertIsNotNone(focus)
        assert focus is not None
        self.assertEqual(focus.source, "mock_tricky")
        self.assertGreater(focus.width, 0.0)
        self.assertGreater(focus.height, 0.0)

    def test_resolve_preview_focus_region_prefers_item_metadata_over_overrides(self) -> None:
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-12a",
            answer_type="lewis_structure",
            page=4,
            professor_score=1.0,
            max_points=2.0,
            professor_mark="partial",
            student_answer="O3 Lewis structure drawn",
            notes="Half annotation.",
            focus_region=FocusRegion(
                page=4,
                x=0.1,
                y=0.2,
                width=0.3,
                height=0.4,
                source="ground_truth",
            ),
        )
        # Overrides carries a different box; item-level metadata must win.
        overrides = {
            ("15-blue", "fr-12a"): FocusRegion(
                page=4,
                x=0.99,
                y=0.99,
                width=0.01,
                height=0.01,
                source="mock_tricky",
            ),
        }

        focus = smoke_vlm._resolve_preview_focus_region(
            item,
            template_document=None,
            focus_region_overrides=overrides,
        )

        self.assertEqual(focus, item.focus_region)

    def test_scorebug_session_meta_labels_tricky_plus_subset(self) -> None:
        parser = smoke_vlm._build_arg_parser()
        args = parser.parse_args(
            ["--model", "qwen3p5-35B-A3B", "--tricky-plus"]
        )

        meta = smoke_vlm._scorebug_session_meta(
            args=args,
            model=args.model,
            subset_count=12,
        )

        self.assertEqual(
            meta,
            {
                "model": "qwen3p5-35B-A3B",
                "set_label": "TRICKY+",
                "subset_count": 12,
            },
        )

    def test_scorebug_session_meta_labels_tricky_plus_plus_subset(self) -> None:
        parser = smoke_vlm._build_arg_parser()
        args = parser.parse_args(
            ["--model", "qwen3p5-35B-A3B", "--tricky-plus-plus"]
        )

        meta = smoke_vlm._scorebug_session_meta(
            args=args,
            model=args.model,
            subset_count=15,
        )

        self.assertEqual(
            meta,
            {
                "model": "qwen3p5-35B-A3B",
                "set_label": "TRICKY++",
                "subset_count": 15,
            },
        )

    def test_tricky_plus_runs_expansion_items_first(self) -> None:
        self.assertEqual(
            smoke_vlm._TRICKY_PLUS_PICKS[:6],
            [
                ("27-blue-2023", "fr-3"),
                ("27-blue-2023", "fr-5b"),
                ("27-blue-2023", "fr-12a"),
                ("39-blue-redacted", "fr-10a"),
                ("34-blue", "fr-8"),
                ("34-blue", "fr-12a"),
            ],
        )

    def test_tricky_plus_items_all_resolve_to_preview_regions(self) -> None:
        from auto_grader.focus_regions import (
            DEFAULT_FOCUS_REGIONS_PATH,
            load_focus_regions,
        )

        overrides = load_focus_regions(DEFAULT_FOCUS_REGIONS_PATH)
        items = [
            EvalItem(
                exam_id=exam_id,
                question_id=question_id,
                answer_type="numeric",
                page=1,
                professor_score=0.0,
                max_points=1.0,
                professor_mark="x",
                student_answer="mock",
                notes="mock",
            )
            for exam_id, question_id in smoke_vlm._TRICKY_PLUS_PICKS
        ]

        resolved = [
            smoke_vlm._resolve_preview_focus_region(
                item,
                template_document=None,
                focus_region_overrides=overrides,
            )
            for item in items
        ]

        self.assertTrue(all(region is not None for region in resolved))

    def test_tricky_plus_plus_prepends_three_more_fifteen_blue_stress_items(self) -> None:
        self.assertEqual(
            smoke_vlm._TRICKY_PLUS_PLUS_PICKS[:9],
            [
                ("15-blue", "fr-10b"),
                ("15-blue", "fr-11c"),
                ("15-blue", "fr-12b"),
                ("27-blue-2023", "fr-3"),
                ("27-blue-2023", "fr-5b"),
                ("27-blue-2023", "fr-12a"),
                ("39-blue-redacted", "fr-10a"),
                ("34-blue", "fr-8"),
                ("34-blue", "fr-12a"),
            ],
        )
    def test_run_dir_help_advertises_durable_root_outside_worktree(self) -> None:
        parser = smoke_vlm._build_arg_parser()
        run_dir_action = next(
            action
            for action in parser._actions
            if "--run-dir" in action.option_strings
        )

        self.assertIn(
            "~/dev/auto-grader-runs",
            run_dir_action.help,
        )

    def test_default_run_dir_uses_durable_root_outside_repo(self) -> None:
        run_dir = smoke_vlm._default_run_dir(
            "qwen3p5-35B-A3B",
            now=smoke_vlm.datetime(2026, 4, 10, 21, 30, 45),
        )

        self.assertEqual(
            run_dir,
            Path.home() / "dev" / "auto-grader-runs" / "20260410-213045-qwen3p5-35B-A3B",
        )

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
            score_basis="Wrong stoichiometric method, so no credit.",
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
            self.assertEqual(
                record["score_basis"],
                "Wrong stoichiometric method, so no credit.",
                "prediction record must persist score_basis for downstream analysis",
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

    def test_main_handles_zero_narrator_dispatches_without_crashing(self):
        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-1",
            answer_type="numeric",
            page=1,
            professor_score=1.0,
            max_points=1.0,
            professor_mark="check",
            student_answer="6.98 g/mL",
            notes="",
        )
        prediction = Prediction(
            exam_id="15-blue",
            question_id="fr-1",
            model_score=1.0,
            model_confidence=0.95,
            model_reasoning="clean read",
            model_read="6.98 g/mL",
            raw_assistant='{"model_score": 1.0}',
            raw_reasoning="",
            upstream_dependency="none",
            if_dependent_then_consistent=None,
        )

        class _FakeManifest:
            def write_status(self, *_args, **_kwargs) -> None:
                return None

        class _FakeSink:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeWriter:
            failures = 0
            last_failure_msg = ""
            _count = 1

            def __init__(self, path: Path, *, model: str, run_dir: Path):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write_one(self, item, pred) -> None:
                return None

        class _FakeNarrator:
            def __init__(self, *_args, **_kwargs):
                return None

            def wrap_up(self, *_args, **_kwargs) -> None:
                return None

            def stats(self) -> dict[str, int]:
                return {
                    "items_started": 1,
                    "dispatches_total": 0,
                    "summaries_emitted": 0,
                    "drops_dedup": 0,
                    "drops_empty": 0,
                    "max_dispatches_one_item": 0,
                }

        report = SimpleNamespace(
            total_scored=1,
            unclear_excluded=0,
            overall_exact_accuracy=1.0,
            overall_tolerance_accuracy=1.0,
            false_positives=0,
            false_negatives=0,
            total_points_possible=1.0,
            total_points_truth=1.0,
            calibration_bins=[],
            per_answer_type_exact={"numeric": 1.0},
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            argv = [
                "smoke_vlm.py",
                "--narrate-stderr",
                "--pick",
                "15-blue:fr-1",
                "--run-dir",
                str(run_dir),
            ]
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "load_ground_truth", return_value=[item])
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "load_focus_regions", return_value={})
                )
                stack.enter_context(
                    mock.patch.object(
                        smoke_vlm,
                        "apply_model_sampling_preset",
                        side_effect=lambda config, **_: config,
                    )
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "_load_template_document", return_value=None)
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "_run_identity", return_value=("run-1", run_dir))
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "_build_manifest", return_value=_FakeManifest())
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "NarratorSink", return_value=_FakeSink())
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "_PredictionWriter", _FakeWriter)
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "ThinkingNarrator", _FakeNarrator)
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "grade_all_items", return_value=[prediction])
                )
                stack.enter_context(
                    mock.patch.object(smoke_vlm, "score_predictions", return_value=report)
                )
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = smoke_vlm.main()

        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("drop rate:            0.0%", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
