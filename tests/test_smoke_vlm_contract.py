from __future__ import annotations

import unittest
from pathlib import Path

from auto_grader.eval_harness import EvalItem, FocusRegion
from auto_grader.thinking_narrator import ThinkingNarrator
from scripts import smoke_vlm


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


class SmokeVlmContract(unittest.TestCase):
    def test_smoke_vlm_defaults_narrator_to_nlmb2p_bonsai(self) -> None:
        parser = smoke_vlm._build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.narrator_url, "http://nlmb2p.local:8002")

    def test_thinking_narrator_defaults_to_nlmb2p_bonsai(self) -> None:
        narrator = ThinkingNarrator(_DummySink())

        self.assertEqual(narrator._base_url, "http://nlmb2p.local:8002")

    def test_validate_narrator_model_rejects_bare_snapshots_directory(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "full snapshot model path",
        ):
            smoke_vlm._validate_narrator_model(
                "/Users/noahlyons/.cache/huggingface/hub/models--prism-ml--bonsai-8b-mlx-1bit/snapshots/"
            )

    def test_validate_narrator_model_accepts_full_snapshot_path(self) -> None:
        model_path = (
            "/Users/noahlyons/.cache/huggingface/hub/models--prism-ml--bonsai-8b-mlx-1bit/"
            "snapshots/d95a01f5e78184d278e21c4cfd57ff417a60ae22"
        )

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

    def test_resolve_preview_focus_region_falls_back_to_mock_tricky_map(self) -> None:
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

        focus = smoke_vlm._resolve_preview_focus_region(item, template_document=None)

        self.assertIsNotNone(focus)
        assert focus is not None
        self.assertEqual(focus.source, "mock_tricky")
        self.assertGreater(focus.width, 0.0)
        self.assertGreater(focus.height, 0.0)

    def test_resolve_preview_focus_region_prefers_item_metadata_over_mock(self) -> None:
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

        focus = smoke_vlm._resolve_preview_focus_region(item, template_document=None)

        self.assertEqual(focus, item.focus_region)

    def test_mock_tricky_focus_regions_are_not_ribbon_thin(self) -> None:
        for focus in smoke_vlm._TRICKY_FOCUS_REGION_MOCKS.values():
            self.assertGreaterEqual(
                focus.height,
                0.18,
                "mock tricky previews should be tall enough to read as focused crops, not banner strips",
            )
            self.assertLessEqual(
                focus.width / focus.height,
                3.5,
                "mock tricky previews should avoid ultra-wide aspect ratios until we have real boxes",
            )

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
            )
            for item in items
        ]

        self.assertTrue(all(region is not None for region in resolved))

    def test_lewis_mock_focus_regions_hug_the_student_work_not_the_whole_page(self) -> None:
        for key in [
            ("15-blue", "fr-12a"),
            ("27-blue-2023", "fr-12a"),
            ("34-blue", "fr-12a"),
        ]:
            focus = smoke_vlm._TRICKY_FOCUS_REGION_MOCKS[key]
            self.assertGreaterEqual(
                focus.y,
                0.06,
                "Lewis mock boxes should start below the top margin so they stop spending preview budget on blank paper",
            )
            self.assertLessEqual(
                focus.width,
                0.54,
                "Lewis mock boxes should tighten around the resonance drawings instead of carrying a page-wide banner crop",
            )
            self.assertLessEqual(
                focus.height,
                0.24,
                "Lewis mock boxes should exclude the next question block so the crop reads like the graded work, not a mini page",
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


if __name__ == "__main__":
    unittest.main()
