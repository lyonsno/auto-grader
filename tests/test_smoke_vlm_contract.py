from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
