from __future__ import annotations

import unittest

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

    def write_topic(self, text: str, verdict: str | None = None) -> None:
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


if __name__ == "__main__":
    unittest.main()
