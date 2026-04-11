from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_grader.narrator_sink import NarratorSink, SinkConfig


class TestWezTermResolution(unittest.TestCase):
    def test_falls_back_to_macos_app_binary_when_path_missing(self):
        sink = NarratorSink()
        with mock.patch("shutil.which", return_value=None):
            resolved = sink._resolve_wezterm_executable()
        self.assertEqual(
            resolved,
            "/Applications/WezTerm.app/Contents/MacOS/wezterm",
        )

    def test_spawn_reaps_existing_narrator_reader_before_opening_new_window(self):
        sink = NarratorSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch("subprocess.run") as run_mock:
                sink._spawn_terminal_window(fifo)

        self.assertGreaterEqual(run_mock.call_count, 2)
        reap_args = run_mock.call_args_list[0].args[0]
        self.assertEqual(reap_args[:2], ["pkill", "-f"])
        self.assertIn("narrator_reader.py", reap_args[2])
        spawn_args = run_mock.call_args_list[1].args[0]
        self.assertEqual(
            spawn_args[:4],
            [
                "/Applications/WezTerm.app/Contents/MacOS/wezterm",
                "cli",
                "spawn",
                "--new-window",
            ],
        )

    def test_commit_live_can_emit_status_mode_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_delta("Tracing")
            sink.commit_live(mode="status")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        commit = next(event for event in events if event["type"] == "commit")
        self.assertEqual(commit["mode"], "status")

    def test_write_delta_can_emit_status_mode_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_delta("Tracing", mode="status")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        delta = next(event for event in events if event["type"] == "delta")
        self.assertEqual(delta["mode"], "status")

    def test_start_can_emit_session_meta_for_model_scorebug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(
                    log_dir=Path(tmpdir),
                    fallback_stream=io.StringIO(),
                    session_meta={
                        "model": "qwen3p5-35B-A3B",
                        "set_label": "TRICKY",
                    },
                )
            )
            sink.start()
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        meta = next(event for event in events if event["type"] == "session_meta")
        self.assertEqual(meta["model"], "qwen3p5-35B-A3B")
        self.assertEqual(meta["set_label"], "TRICKY")

    def test_write_topic_can_emit_scorebug_tally_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_topic(
                "31s · Grader: 0/4. Prof: 0/4.",
                verdict="match",
                grader_score=0.0,
                truth_score=0.0,
                max_points=4.0,
            )
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        topic = next(event for event in events if event["type"] == "topic")
        self.assertEqual(topic["verdict"], "match")
        self.assertEqual(topic["grader_score"], 0.0)
        self.assertEqual(topic["truth_score"], 0.0)
        self.assertEqual(topic["max_points"], 4.0)

    def test_write_checkpoint_emits_checkpoint_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_checkpoint(
                "Core issue: ozone drawing misses resonance and central octet."
            )
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        checkpoint = next(event for event in events if event["type"] == "checkpoint")
        self.assertEqual(
            checkpoint["text"],
            "Core issue: ozone drawing misses resonance and central octet.",
        )

    def test_structured_rows_emit_named_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_basis("Correct setup, lost credit for octet violation.")
            sink.write_credit_preserved("Correct stoichiometric setup and unit basis.")
            sink.write_deduction("Boxed answer adds reactant moles instead of NH3.")
            sink.write_review_marker("Human review warranted after bounded ambiguity pass.")
            sink.write_professor_mismatch("Historical professor awarded 2/2; corrected truth is 0/2.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        self.assertEqual(
            [
                event["type"]
                for event in events
                if event["type"]
                in {
                    "basis",
                    "credit_preserved",
                    "deduction",
                    "review_marker",
                    "professor_mismatch",
                }
            ],
            [
                "basis",
                "credit_preserved",
                "deduction",
                "review_marker",
                "professor_mismatch",
            ],
        )


if __name__ == "__main__":
    unittest.main()
