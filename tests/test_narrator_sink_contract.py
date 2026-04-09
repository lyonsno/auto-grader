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


if __name__ == "__main__":
    unittest.main()
