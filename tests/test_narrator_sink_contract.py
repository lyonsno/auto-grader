from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_grader.narrator_sink import NarratorSink


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


if __name__ == "__main__":
    unittest.main()
