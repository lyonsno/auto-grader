from __future__ import annotations

import errno
import io
import json
import tempfile
import unittest
import base64
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

    def test_spawn_runner_uses_checkout_python_not_uv_run(self):
        sink = NarratorSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch("subprocess.run"):
                sink._spawn_terminal_window(fifo)

            runner = fifo.parent / "run.sh"
            script = runner.read_text()

        self.assertIn(".venv/bin/python", script)
        self.assertNotIn("uv run python", script)

    def test_spawn_runner_clears_inline_preview_diagnostic_flag(self):
        sink = NarratorSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch("subprocess.run"):
                sink._spawn_terminal_window(fifo)

            runner = fifo.parent / "run.sh"
            script = runner.read_text()

        self.assertIn(
            "unset PAINT_DRY_NO_INLINE_IMAGES",
            script,
            "spawned Paint Dry windows must not inherit the shell-level "
            "diagnostic flag that forces the old half-block fallback",
        )

    def test_spawn_raises_if_wezterm_cli_hangs(self):
        sink = NarratorSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()
            timeout = __import__("subprocess").TimeoutExpired(
                cmd=["wezterm", "cli", "spawn"],
                timeout=5.0,
            )

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch(
                "subprocess.run",
                side_effect=[mock.Mock(returncode=0), timeout],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Timed out spawning WezTerm window for narrator",
                ):
                    sink._spawn_terminal_window(fifo)

    def test_spawn_runner_captures_reader_stderr_for_crash_diagnosis(self):
        sink = NarratorSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch("subprocess.run"):
                sink._spawn_terminal_window(fifo)

            runner = fifo.parent / "run.sh"
            script = runner.read_text()

        self.assertIn("reader.stderr", script)
        self.assertIn("Press Enter to close", script)
        self.assertNotIn("reader.stdout", script)
        launch_line = next(
            line for line in script.splitlines() if "narrator_reader.py" in line
        )
        self.assertNotRegex(launch_line, r"(^|\\s)(?:1>|>)")
        self.assertNotIn(" 1>", launch_line)
        self.assertIn("2>", launch_line)

    def test_spawn_runner_persists_reader_diagnostics_in_log_dir_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as logdir:
            sink = NarratorSink(SinkConfig(log_dir=Path(logdir)))
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            with mock.patch.object(
                sink,
                "_resolve_wezterm_executable",
                return_value="/Applications/WezTerm.app/Contents/MacOS/wezterm",
            ), mock.patch("subprocess.run"):
                sink._spawn_terminal_window(fifo)

            runner = fifo.parent / "run.sh"
            script = runner.read_text()

        self.assertIn(str(Path(logdir) / "reader.stderr"), script)
        self.assertIn(str(Path(logdir) / "reader.exit"), script)
        self.assertNotIn(str(fifo.parent / "reader.stderr"), script)

    def test_start_raises_if_reader_never_connects_to_fifo(self):
        sink = NarratorSink(SinkConfig(spawn_terminal=True))
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "narrator.fifo"
            fifo.touch()

            now = [0.0]

            with mock.patch.object(sink, "_make_fifo", return_value=fifo), mock.patch.object(
                sink, "_spawn_terminal_window"
            ), mock.patch("os.open", side_effect=OSError(errno.ENXIO, "no reader")), mock.patch(
                "time.monotonic", side_effect=lambda: now[0]
            ), mock.patch(
                "time.sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Narrator reader did not connect to the FIFO",
                ):
                    sink.start()

    def test_fifo_writer_switches_back_to_blocking_mode_after_connect(self):
        sink = NarratorSink(SinkConfig(spawn_terminal=True))
        fifo = Path("/tmp/mock-narrator.fifo")

        with mock.patch("os.open", return_value=123), mock.patch(
            "os.set_blocking"
        ) as set_blocking_mock, mock.patch(
            "os.fdopen", return_value=mock.sentinel.writer
        ):
            writer = sink._open_fifo_writer_with_timeout(fifo)

        self.assertIs(writer, mock.sentinel.writer)
        set_blocking_mock.assert_called_once_with(123, True)

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
                "31s · Grader: 1/3. Prof: 1.5/3. · Within acceptable range, below ceiling.",
                verdict="within_band",
                grader_score=1.0,
                truth_score=1.5,
                max_points=3.0,
                acceptable_score_floor=1.0,
                acceptable_score_ceiling=1.5,
            )
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        topic = next(event for event in events if event["type"] == "topic")
        self.assertEqual(topic["verdict"], "within_band")
        self.assertEqual(topic["grader_score"], 1.0)
        self.assertEqual(topic["truth_score"], 1.5)
        self.assertEqual(topic["max_points"], 3.0)
        self.assertEqual(topic["acceptable_score_floor"], 1.0)
        self.assertEqual(topic["acceptable_score_ceiling"], 1.5)

    def test_fifo_writer_drop_persists_diagnostic_in_log_dir(self):
        class _BrokenWriter:
            def write(self, _line: str) -> None:
                raise BrokenPipeError("reader vanished")

            def flush(self) -> None:
                raise AssertionError("flush should not run after write failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink._fifo_writer = _BrokenWriter()

            sink.write_header("[item 1/15] 15-blue/fr-10b (numeric, 1.0 pts)")

            diag_path = Path(tmpdir) / "fifo_writer_failure.txt"
            self.assertTrue(
                diag_path.exists(),
                "writer-side FIFO loss should leave a durable breadcrumb in the run dir",
            )
            diag = diag_path.read_text()
            self.assertIn("event=header", diag)
            self.assertIn("BrokenPipeError", diag)
            self.assertIsNone(
                sink._fifo_writer,
                "sink should stop using the dead FIFO writer after recording the failure",
            )
            sink.close()

    def test_write_focus_preview_emits_base64_preview_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_focus_preview(
                b"preview-bytes",
                label="15-blue/fr-12a",
                source="mock_tricky",
            )
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        preview = next(event for event in events if event["type"] == "focus_preview")
        self.assertEqual(preview["label"], "15-blue/fr-12a")
        self.assertEqual(preview["source"], "mock_tricky")
        self.assertEqual(
            base64.b64decode(preview["png_base64"]),
            b"preview-bytes",
        )

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

    def test_write_basis_emits_basis_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_basis("Correct setup, lost credit for octet violation.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        basis = next(event for event in events if event["type"] == "basis")
        self.assertEqual(
            basis["text"],
            "Correct setup, lost credit for octet violation.",
        )

    def test_write_review_marker_emits_review_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_review_marker("Human review warranted.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        review = next(event for event in events if event["type"] == "review_marker")
        self.assertEqual(review["text"], "Human review warranted.")


if __name__ == "__main__":
    unittest.main()
