from __future__ import annotations

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

    def test_write_ambiguity_emits_ambiguity_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_ambiguity("Scan leaves the coefficient ambiguous.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        ambiguity = next(event for event in events if event["type"] == "ambiguity")
        self.assertEqual(ambiguity["text"], "Scan leaves the coefficient ambiguous.")

    def test_write_deduction_emits_deduction_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_deduction("Lost credit for missing net ionic form.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        deduction = next(event for event in events if event["type"] == "deduction")
        self.assertEqual(deduction["text"], "Lost credit for missing net ionic form.")

    def test_write_credit_preserved_emits_credit_preserved_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_credit_preserved("Correct setup and carry-forward handling.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        credit = next(
            event for event in events if event["type"] == "credit_preserved"
        )
        self.assertEqual(
            credit["text"], "Correct setup and carry-forward handling."
        )

    def test_write_professor_mismatch_emits_professor_mismatch_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = NarratorSink(
                SinkConfig(log_dir=Path(tmpdir), fallback_stream=io.StringIO())
            )
            sink.start()
            sink.write_professor_mismatch("Historical professor awarded 2/4; corrected truth is 4/4.")
            sink.close()

            events = [
                json.loads(line)
                for line in (Path(tmpdir) / "narrator.jsonl").read_text().splitlines()
            ]

        mismatch = next(
            event for event in events if event["type"] == "professor_mismatch"
        )
        self.assertEqual(
            mismatch["text"],
            "Historical professor awarded 2/4; corrected truth is 4/4.",
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
