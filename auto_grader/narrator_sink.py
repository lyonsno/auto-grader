"""Narrator sink — fanout for the Project Paint Dry display.

Owns:
- The fifo that the rich-display reader process tails (when --narrator-window)
- The Terminal.app window spawn (macOS osascript)
- A JSONL log file (machine-replayable transcript of every event)
- A plain-text transcript file (human-skimmable)

The narrator code (ThinkingNarrator) talks ONLY to a NarratorSink. The
sink doesn't care whether tokens come from one bonsai call or many — it
just routes events to its outputs. JSON line protocol over the fifo:

    {"type": "header", "text": "[item 1/38] 15-blue/fr-1 ..."}
    {"type": "delta",  "text": "Reading"}
    {"type": "delta",  "text": " the"}
    ...
    {"type": "commit", "mode": "thought"}
    {"type": "topic",  "text": "Thought for 47s · density calc"}
    {"type": "end"}
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO


@dataclass
class SinkConfig:
    spawn_terminal: bool = False
    log_dir: Path | None = None  # if set, JSONL + .txt are written here
    fallback_stream: IO[str] | None = None  # used when no terminal
    session_meta: dict | None = None  # one-shot session metadata for the reader


class NarratorSink:
    """Fan-out destination for narrator events.

    Use as a context manager:

        with NarratorSink(config) as sink:
            sink.write_header("[item 1/38] ...")
            sink.write_delta("Reading")
            sink.write_delta(" the")
            sink.commit_live()
            sink.write_topic("Thought for 47s · density calc")
    """

    def __init__(self, config: SinkConfig | None = None):
        self.config = config or SinkConfig()
        self._fifo_path: Path | None = None
        self._fifo_writer: IO[str] | None = None
        self._owns_tmpdir: Path | None = None
        self._jsonl_file: IO[str] | None = None
        self._txt_file: IO[str] | None = None
        self._fallback = self.config.fallback_stream or sys.stderr
        self._live_buffer = ""  # accumulator for the txt transcript only
        self._lock = threading.Lock()
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "NarratorSink":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        # Set up log files
        if self.config.log_dir is not None:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(
                self.config.log_dir / "narrator.jsonl", "w", buffering=1
            )
            self._txt_file = open(
                self.config.log_dir / "narrator.txt", "w", buffering=1
            )
            self._txt_file.write(
                f"# Project Paint Dry transcript\n"
                f"# Started: {datetime.now().isoformat(timespec='seconds')}\n\n"
            )

        if self.config.spawn_terminal:
            self._fifo_path = self._make_fifo()
            self._spawn_terminal_window(self._fifo_path)
            # Open the fifo for writing — blocks until reader connects.
            # Use line buffering so messages flush on \n.
            self._fifo_writer = open(self._fifo_path, "w", buffering=1)

        if self.config.session_meta:
            self._emit({"type": "session_meta", **self.config.session_meta})

    def close(self) -> None:
        if not self._started:
            return
        try:
            self._emit({"type": "end"})
        except Exception:
            pass
        if self._fifo_writer is not None:
            try:
                self._fifo_writer.close()
            except Exception:
                pass
            self._fifo_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._txt_file is not None:
            self._txt_file.write(
                f"\n# Ended: {datetime.now().isoformat(timespec='seconds')}\n"
            )
            self._txt_file.close()
            self._txt_file = None
        if self._owns_tmpdir is not None:
            try:
                if self._fifo_path is not None:
                    self._fifo_path.unlink(missing_ok=True)
                for f in self._owns_tmpdir.iterdir():
                    f.unlink(missing_ok=True)
                self._owns_tmpdir.rmdir()
            except Exception:
                pass
        self._started = False

    # -- public api --------------------------------------------------------

    def write_header(self, text: str) -> None:
        """Write a section header (item boundary) to the sink."""
        with self._lock:
            self._emit({"type": "header", "text": text})
            if self._txt_file is not None:
                self._txt_file.write(f"\n{text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"\n{text}\n")
                self._fallback.flush()

    def write_delta(self, text: str, *, mode: str = "thought") -> None:
        """Append a token delta to the live line."""
        if not text:
            return
        with self._lock:
            self._emit({"type": "delta", "text": text, "mode": mode})
            self._live_buffer += text
            if not self.config.spawn_terminal:
                self._fallback.write(text)
                self._fallback.flush()

    def commit_live(self, *, mode: str = "thought") -> None:
        """Finalize the current live line and push it into the history."""
        with self._lock:
            self._emit({"type": "commit", "mode": mode})
            if self._txt_file is not None and self._live_buffer:
                self._txt_file.write(f"  {self._live_buffer}\n")
            if not self.config.spawn_terminal:
                self._fallback.write("\n")
                self._fallback.flush()
            self._live_buffer = ""

    def write_topic(
        self,
        text: str,
        verdict: str | None = None,
        *,
        grader_score: float | None = None,
        truth_score: float | None = None,
        max_points: float | None = None,
    ) -> None:
        """Write the per-item collapsed 'Thought for Xs · topic' line.

        verdict is one of "match", "overshoot", "undershoot", or None
        (when unknown). The reader uses it to color the topic line —
        cool sage for match, warm coral for overshoot, warm amber for
        undershoot, dusty plum for unknown. Optional structured scoring
        metadata lets the reader maintain a running scoreboard without
        reverse-engineering numbers back out of the prose after-action line.
        """
        with self._lock:
            msg = {"type": "topic", "text": text}
            if verdict is not None:
                msg["verdict"] = verdict
            if grader_score is not None:
                msg["grader_score"] = grader_score
            if truth_score is not None:
                msg["truth_score"] = truth_score
            if max_points is not None:
                msg["max_points"] = max_points
            self._emit(msg)
            if self._txt_file is not None:
                self._txt_file.write(f"  -> {text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"  -> {text}\n")
                self._fallback.flush()

    def write_drop(self, reason: str, text: str) -> None:
        """Record a dropped summary (dedup, empty, etc.) for observability.

        Drops appear in the JSONL log with their reason + the dropped text,
        and surface as faint lines in the rich display so the user can see
        the narrator was working but the line was rejected. They are NOT
        written to the .txt transcript (which is the canonical accepted feed).
        """
        with self._lock:
            self._emit({"type": "drop", "reason": reason, "text": text})
            if not self.config.spawn_terminal:
                self._fallback.write(f"  [drop:{reason}] {text}\n")
                self._fallback.flush()

    def write_focus_preview(
        self,
        png_bytes: bytes,
        *,
        label: str | None = None,
        source: str | None = None,
    ) -> None:
        """Emit a terminal-preview image for the current item.

        Preview images are JSONL/fifo-only. They intentionally do not
        go into the plain-text transcript because the transcript is the
        accepted textual narrator feed, not an image log.
        """
        if not png_bytes:
            return
        with self._lock:
            msg = {
                "type": "focus_preview",
                "png_base64": base64.b64encode(png_bytes).decode("ascii"),
            }
            if label:
                msg["label"] = label
            if source:
                msg["source"] = source
            self._emit(msg)

    def rollback_live(self) -> None:
        """Discard the in-flight live line without committing it to history.

        Used when a dispatched line is going to be dropped (dedup) — we
        still want the user to have seen the typewriter effect, but the
        line should not commit to history. The reader clears the live row.
        """
        with self._lock:
            self._emit({"type": "rollback_live"})
            self._live_buffer = ""
            if not self.config.spawn_terminal:
                # Stderr fallback can't unwrite chars; just newline so the
                # next output starts fresh.
                self._fallback.write("\n")
                self._fallback.flush()

    def start_wrap_up(self) -> None:
        """Signal that the post-game wrap-up generation has started.

        Sent the moment narrator.wrap_up() begins, BEFORE the (slow)
        chat completion call to the grader server. The reader uses
        this to immediately show a 'writing post-game commentary...'
        placeholder in the post-game panel so the user knows the
        script is alive and working on the wrap-up rather than hung.
        """
        with self._lock:
            self._emit({"type": "wrap_up_pending"})

    def write_wrap_up(self, text: str) -> None:
        """Final color-commentary line at the end of the run."""
        with self._lock:
            self._emit({"type": "wrap_up", "text": text})
            if self._txt_file is not None:
                self._txt_file.write(f"\n=== WRAP-UP ===\n{text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"\n=== WRAP-UP ===\n{text}\n")
                self._fallback.flush()

    # -- internals ---------------------------------------------------------

    def _emit(self, msg: dict) -> None:
        """Send one JSON message to fifo + jsonl. Caller holds the lock."""
        msg = {"ts": time.time(), **msg}
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        if self._jsonl_file is not None:
            self._jsonl_file.write(line)
        if self._fifo_writer is not None:
            try:
                self._fifo_writer.write(line)
                self._fifo_writer.flush()
            except (BrokenPipeError, OSError):
                # Reader closed — we'll keep logging to disk only
                self._fifo_writer = None

    @staticmethod
    def _make_fifo() -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="paint-dry-"))
        fifo = tmp / "narrator.fifo"
        os.mkfifo(fifo)
        return fifo

    @staticmethod
    def _resolve_wezterm_executable() -> str:
        """Find the WezTerm CLI binary.

        Prefer the shell PATH, but fall back to the standard macOS app-bundle
        binary when the CLI shim is not installed.
        """
        from_path = shutil.which("wezterm")
        if from_path:
            return from_path

        macos_app_binary = "/Applications/WezTerm.app/Contents/MacOS/wezterm"
        if Path(macos_app_binary).exists():
            return macos_app_binary

        raise FileNotFoundError(
            "wezterm not found on PATH and macOS app binary is missing at "
            f"{macos_app_binary}"
        )

    def _spawn_terminal_window(self, fifo: Path) -> None:
        """Open a WezTerm window running the rich reader script.

        Spawns a new window in the existing WezTerm instance via
        `wezterm cli spawn --new-window`. WezTerm supports 24-bit
        color escape codes (`\\e[38;2;R;G;B m`); Apple Terminal.app
        does not, and previously misparsed truecolor escapes as
        256-color palette indices, producing neon magenta/cyan
        artifacts in place of the narrator's actual palette.

        Sidesteps shell-quoting hell by writing a real shell script to disk.
        """
        # Locate the reader script relative to this file
        reader_script = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "narrator_reader.py"
        )
        if not reader_script.exists():
            raise RuntimeError(
                f"narrator_reader.py not found at {reader_script}"
            )

        self._reap_existing_viewers(reader_script)

        # Project root for uv run
        project_root = Path(__file__).resolve().parent.parent

        runner = fifo.parent / "run.sh"
        runner.write_text(
            "#!/bin/bash\n"
            f"cd {project_root}\n"
            f"exec uv run python {reader_script} {fifo}\n"
        )
        runner.chmod(0o755)
        self._owns_tmpdir = fifo.parent

        try:
            wezterm = self._resolve_wezterm_executable()
            subprocess.run(
                [
                    wezterm, "cli", "spawn", "--new-window",
                    "--", str(runner),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = getattr(e, "stderr", "") or ""
            raise RuntimeError(
                f"Could not spawn WezTerm window for narrator: {e}\n"
                f"stderr: {stderr}\n"
                "(requires WezTerm installed and a running WezTerm instance)"
            )

    @staticmethod
    def _reap_existing_viewers(reader_script: Path) -> None:
        """Kill stale narrator readers before spawning a new viewer.

        The reader runs a 30 FPS truecolor refresh loop, so letting old
        windows accumulate can turn into a real CPU problem for WezTerm.
        Reap by the concrete script path so we only target Paint Dry
        readers from this checkout lineage.
        """
        subprocess.run(
            ["pkill", "-f", str(reader_script)],
            check=False,
            capture_output=True,
            text=True,
        )
