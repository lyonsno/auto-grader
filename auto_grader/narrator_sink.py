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
    {"type": "commit"}
    {"type": "topic",  "text": "Thought for 47s · density calc"}
    {"type": "end"}
"""

from __future__ import annotations

import json
import os
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

    def write_delta(self, text: str) -> None:
        """Append a token delta to the live line."""
        if not text:
            return
        with self._lock:
            self._emit({"type": "delta", "text": text})
            self._live_buffer += text
            if not self.config.spawn_terminal:
                self._fallback.write(text)
                self._fallback.flush()

    def commit_live(self) -> None:
        """Finalize the current live line and push it into the history."""
        with self._lock:
            self._emit({"type": "commit"})
            if self._txt_file is not None and self._live_buffer:
                self._txt_file.write(f"  {self._live_buffer}\n")
            if not self.config.spawn_terminal:
                self._fallback.write("\n")
                self._fallback.flush()
            self._live_buffer = ""

    def write_topic(self, text: str) -> None:
        """Write the per-item collapsed 'Thought for Xs · topic' line."""
        with self._lock:
            self._emit({"type": "topic", "text": text})
            if self._txt_file is not None:
                self._txt_file.write(f"  -> {text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"  -> {text}\n")
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

    def _spawn_terminal_window(self, fifo: Path) -> None:
        """Open a Terminal.app window running the rich reader script.

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

        script = f'tell application "Terminal" to do script "{runner}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["osascript", "-e", 'tell application "Terminal" to activate'],
                check=False,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = getattr(e, "stderr", "") or ""
            raise RuntimeError(
                f"Could not spawn Terminal.app window for narrator: {e}\n"
                f"stderr: {stderr}"
            )
