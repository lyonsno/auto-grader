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
    {"type": "basis", "text": "Correct setup, lost credit for units."}
    {"type": "review_marker", "text": "Human review warranted."}
    {"type": "end"}
"""

from __future__ import annotations

import base64
import errno
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
    _FIFO_CONNECT_TIMEOUT_S = 5.0
    _FIFO_CONNECT_POLL_S = 0.05
    _WEZTERM_SPAWN_TIMEOUT_S = 5.0

    """Fan-out destination for narrator events.

    Use as a context manager:

        with NarratorSink(config) as sink:
            sink.write_header("[item 1/38] ...")
            sink.write_delta("Reading")
            sink.write_delta(" the")
            sink.commit_live()
            sink.write_topic("Thought for 47s · density calc")
            sink.write_basis("Correct setup, lost credit for units.")
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
        self._fifo_failure_path: Path | None = None

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
            self._fifo_failure_path = self.config.log_dir / "fifo_writer_failure.txt"

        if self.config.spawn_terminal:
            self._fifo_path = self._make_fifo()
            self._spawn_terminal_window(self._fifo_path)
            self._fifo_writer = self._open_fifo_writer_with_timeout(self._fifo_path)

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

    def write_checkpoint(self, text: str) -> None:
        """Write a compact history checkpoint without touching the live row."""
        with self._lock:
            self._emit({"type": "checkpoint", "text": text})
            if self._txt_file is not None:
                self._txt_file.write(f"  ≈ {text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"  ≈ {text}\n")
                self._fallback.flush()

    def write_basis(self, text: str) -> None:
        self._write_structured_row("basis", "Basis", text)

    def write_ambiguity(self, text: str) -> None:
        self._write_structured_row("ambiguity", "Ambiguity", text)

    def write_credit_preserved(self, text: str) -> None:
        self._write_structured_row(
            "credit_preserved", "Credit preserved for", text
        )

    def write_deduction(self, text: str) -> None:
        self._write_structured_row("deduction", "Deduction", text)

    def write_review_marker(self, text: str) -> None:
        self._write_structured_row("review_marker", "Review needed", text)

    def write_professor_mismatch(self, text: str) -> None:
        self._write_structured_row(
            "professor_mismatch", "Professor mismatch", text
        )

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
            except (BrokenPipeError, OSError) as exc:
                self._record_fifo_writer_failure(msg.get("type"), exc)
                # Reader closed — we'll keep logging to disk only
                self._fifo_writer = None

    def _record_fifo_writer_failure(self, event_type: str | None, exc: BaseException) -> None:
        detail = (
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"event={event_type or 'unknown'} "
            f"error={type(exc).__name__}: {exc}\n"
        )
        try:
            self._fallback.write(f"[narrator fifo lost] {detail}")
            self._fallback.flush()
        except Exception:
            pass
        if self._fifo_failure_path is not None:
            try:
                with open(self._fifo_failure_path, "a", buffering=1) as fh:
                    fh.write(detail)
            except Exception:
                pass

    def _write_structured_row(self, event_type: str, label: str, text: str) -> None:
        """Write one structured legibility row under the current item."""
        if not text:
            return
        with self._lock:
            self._emit({"type": event_type, "text": text})
            if self._txt_file is not None:
                self._txt_file.write(f"  {label}: {text}\n")
            if not self.config.spawn_terminal:
                self._fallback.write(f"  {label}: {text}\n")
                self._fallback.flush()

    @staticmethod
    def _make_fifo() -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="paint-dry-"))
        fifo = tmp / "narrator.fifo"
        os.mkfifo(fifo)
        return fifo

    def _open_fifo_writer_with_timeout(self, fifo: Path) -> IO[str]:
        """Open the writer side of the narrator FIFO without wedging forever.

        A plain blocking open() can hang the whole smoke run indefinitely
        if the reader window fails to boot or exits before connecting.
        Retry a non-blocking open for a short bounded window, then fail
        loudly so the operator gets a real startup error instead of a
        silent stall at 'Run dir:'.
        """
        deadline = time.monotonic() + self._FIFO_CONNECT_TIMEOUT_S
        last_err: OSError | None = None
        while time.monotonic() < deadline:
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                os.set_blocking(fd, True)
                return os.fdopen(fd, "w", buffering=1)
            except OSError as exc:
                last_err = exc
                if exc.errno not in {errno.ENXIO, errno.ENOENT}:
                    raise RuntimeError(
                        f"Could not open narrator FIFO writer at {fifo}: {exc}"
                    ) from exc
                time.sleep(self._FIFO_CONNECT_POLL_S)

        detail = f"{last_err}" if last_err is not None else "reader never attached"
        raise RuntimeError(
            "Narrator reader did not connect to the FIFO within "
            f"{self._FIFO_CONNECT_TIMEOUT_S:.1f}s: {detail}"
        )

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
        project_python = project_root / ".venv" / "bin" / "python"
        if not project_python.exists():
            raise RuntimeError(
                f"project python not found at {project_python}"
            )

        runner = fifo.parent / "run.sh"
        diagnostics_dir = self.config.log_dir or fifo.parent
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        reader_stderr = diagnostics_dir / "reader.stderr"
        reader_exit = diagnostics_dir / "reader.exit"
        runner.write_text(
            "#!/bin/bash\n"
            "set -u\n"
            f"cd {project_root}\n"
            "unset PAINT_DRY_NO_INLINE_IMAGES\n"
            "if [ -r /dev/tty ]; then\n"
            "  exec </dev/tty\n"
            "fi\n"
            f"\"{project_python}\" \"{reader_script}\" \"{fifo}\" "
            f"2>\"{reader_stderr}\"\n"
            "status=$?\n"
            f"printf '%s\\n' \"$status\" >\"{reader_exit}\"\n"
            "if [ \"$status\" -ne 0 ]; then\n"
            "  echo \"Project Paint Dry reader exited with status $status\"\n"
            f"  echo \"stderr log: {reader_stderr}\"\n"
            f"  echo \"exit log: {reader_exit}\"\n"
            "  if [ -s "
            f"\"{reader_stderr}\""
            " ]; then\n"
            f"    tail -n 80 \"{reader_stderr}\"\n"
            "  fi\n"
            "  echo \"Press Enter to close\"\n"
            "  read -r _\n"
            "fi\n"
            "exit \"$status\"\n"
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
                timeout=self._WEZTERM_SPAWN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "Timed out spawning WezTerm window for narrator after "
                f"{self._WEZTERM_SPAWN_TIMEOUT_S:.1f}s: {e}"
            ) from e
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
