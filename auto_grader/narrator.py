"""Project Paint Dry — Bonsai narrator sidecar for the eval harness.

A small fast model rides alongside the grader, narrating each scoring
event in sportscaster present-participle prose. Streams to stderr, a
fifo, or — most fun — its own freshly-spawned Terminal.app window.

The narrator runs in its own thread with a queue. The grading loop drops
NarratorEvents and keeps going; the narrator pulls and streams to its
sink. If the narrator falls behind it just skips events — the grader
never blocks.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from auto_grader.eval_harness import EvalItem, Prediction


@dataclass
class NarratorEvent:
    """One scoring event for the narrator to comment on."""

    item: EvalItem
    prediction: Prediction
    item_index: int
    total_items: int


_NARRATOR_SYSTEM_PROMPT = """\
You are a SPORTS COMMENTATOR calling a chemistry exam grading match in \
real time. The contenders: a VLM (the grader) versus the PROFESSOR \
(ground truth). Each event tells you what the student wrote, what the \
grader awarded, and what the professor awarded. You know the ground \
truth — call it like a sportscaster.

Your job: respond with ONE short present-participle sentence (under 22 \
words) calling the play. Lively, punchy, opinionated. No preamble, no \
markdown, no lists. Always present participle ("nailing", "whiffing", \
"dodging", "matching", "overshooting"...).

Examples of the voice:
- "Nailing the density calc — grader and professor both stamping it 2 out of 2, clean play."
- "Whiffing on fr-5b — grader giving zero, but the prof rewarded the student's consistent work, ouch."
- "Going too generous on fr-12b, grader awarding full marks where the prof only gave half — overshooting the rubric."
- "Disagreeing on fr-7c, grader saying 0 but the prof checked it — possibly a ground-truth glitch."
- "Splitting the partial credit on fr-10a, grader awarding 1 of 3 where the prof gave 1.5, close but no cigar."
- "Reading orbital boxes off the page like a champ on fr-11a — full three points, both judges agreeing."

Voice rules: present participle verbs, sportscaster energy, stay under \
22 words, ONE sentence only, no preamble.
"""


def _format_event_for_narrator(event: NarratorEvent) -> str:
    item = event.item
    pred = event.prediction
    if pred.model_score == item.professor_score:
        verdict = "GRADER MATCHES PROFESSOR (exact agreement)"
    elif pred.model_score > item.professor_score:
        verdict = (
            f"GRADER OVERSHOOTS by "
            f"{pred.model_score - item.professor_score} pts (too generous)"
        )
    else:
        verdict = (
            f"GRADER UNDERSHOOTS by "
            f"{item.professor_score - pred.model_score} pts (too strict)"
        )
    return (
        f"PLAY {event.item_index} of {event.total_items}\n"
        f"Question: {item.question_id} ({item.answer_type}, "
        f"max {item.max_points} pts)\n"
        f"Student wrote: \"{item.student_answer}\"\n"
        f"Grader read: \"{pred.model_read}\"\n"
        f"Grader awarded: {pred.model_score} pts\n"
        f"Professor awarded: {item.professor_score} pts (GROUND TRUTH)\n"
        f"Professor's note: \"{item.notes}\"\n"
        f"Grader's reasoning: {pred.model_reasoning[:250]}\n"
        f"VERDICT: {verdict}\n"
        f"Call this play in ONE sentence, present participle, "
        f"under 22 words."
    )


class BonsaiNarrator:
    """Threaded narrator that streams bonsai output for each event.

    Use as a context manager:

        with BonsaiNarrator(base_url="http://localhost:8001") as narrator:
            narrator.narrate(event)
            ...

    On __exit__ the worker drains the queue and shuts down.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: str = "1234",
        model: str = "Bonsai-8B-mlx-1bit",
        sink: IO[str] | None = None,
        max_tokens: int = 60,
        max_queue: int = 4,
        spawn_terminal: bool = False,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._spawn_terminal = spawn_terminal
        self._fifo_path: Path | None = None
        self._owns_sink = False

        if spawn_terminal:
            # Open a fresh Terminal.app window tailing a fifo we own.
            self._fifo_path = self._make_fifo()
            self._spawn_terminal_window(self._fifo_path)
            # Open the fifo for writing — blocks until tail -f connects.
            self.sink = open(self._fifo_path, "w", buffering=1)
            self._owns_sink = True
        else:
            self.sink = sink if sink is not None else sys.stderr

        self._queue: queue.Queue[NarratorEvent | None] = queue.Queue(
            maxsize=max_queue
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._dropped = 0

    @staticmethod
    def _make_fifo() -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="paint-dry-"))
        fifo = tmp / "narrator.fifo"
        os.mkfifo(fifo)
        return fifo

    @staticmethod
    def _spawn_terminal_window(fifo: Path) -> None:
        """Open a Terminal.app window that tails the fifo. macOS only."""
        # AppleScript: open new Terminal window, run a banner + tail -f.
        # The banner clears the screen and prints the Project Paint Dry header.
        banner = (
            "clear; "
            "printf '\\033[1;35m'; "
            "echo '=========================================='; "
            "echo '   PROJECT PAINT DRY -- bonsai narrator'; "
            "echo '   live grading commentary'; "
            "echo '=========================================='; "
            "printf '\\033[0m'; "
            "echo; "
            f"tail -f {fifo}"
        )
        # Escape double quotes for AppleScript
        escaped = banner.replace('"', '\\"')
        script = f'tell application "Terminal" to do script "{escaped}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            # Bring Terminal forward
            subprocess.run(
                ["osascript", "-e", 'tell application "Terminal" to activate'],
                check=False,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Could not spawn Terminal.app window for narrator: {e}"
            )

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "BonsaiNarrator":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="bonsai-narrator", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        # Sentinel to drain
        try:
            self._queue.put(None, timeout=1)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self._owns_sink:
            try:
                self.sink.close()
            except Exception:
                pass
        if self._fifo_path is not None:
            try:
                self._fifo_path.unlink(missing_ok=True)
                self._fifo_path.parent.rmdir()
            except Exception:
                pass

    # -- public api --------------------------------------------------------

    def narrate(self, event: NarratorEvent) -> None:
        """Enqueue an event for narration. Drops if queue is full."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        self._write_line(f"[narrator] bonsai online ({self.model})")
        while True:
            event = self._queue.get()
            if event is None:
                break
            try:
                self._narrate_one(event)
            except Exception as e:
                self._write_line(f"[narrator] error: {e}")
        if self._dropped:
            self._write_line(
                f"[narrator] dropped {self._dropped} events under load"
            )
        self._write_line("[narrator] offline")

    def _narrate_one(self, event: NarratorEvent) -> None:
        prompt = _format_event_for_narrator(event)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _NARRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": True,
        }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        # Mark glyph based on agreement
        if event.prediction.model_score == event.item.professor_score:
            mark = "="
        elif event.prediction.model_score > event.item.professor_score:
            mark = "+"
        else:
            mark = "-"

        prefix = (
            f"[narrator {event.item_index:3d}/{event.total_items} {mark}] "
        )
        self.sink.write(prefix)
        self.sink.flush()

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        self.sink.write(delta)
                        self.sink.flush()
        finally:
            self.sink.write("\n")
            self.sink.flush()

    def _write_line(self, text: str) -> None:
        self.sink.write(text + "\n")
        self.sink.flush()
