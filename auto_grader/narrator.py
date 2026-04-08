"""Project Paint Dry — Bonsai narrator sidecar for the eval harness.

A small fast model rides alongside the grader, narrating each scoring
event in present-participle prose. Streams to stderr (or any writable
file) so it doesn't pollute the main eval output on stdout.

The narrator runs in its own thread with a queue. The grading loop drops
NarratorEvents and keeps going; the narrator pulls and streams to its
sink. If the narrator falls behind it just skips events — the grader
never blocks.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
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
You are narrating a chemistry exam grading process in real time. Each event \
describes one question that the grader just scored. Your job: respond with \
ONE short present-participle sentence (under 18 words) describing what just \
happened, in a lively but factual voice. No preamble, no markdown, no lists.

Examples of the voice:
- "Reading a balanced equation for phosphorus and chlorine, awarding full marks."
- "Catching a wrong limiting reagent on fr-5a but giving zero credit per the key."
- "Disagreeing with the professor on fr-7c — the model thinks 180 degrees, key says 120."
- "Watching the model give partial credit on a Lewis structure of ozone."
- "Splitting the difference on fr-10a, awarding 1 point of 3 for partial work."

Stay under 18 words. Present participle. Lively. Factual. One sentence only.
"""


def _format_event_for_narrator(event: NarratorEvent) -> str:
    item = event.item
    pred = event.prediction
    agreement = (
        "matches"
        if pred.model_score == item.professor_score
        else (
            "is more generous than"
            if pred.model_score > item.professor_score
            else "is stricter than"
        )
    )
    return (
        f"Item {event.item_index}/{event.total_items}: "
        f"question {item.question_id} ({item.answer_type}, "
        f"{item.max_points} pts max). "
        f"Student wrote: \"{item.student_answer}\". "
        f"Model read: \"{pred.model_read}\". "
        f"Model awarded {pred.model_score}, professor awarded "
        f"{item.professor_score}. "
        f"Model {agreement} the professor. "
        f"Model reasoning: {pred.model_reasoning[:300]}"
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
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.sink = sink if sink is not None else sys.stderr
        self.max_tokens = max_tokens
        self._queue: queue.Queue[NarratorEvent | None] = queue.Queue(
            maxsize=max_queue
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._dropped = 0

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
