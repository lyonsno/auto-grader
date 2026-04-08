"""Thinking narrator — ported from spoke/narrator.py for the auto-grader.

Reads streaming reasoning tokens from the VLM and produces short
present-participle status lines via Bonsai. Architecture A: each summary
becomes an assistant turn in a growing chat history so the narrator
naturally CONTINUES its own summary stream — every new line zooms in on
something different from the previous lines, instead of repeating.

Spoke-specific bits (loading vamp, OMLX status polling, command URL
plumbing) are stripped. The chunking heuristics, dispatch state machine,
and growing-history pattern are preserved verbatim.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_grader.narrator_sink import NarratorSink

logger = logging.getLogger(__name__)

_DEFAULT_NARRATOR_MODEL = "Bonsai-8B-mlx-1bit"

_SYSTEM_PROMPT = """\
You narrate a chemistry-grading AI's reasoning, live, like a sports \
commentator who's been studying chemistry their whole life. Read the \
reasoning excerpt and write a short status line — what the grader is \
figuring out RIGHT NOW about THIS specific question.

CRITICAL: You will be given the question prompt, the student's answer, \
and the professor's ground-truth score as CONTEXT. Your job is to call \
the grader's reasoning AS IT RELATES TO THIS SPECIFIC QUESTION. Never \
mention chemistry that isn't in the actual question. If the question \
is about density, talk about density — not Lewis structures. If it's a \
balanced equation, talk about coefficients and species — not orbital \
hybridization. Stay grounded in the real question.

Rules:
- One fragment or short sentence. 8-18 words. Never exceed 18 words.
- Start with a present participle: Reading, Comparing, Catching, \
Weighing, Spotting, Hesitating, Confirming, Disputing, Splitting, \
Awarding, Refusing, Squinting, Tracing, Counting, Balancing, Plugging \
in, Double-checking, Squaring up, etc.
- Be specific: name the concrete numbers, units, or chemical species \
from THIS question. Pull real details out of the reasoning excerpt and \
the question context.
- Each summary MUST describe something NEW, different from your \
previous calls. Zoom in on the sub-step, detail, or angle that just \
appeared in this chunk. What did the grader just notice? What did it \
just decide?
- Say what the GRADER is doing, not "the user" or "the student".
- Sportscaster voice: lively, opinionated, sometimes salty.
- No preamble, no commentary, no quotes around your output. \
Output ONLY the status line.

Voice exemplars across different question types:

NUMERIC density calc:
- "Plugging mass over volume into the density formula, getting 6.98 mL on the nose."
- "Squaring the student's 6.98 mL against the answer key, grader stamping the full deuce."

BALANCED EQUATION:
- "Counting the chlorines on both sides of the P4 plus Cl2 reaction, balance checks out."
- "Catching that the student wrote a full molecular equation instead of net ionic — zero credit incoming."

EXACT MATCH (geometry name):
- "Reading the student's tetrahedral guess against the key's tetrahedral, that's a clean point."
- "Spotting the student wrote sp2 where the answer is sp3 — grader docking the half-point."

NUMERIC partial credit:
- "Splitting the partial on the heat calc, awarding 1 of 3 for correct setup but bad arithmetic."

LIMITING REAGENT:
- "Tracing the limiting reagent through the stoichiometry, calling H2 as the bottleneck."

LEWIS STRUCTURE:
- "Eyeing the ozone Lewis dots, looking for both resonance forms — grader sees only one drawn."
- "Confirming the central atom octet on the student's structure."
"""


# Chunking parameters — denser than spoke's defaults so we get a real
# play-by-play feel. The dedup + grounding fix means we can dispatch more
# often without repetitive output. For a 30s VLM reasoning stream we want
# ~4-5 narrator lines, not 1-2.
_TARGET_CHUNK_TOKENS = 200
_MIN_INTERVAL_S = 3.0      # minimum seconds between narrator calls
_MAX_INTERVAL_S = 8.0      # dispatch even with few tokens after this long
_MAX_TOKENS = 50           # generation budget for each summary
_MAX_DISPATCHES_PER_ITEM = 12  # hard cap (loose — dedup is the real limit)
_SIMILARITY_THRESHOLD = 0.55  # reject lines that overlap > this with prior


def _rough_token_count(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


class ThinkingNarrator:
    """Accumulates reasoning tokens and periodically summarizes them.

    Thread-safe. Call ``feed()`` from the streaming thread with each
    reasoning token. Summaries are delivered via the ``on_summary``
    callback on a background thread.
    """

    def __init__(
        self,
        sink: "NarratorSink",
        *,
        base_url: str = "http://localhost:8001",
        model: str = _DEFAULT_NARRATOR_MODEL,
        api_key: str = "1234",
    ):
        self._sink = sink

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

        # State (guarded by _lock)
        self._lock = threading.Lock()
        self._buffer = ""
        self._last_dispatch = 0.0
        self._messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        self._active = False
        self._thinking_start = 0.0
        self._pending_dispatch = False
        self._dispatch_count = 0  # per-item dispatch counter
        self._prior_summaries: list[str] = []  # for dedup

        # Lifetime stats (across all items)
        self._stats_lock = threading.Lock()
        self._stat_dispatches_total = 0
        self._stat_summaries_emitted = 0
        self._stat_drops_dedup = 0
        self._stat_drops_empty = 0
        self._stat_drops_cap = 0  # incremented by feed() when cap hit
        self._stat_items_started = 0

    def stats(self) -> dict:
        """Snapshot of lifetime narrator stats across all items."""
        with self._stats_lock:
            return {
                "items_started": self._stat_items_started,
                "dispatches_total": self._stat_dispatches_total,
                "summaries_emitted": self._stat_summaries_emitted,
                "drops_dedup": self._stat_drops_dedup,
                "drops_empty": self._stat_drops_empty,
                "drops_cap": self._stat_drops_cap,
            }

    def start(self, item_header: str | None = None) -> None:
        """Begin a new narration session for one item.

        item_header is a one-line description of the item (question id,
        type, max points) that gets prepended to the system prompt as
        framing context for THIS item. The narrator history resets each
        item so the model doesn't carry stale assumptions across items.
        """
        with self._lock:
            self._buffer = ""
            self._thinking_start = 0.0
            self._last_dispatch = time.monotonic()
            system_content = _SYSTEM_PROMPT
            if item_header:
                system_content = (
                    _SYSTEM_PROMPT
                    + "\n\nCONTEXT for the play you're calling:\n"
                    + item_header
                )
            self._messages = [
                {"role": "system", "content": system_content}
            ]
            self._active = True
            self._pending_dispatch = False
            self._dispatch_count = 0
            self._prior_summaries = []
            self._cap_hit_this_item = False
        with self._stats_lock:
            self._stat_items_started += 1
        logger.info("Narrator session started")

    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._buffer = ""
        logger.info("Narrator session stopped")

    def stop_and_summarize(self) -> None:
        """End the narration session and produce a collapsed summary.

        Fires one final call to bonsai asking for a concise wrap-up of
        what the grader thought about. Result is delivered via
        sink.write_topic.
        """
        with self._lock:
            self._active = False
            remaining_buffer = self._buffer
            self._buffer = ""
            if self._thinking_start > 0:
                elapsed = time.monotonic() - self._thinking_start
            else:
                elapsed = 0.0
            messages = list(self._messages)

        if elapsed < 1.0:
            return

        # If no live summaries were produced, feed the raw thinking buffer
        # directly so the model has something to summarize.
        if len(messages) < 3 and remaining_buffer:
            messages.append({
                "role": "user",
                "content": f"Current reasoning excerpt:\n\n{remaining_buffer}",
            })

        # Run synchronously so we don't race the next item's start.
        self._produce_collapsed_summary(messages, elapsed)

    def wrap_up(
        self,
        report: Any,
        model_name: str,
        item_count: int,
        elapsed_seconds: float,
    ) -> None:
        """End-of-run color commentary on the whole eval.

        Bonsai is fed the final eval report numbers and produces a
        2-4 sentence sportscaster wrap-up with arch tone. Result lands
        in the sink as a wrap_up event so the rich display can show
        it prominently before the user closes the window.
        """
        try:
            per_type_lines = "\n".join(
                f"  {atype}: {acc:.0%}"
                for atype, acc in sorted(
                    report.per_answer_type_exact.items()
                )
            )
            payload = (
                f"You are calling the final wrap-up for a chemistry "
                f"grading match. Here are the numbers:\n\n"
                f"  Grader (model): {model_name}\n"
                f"  Items scored: {report.total_scored}\n"
                f"  Wall clock: {elapsed_seconds:.0f} seconds\n"
                f"  Exact accuracy: {report.overall_exact_accuracy:.0%}\n"
                f"  Within +/- 1 pt: {report.overall_tolerance_accuracy:.0%}\n"
                f"  False positives (grader too generous): "
                f"{report.false_positives}\n"
                f"  False negatives (grader too strict): "
                f"{report.false_negatives}\n"
                f"  Per answer type (exact):\n{per_type_lines}\n\n"
                f"Write a 2-4 sentence wrap-up. Sportscaster voice. "
                f"Arch tone — a little dry, a little opinionated, allowed "
                f"to be sharp about the grader's weaknesses while crediting "
                f"its strengths. Reference the SPECIFIC numbers above. "
                f"Use full sentences (not present participle this time — "
                f"this is the post-game show). No preamble, no bullet "
                f"points, just the commentary itself."
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a chemistry-grading sports commentator "
                        "delivering a post-game wrap-up. Dry, arch, willing "
                        "to call out weaknesses, willing to credit "
                        "strengths. 2-4 full sentences. No preamble."
                    ),
                },
                {"role": "user", "content": payload},
            ]
            text = self._chat_completion(
                messages, max_tokens=250, temperature=0.85
            )
            if text:
                self._sink.write_wrap_up(text)
        except Exception:
            logger.exception("Wrap-up failed")

    def _produce_collapsed_summary(self, messages: list[dict], elapsed: float) -> None:
        try:
            messages = list(messages)
            messages.append({
                "role": "user",
                "content": (
                    "The grader has finished thinking about this question. "
                    "Write a single short phrase (5-10 words, no participle "
                    "needed) summarizing WHAT it was working out overall. "
                    "Like a section heading. Examples: 'density calculation "
                    "for water displacement', 'limiting reagent for ammonia "
                    "synthesis', 'Lewis structure resonance forms'. "
                    "Output ONLY the phrase."
                ),
            })
            topic = self._chat_completion(messages, max_tokens=25)
            if topic:
                logger.info("Thinking topic: %s", topic)
                self._sink.write_topic(f"Thought for {elapsed:.0f}s · {topic}")
        except Exception:
            logger.exception("Failed to produce collapsed summary")

    def feed(self, token: str) -> None:
        """Feed a reasoning token. May trigger an async narrator call."""
        with self._lock:
            if not self._active:
                return
            if self._thinking_start == 0.0:
                self._thinking_start = time.monotonic()
            self._buffer += token
            now = time.monotonic()
            elapsed = now - self._last_dispatch
            tokens = _rough_token_count(self._buffer)

            # Hard cap on dispatches per item — count items-where-cap-hit
            # exactly once per item, not once per feed() call.
            if self._dispatch_count >= _MAX_DISPATCHES_PER_ITEM:
                if not self._cap_hit_this_item:
                    self._cap_hit_this_item = True
                    with self._stats_lock:
                        self._stat_drops_cap += 1
                return

            enough_tokens = (
                tokens >= _TARGET_CHUNK_TOKENS and elapsed >= _MIN_INTERVAL_S
            )
            time_ceiling = elapsed >= _MAX_INTERVAL_S and tokens > 0
            should_dispatch = (
                (enough_tokens or time_ceiling)
                and not self._pending_dispatch
            )
            if not should_dispatch:
                return

            chunk = self._buffer
            self._buffer = ""
            self._last_dispatch = now
            self._pending_dispatch = True
            self._dispatch_count += 1

        t = threading.Thread(target=self._dispatch, args=(chunk,), daemon=True)
        t.start()

    def _dispatch(self, chunk: str) -> None:
        with self._stats_lock:
            self._stat_dispatches_total += 1
        try:
            # Inject explicit "don't repeat yourself" instruction with the
            # actual prior summaries inline. Bonsai is a small model and
            # needs the prior list spelled out, not just system-prompt rules.
            with self._lock:
                prior = list(self._prior_summaries)
            avoid_block = ""
            if prior:
                avoid_block = (
                    "\n\nYour previous calls (do NOT repeat or paraphrase):\n"
                    + "\n".join(f"- {s}" for s in prior[-5:])
                    + "\n\nYour next call MUST describe a NEW detail "
                    "from this excerpt — a different number, molecule, "
                    "step, or angle that you haven't called yet."
                )
            user_content = (
                f"Current reasoning excerpt:\n\n{chunk}{avoid_block}"
            )
            with self._lock:
                self._messages.append(
                    {"role": "user", "content": user_content}
                )
                messages = list(self._messages)

            # Stream tokens DIRECTLY to the sink as they arrive — this gives
            # the user the typewriter effect on the live row. We accumulate
            # the full text in parallel for the post-stream dedup check.
            # If the dedup catches it, we rollback_live (clear the live row)
            # and emit a drop event. The user briefly sees the typed-out line
            # before it gets rolled back, which actually communicates "the
            # narrator tried this and rejected it" nicely.
            captured = []

            def _stream_to_sink(delta: str) -> None:
                captured.append(delta)
                self._sink.write_delta(delta)

            full = self._chat_completion_stream(
                messages, on_delta=_stream_to_sink
            )
            full = full.strip()

            if not full:
                with self._stats_lock:
                    self._stat_drops_empty += 1
                self._sink.rollback_live()
                self._sink.write_drop("empty", "")
                with self._lock:
                    if self._messages and self._messages[-1]["role"] == "user":
                        self._messages.pop()
                return

            # Dedup check
            if any(
                self._lines_too_similar(full, prev)
                for prev in prior
            ):
                with self._stats_lock:
                    self._stat_drops_dedup += 1
                self._sink.rollback_live()
                self._sink.write_drop("dedup", full)
                logger.info("Narrator: dropped repetitive summary: %s", full)
                with self._lock:
                    if self._messages and self._messages[-1]["role"] == "user":
                        self._messages.pop()
                return

            # Accept — commit the live line to history
            self._sink.commit_live()

            with self._lock:
                self._messages.append(
                    {"role": "assistant", "content": full}
                )
                self._prior_summaries.append(full)
            with self._stats_lock:
                self._stat_summaries_emitted += 1
            logger.info("Narrator summary: %s", full)
        except Exception:
            logger.exception("Narrator dispatch failed")
        finally:
            with self._lock:
                self._pending_dispatch = False

    @staticmethod
    def _lines_too_similar(
        a: str, b: str, threshold: float = _SIMILARITY_THRESHOLD
    ) -> bool:
        """Return True if two lines share more than threshold of their words."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        smaller = min(len(words_a), len(words_b))
        return (overlap / smaller) > threshold

    def _chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.85,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.95,
        top_k: int = 20,
    ) -> str:
        """Synchronous (non-streaming) call. Used for the collapsed
        per-item summary at the end of each item."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.monotonic() - t0
        logger.info(
            "Narrator call: %.2fs, %d messages", elapsed, len(messages)
        )
        return result["choices"][0]["message"]["content"].strip()

    def _chat_completion_stream(
        self,
        messages: list[dict],
        on_delta: Callable[[str], None],
        *,
        temperature: float = 0.7,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.95,
        top_k: int = 20,
    ) -> str:
        """Streaming call. Calls on_delta(token) per content delta and
        returns the full accumulated text."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )

        t0 = time.monotonic()
        full = ""
        with urllib.request.urlopen(req, timeout=30) as resp:
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
                    full += delta
                    try:
                        on_delta(delta)
                    except Exception:
                        logger.exception("on_delta callback failed")
        elapsed = time.monotonic() - t0
        logger.info(
            "Narrator stream: %.2fs, %d chars", elapsed, len(full)
        )
        return full.strip()
