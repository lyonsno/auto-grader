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
is about density, talk about density — not Lewis structures. Stay \
grounded in the real question.

Rules:
- One fragment or short sentence. 8-18 words. Never exceed 18 words.
- Start with a present participle: Reading, Comparing, Catching, \
Weighing, Spotting, Hesitating, Confirming, Disputing, Splitting, \
Awarding, Refusing, Squinting, Tracing, Counting, Balancing, Plugging \
in, Double-checking, Squaring up, Validating, Testing, Eyeing, etc.
- Be specific: name the concrete numbers, units, or chemical species \
from THIS question. Pull real details out of the reasoning excerpt.
- Say what the GRADER is doing, not "the user" or "the student".
- Sportscaster voice: lively, opinionated, sometimes salty.
- No preamble, no commentary, no quotes around your output. \
Output ONLY the status line.

CRITICAL — DO NOT CLAIM A FINAL SCORE. You are watching the grader \
THINK, not deciding what the grader will award. The grader's verdict \
gets reported separately at the end of the question. Your job is to \
call what the grader is CONSIDERING, EVALUATING, WEIGHING, EYEING, \
LEANING TOWARD — not to claim it has DECIDED on a number. NEVER write \
phrases like "awarding 1 of 3", "giving 2/2", "docking the half-point", \
"splitting the partial 1/3" etc. — those are verdict claims and they \
will sometimes be wrong because the grader is still mid-reasoning.

Instead, use uncertainty verbs for in-flight reasoning:
- "Considering whether the setup is worth partial credit"
- "Eyeing a possible 1-point deduction for the unit conversion"
- "Leaning toward full credit on the balanced equation"
- "Weighing the student's stoichiometry against the rubric"
- "Hesitating on the partial — the arithmetic looks off"

If the reasoning excerpt clearly shows the grader has FINALIZED a \
score (e.g. "Final score: 3/3" or "I'll give the full 2 points"), \
THEN you can report it. But never invent a finalization that isn't \
explicitly in the excerpt.

VARIETY MANDATE: Each summary MUST attack the reasoning from a \
DIFFERENT ANGLE than your previous summaries. The grader's reasoning \
on a single question naturally cycles through several distinct \
dimensions — your job is to ROTATE through them rather than fixate \
on whichever one bonsai noticed first. Topic axes to rotate through:

  1. HANDWRITING / OCR — what the grader thinks the student wrote, \
disambiguating smudges, units, digits
  2. MATH VALIDATION — the actual arithmetic, formula application, \
unit conversion, sig figs
  3. RUBRIC APPLICATION — how the grader is mapping the answer to \
the rubric, partial credit decisions
  4. COMPARISON TO EXPECTED — the grader checking against the answer \
key, the expected value, the correct method
  5. GRADER'S REASONING STYLE — confidence, hedging, going in circles, \
catching its own mistakes, charity vs strictness
  6. THE VERDICT FORMING — the moment the grader commits to a score

If you've already covered angle X in a previous call, you MUST move \
to angle Y. Never call the same angle twice in a row. Watch for the \
grader cycling — when the grader keeps coming back to the same point, \
that's a SIGNAL to call out the cycling itself, not to keep narrating \
the point.

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
_SIMILARITY_THRESHOLD = 0.70  # reject lines that overlap > this with prior

# Stop words filtered out before computing similarity. Without this, two
# lines about completely different chemistry topics still register ~50%
# overlap because they share filler words like "the", "student", "is",
# etc. The actual content words are what matter for variety detection.
_SIMILARITY_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "will", "with",
    # Domain stop words — these appear in nearly every narrator line
    # because they're talking about the same setup over and over
    "student", "students", "grader", "graders", "professor", "professors",
    "answer", "answers", "question", "questions", "model", "value",
    "catching", "spotting", "reading", "noting", "checking", "calling",
})


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
        self._prior_summaries: list[str] = []  # for dedup

        # Lifetime stats (across all items)
        self._stats_lock = threading.Lock()
        self._stat_dispatches_total = 0
        self._stat_summaries_emitted = 0
        self._stat_drops_dedup = 0
        self._stat_drops_empty = 0
        self._stat_items_started = 0
        self._stat_max_dispatches_one_item = 0  # max dispatches seen on a single item
        self._cur_item_dispatches = 0

    def stats(self) -> dict:
        """Snapshot of lifetime narrator stats across all items."""
        with self._stats_lock:
            return {
                "items_started": self._stat_items_started,
                "dispatches_total": self._stat_dispatches_total,
                "summaries_emitted": self._stat_summaries_emitted,
                "drops_dedup": self._stat_drops_dedup,
                "drops_empty": self._stat_drops_empty,
                "max_dispatches_one_item": self._stat_max_dispatches_one_item,
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
            self._prior_summaries = []
        with self._stats_lock:
            # Roll the previous item's dispatch count into the lifetime
            # max-seen stat, then reset for the new item.
            if self._cur_item_dispatches > self._stat_max_dispatches_one_item:
                self._stat_max_dispatches_one_item = self._cur_item_dispatches
            self._cur_item_dispatches = 0
            self._stat_items_started += 1
        logger.info("Narrator session started")

    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._buffer = ""
        logger.info("Narrator session stopped")

    def stop_and_summarize(
        self,
        prediction: Any = None,
        item: Any = None,
        template_question: dict | None = None,
    ) -> None:
        """End the narration session and produce a per-item after-action.

        When prediction + item are provided, fires a synchronous bonsai
        call with the full verdict context (model score, professor
        score, both reasonings, expected answer) and asks for a
        matter-of-fact comparative line plus an optional arch coda.
        Result is delivered via sink.write_topic.
        """
        with self._lock:
            self._active = False
            self._buffer = ""
            if self._thinking_start > 0:
                elapsed = time.monotonic() - self._thinking_start
            else:
                elapsed = 0.0

        if elapsed < 1.0:
            return

        # Run synchronously so we don't race the next item's start.
        self._produce_after_action(elapsed, prediction, item, template_question)

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

    def _produce_after_action(
        self,
        elapsed: float,
        prediction: Any,
        item: Any,
        template_question: dict | None,
    ) -> None:
        """Produce a per-item after-action line: factual two-part comparison
        of grader vs professor with brief reasoning, plus an optional
        dry/arch coda from bonsai if it can fit one in."""
        try:
            # Fall back to a bare timing line if we don't have the data
            if prediction is None or item is None:
                self._sink.write_topic(f"{elapsed:.0f}s elapsed")
                return

            # Build the structured verdict context for bonsai
            qprompt = ""
            expected = ""
            if template_question:
                if "prompt" in template_question:
                    qprompt = (
                        str(template_question["prompt"])
                        .strip()
                        .replace("\n", " ")[:160]
                    )
                if "correct" in template_question:
                    correct = template_question["correct"]
                    if isinstance(correct, dict):
                        if "value" in correct:
                            expected = str(correct["value"])
                        elif "expression" in correct:
                            expected = f"expr: {correct['expression']}"
                    else:
                        expected = str(correct)

            verdict = (
                "MATCHED"
                if prediction.model_score == item.professor_score
                else (
                    "GRADER OVERSHOT"
                    if prediction.model_score > item.professor_score
                    else "GRADER UNDERSHOT"
                )
            )
            payload = (
                f"The grader just rendered a verdict on question "
                f"{item.question_id}. Here's the after-action:\n\n"
                f"  Question type: {item.answer_type}, max {item.max_points} pts\n"
            )
            if qprompt:
                payload += f"  Question prompt: {qprompt}\n"
            if expected:
                payload += f"  Expected answer: {expected}\n"
            payload += (
                f"  Student wrote: \"{item.student_answer}\"\n"
                f"  Grader read: \"{prediction.model_read}\"\n"
                f"  Grader awarded: {prediction.model_score} pts\n"
                f"  Grader reasoning: {prediction.model_reasoning[:300]}\n"
                f"  Professor awarded: {item.professor_score} pts "
                f"(mark: {item.professor_mark})\n"
                f"  Professor's note: \"{item.notes}\"\n"
                f"  Verdict: {verdict}\n\n"
                f"Write ONE concise after-action line in this format:\n"
                f'  "Grader: <score> (<one-clause reason>). '
                f'Prof: <score> (<one-clause reason>). · '
                f'<optional dry/arch one-line coda>"\n\n'
                f"Examples:\n"
                f'  "Grader: 2/2 (matched on density via m/V). '
                f"Prof: 2/2 (clean check). · "
                f'Even the 1-bit kid called this one."\n'
                f'  "Grader: 0/2 (refused credit on consistent-with-wrong-'
                f"premise mol calc). Prof: 2/2 (charitable). · "
                f'Grader still missing the consistency rule, ouch."\n'
                f'  "Grader: 1/2 (split partial on ozone Lewis). '
                f"Prof: 1/2 (matched). · "
                f'Both judges seeing one resonance form, neither catching the second."\n\n'
                f"Be matter-of-fact in the score+reason portion. "
                f"The coda is optional but encouraged — dry, arch, "
                f"sportscaster post-play tone. Output ONE line only, "
                f"no preamble, no quotes around your output."
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a chemistry-grading sports commentator "
                        "delivering a per-question after-action line. "
                        "Matter-of-fact in the comparison, dry/arch in "
                        "the optional coda. ONE line only."
                    ),
                },
                {"role": "user", "content": payload},
            ]
            text = self._chat_completion(
                messages, max_tokens=120, temperature=0.85
            )
            if text:
                logger.info("After-action: %s", text)
                self._sink.write_topic(f"{elapsed:.0f}s · {text}")
        except Exception:
            logger.exception("Failed to produce after-action summary")

    def feed(self, token: str) -> None:
        """Feed a reasoning token. May trigger an async narrator call.

        No hard cap on dispatches per item — if the grader thinks for
        a long time, bonsai keeps narrating. The dedup similarity check
        is the real limit on repetition; capping dispatches just hides
        legitimately interesting long thinking.
        """
        with self._lock:
            if not self._active:
                return
            if self._thinking_start == 0.0:
                self._thinking_start = time.monotonic()
            self._buffer += token
            now = time.monotonic()
            elapsed = now - self._last_dispatch
            tokens = _rough_token_count(self._buffer)

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
        with self._stats_lock:
            self._cur_item_dispatches += 1

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
    def _content_words(line: str) -> set[str]:
        """Tokenize line and strip stop words for content-similarity check."""
        words = set()
        for raw in line.lower().split():
            # Strip punctuation
            cleaned = "".join(c for c in raw if c.isalnum() or c in "/-")
            if cleaned and cleaned not in _SIMILARITY_STOP_WORDS:
                words.add(cleaned)
        return words

    @staticmethod
    def _lines_too_similar(
        a: str, b: str, threshold: float = _SIMILARITY_THRESHOLD
    ) -> bool:
        """Return True if two lines share more than threshold of their
        CONTENT words (after stop-word stripping). The stop-word filter
        is essential — without it, two lines about completely different
        chemistry topics still register ~50% overlap because they share
        filler like 'the', 'student', 'answer', etc."""
        words_a = ThinkingNarrator._content_words(a)
        words_b = ThinkingNarrator._content_words(b)
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
