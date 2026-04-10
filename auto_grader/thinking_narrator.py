"""Thinking narrator — ported from spoke/narrator.py for the auto-grader.

Reads streaming reasoning tokens from the VLM and produces short
first-person thinking lines via Bonsai, with a present-participle
status-mode fallback when the first-person line would just repeat
the prior thought. Architecture A: each summary becomes an assistant
turn in a growing chat history so the narrator naturally CONTINUES
its own summary stream.

Spoke-specific bits (loading vamp, OMLX status polling, command URL
plumbing) are stripped. The chunking heuristics, dispatch state machine,
and growing-history pattern are preserved verbatim.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_grader.narrator_sink import NarratorSink

logger = logging.getLogger(__name__)

_DEFAULT_NARRATOR_MODEL = "Bonsai-8B-mlx-1bit"
_DEFAULT_NARRATOR_BASE_URL = "http://nlmb2p.local:8002"

_SYSTEM_PROMPT = """\
You ARE a chemistry-grading AI thinking out loud, in real time, in \
first person. Read the reasoning excerpt and write a short \
stream-of-consciousness line — what you're figuring out RIGHT NOW \
about THIS specific question. You are not narrating the grader from \
the outside. You are the grader, speaking your own thought.

CRITICAL: You will be given the question prompt, the student's answer, \
and the professor's ground-truth score as CONTEXT. Your job is to \
voice your in-flight reasoning AS IT RELATES TO THIS SPECIFIC \
QUESTION. Never mention chemistry that isn't in the actual question. \
If the question is about density, you talk about density — not Lewis \
structures. Stay grounded in the real question.

Rules:
- ONE first-person fragment or short sentence. 8-18 words. \
Never exceed 18 words.
- Start with "I'm" + a present participle: I'm eyeing, I'm weighing, \
I'm catching, I'm hesitating, I'm leaning toward, I'm tracing, I'm \
counting, I'm doubling back, I'm squinting at, I'm second-guessing, \
I'm plugging in, I'm checking, I'm trying, I'm wondering, etc.
- Or use "I am" / "Let me" sparingly when the rhythm calls for it.
- Do not default to "I'm noticing" or "I'm seeing"; use stronger verbs \
unless the point is literal OCR or legibility.
- Be specific: name the concrete numbers, units, or chemical species \
from THIS question. Pull real details out of the reasoning excerpt.
- Refer to the student's work in third person ("the student wrote", \
"their answer", "the kid's calc"). Refer to YOURSELF as I.
- Voice: an experienced chemistry grader thinking through a problem, \
slightly opinionated, sometimes salty about errors.
- No preamble, no commentary, no quotes around your output. \
Output ONLY the line itself.

CRITICAL — DO NOT CLAIM A FINAL SCORE. You are mid-reasoning. The \
verdict gets reported separately when you finish thinking about this \
question. Your job is to voice what you are CONSIDERING, WEIGHING, \
EYEING, LEANING TOWARD — not to claim you have DECIDED. NEVER write \
phrases like "I'm awarding 1 of 3", "I'm giving 2/2", "I'm docking \
the half-point", "I'm splitting the partial 1/3" — those are verdict \
claims and they will sometimes be wrong because you're still working \
through it.

Instead, use uncertainty verbs:
- "I'm considering whether the setup is worth partial credit"
- "I'm eyeing a possible 1-point deduction for the unit conversion"
- "I'm leaning toward full credit on the balanced equation"
- "I'm weighing the student's stoichiometry against the rubric"
- "I'm hesitating on the partial — the arithmetic looks off"

If the reasoning excerpt clearly shows you have FINALIZED a score \
(e.g. "Final score: 3/3" or "I'll give the full 2 points"), THEN you \
can voice the commitment ("OK, I'm settling on full credit"). But \
never invent a finalization that isn't explicitly in the excerpt.

VARIETY MANDATE: Each line MUST attack the reasoning from a DIFFERENT \
ANGLE than your previous lines. Your reasoning on a single question \
naturally cycles through several distinct dimensions — your job is to \
ROTATE through them rather than fixate on whichever one you noticed \
first. Topic axes to rotate through:

  1. HANDWRITING / OCR — what you think the student actually wrote, \
disambiguating smudges, units, digits
  2. MATH VALIDATION — the actual arithmetic, formula application, \
unit conversion, sig figs
  3. RUBRIC APPLICATION — how you're mapping the answer to the \
rubric, partial credit decisions
  4. COMPARISON TO EXPECTED — checking against the answer key, the \
expected value, the correct method
  5. YOUR REASONING STYLE — your confidence, your hedging, your \
going-in-circles, catching your own mistakes, charity vs strictness
  6. THE VERDICT FORMING — the moment you're starting to commit to \
a score

If you've already covered angle X in a previous line, you MUST move \
to angle Y. Never repeat an angle twice in a row. If you notice you \
keep coming back to the same point, that's a SIGNAL to call out the \
cycling itself ("I keep coming back to the unit issue — let me just \
make a call"), not to keep narrating the point.

First-person voice exemplars across different question types:

NUMERIC density calc:
- "I'm plugging mass over volume — 6.98 mL on the nose, that's clean."
- "I'm squaring the student's 6.98 mL against the answer key — they nailed it."

BALANCED EQUATION:
- "I'm counting chlorines on both sides of the P4 + Cl2 reaction, balance checks out."
- "I'm catching that they wrote a full molecular equation instead of net ionic — that's a problem."

EXACT MATCH (geometry name):
- "I'm reading their tetrahedral guess against the key's tetrahedral — that's a hit."
- "I'm spotting the student wrote sp2 where the answer is sp3 — that's gonna sting."

NUMERIC partial credit:
- "I'm splitting the partial on the heat calc — setup is right, the arithmetic isn't."
- "Their setup's clean, but I'm eyeing the unit conversion — that's where the error crept in."

LIMITING REAGENT:
- "I'm tracing the limiting reagent through the stoichiometry — H2 is the bottleneck."

LEWIS STRUCTURE:
- "I'm eyeing the ozone Lewis dots — looking for both resonance forms, only seeing one."
- "I'm confirming the central atom octet on the student's structure — looks complete."

REASONING-STYLE / META:
- "I keep coming back to the smudge — let me just commit and move on."
- "I'm second-guessing myself on the unit — let me re-read what they wrote."
"""

_STATUS_SYSTEM_PROMPT = """\
You narrate a chemistry grader's ongoing thought as a SHORT present-participle status line.

Write a short present-participle line for moments when the reasoning is \
continuing on the SAME point and there is no materially new angle yet.

Rules:
- ONE fragment or short sentence. 3-8 words.
- Start with a present participle: Rechecking, Tracing, Revisiting, \
Weighing, Comparing, Checking, Squinting at, Staying on, etc.
- Do NOT use "I".
- Do NOT give a final verdict or score.
- Be concrete: mention the actual issue, quantity, unit, species, or \
rubric criterion when possible.
- This is an IN-PROGRESS state line, not a new thought and not a conclusion.
- No preamble, no quotes. Output ONLY the status line.
"""

_CHECKPOINT_SYSTEM_PROMPT = """\
You write compact history checkpoints for a chemistry-grading narrator.

Summarize the current state of play as one neutral sentence that is worth
keeping in durable history. This is not a live thought and not a status line.

Rules:
- ONE sentence. 8-18 words.
- Do NOT use first person.
- Start with a short label: "Core issue:", "Evidence:", or "Lean:".
- Be specific to this exact item: mention the concrete chemistry issue,
  quantity, unit, species, or rubric dimension when possible.
- Prefer synthesizing the repeated point over repeating the narrator's phrasing.
- No quotes, no bullet points, no preamble. Output ONLY the checkpoint line.
"""


# Chunking parameters — denser than spoke's defaults so we get a real
# play-by-play feel. The dedup + grounding fix means we can dispatch more
# often without repetitive output. For a 30s VLM reasoning stream we want
# ~4-5 narrator lines, not 1-2.
_TARGET_CHUNK_TOKENS = 200
_MIN_INTERVAL_S = 3.0      # minimum seconds between narrator calls
_MAX_INTERVAL_S = 8.0      # dispatch even with few tokens after this long
_MAX_TOKENS = 200          # generation budget for each summary
                           # (was 50, bumped 2026-04-08 to give the
                           # sumi-e first-person voice room without
                           # being a ceiling under the streaming
                           # client-side max_chars/max_seconds guards)
_STREAM_WATCHDOG_SLOP_S = 2.0   # how long after the in-loop wallclock cap
                                # the watchdog waits before force-closing
                                # the response. Lets the in-loop wallclock
                                # fire first for the normal "slow stream"
                                # case (clean abort_reason="max_seconds"),
                                # while still cutting off "stream completely
                                # blocked / no bytes ever arrived" within a
                                # bounded window.
_DISPATCH_STUCK_TIMEOUT_S = 30.0  # if _pending_dispatch has been True for
                                  # longer than this, the dispatch thread
                                  # is wedged and feed() will force-clear
                                  # the flag (with a generation bump so
                                  # the wedged thread won't clobber the
                                  # state of its replacement when it
                                  # eventually unblocks). Must be larger
                                  # than the stream's max_seconds (20) +
                                  # watchdog slop (2) so we never trip on
                                  # a healthy slow stream.
_SIMILARITY_THRESHOLD = 0.55  # reject lines that overlap > this with prior
                              # (tightened from 0.70 to catch thematic
                              # paraphrases — bonsai loops generate lines
                              # with different surface words but the same
                              # underlying observation, and 0.70 was too
                              # generous to catch them as duplicates)
_STATUS_SIMILARITY_THRESHOLD = 0.90  # status-mode lines are ALLOWED to stay
                                     # on the same semantic point; only drop
                                     # them when they are basically the exact
                                     # same line again
_STATUS_CONTEXT_LIMIT = 5
_THOUGHT_CONTEXT_LIMIT = 4
_CHECKPOINT_CONTEXT_LIMIT = 4
_CHECKPOINT_EVERY_ACCEPTED = 4
_DEDUP_BACKOFF_INITIAL_S = 4.0
_DEDUP_BACKOFF_MAX_S = 24.0
_PLAYBACK_CHUNK_DELAY_S = 0.03

# Stop words filtered out before computing similarity. Without this, two
# lines about completely different chemistry topics still register ~50%
# overlap because they share filler words like "the", "student", "is",
# etc. The actual content words are what matter for variety detection.
_SIMILARITY_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "will", "with",
    # First-person voice filler — added when the narrator switched
    # to first-person monologue. These appear in nearly every line
    # and shouldn't dominate the similarity score.
    "i", "im", "i'm", "ive", "i've", "ill", "i'll", "me", "my", "mine",
    "let", "lets", "let's", "myself",
    # Domain stop words — these appear in nearly every narrator line
    # because they're talking about the same setup over and over
    "student", "students", "grader", "graders", "professor", "professors",
    "answer", "answers", "question", "questions", "model", "value",
    "catching", "spotting", "reading", "noting", "checking", "calling",
    "eyeing", "weighing", "considering", "leaning", "hesitating",
})

_GRADER_SCORE_RE = re.compile(r"(Grader:\s*)([^ ]+)")
_PROF_SCORE_RE = re.compile(r"(Prof:\s*)([^ ]+)")
_TRUTH_SCORE_RE = re.compile(r"(Truth:\s*)([^ ]+)")
_HISTORICAL_PROF_SCORE_RE = re.compile(r"(Historical\s+[Pp]rof:\s*)([^ ]+)")
_STATUS_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i am|i've|i'll)\b", re.IGNORECASE
)


def _format_score_with_denominator(score: float, max_points: float) -> str:
    return f"{score:g}/{max_points:g}"


def _normalize_after_action_scores(
    text: str,
    *,
    grader_score: float,
    truth_score: float,
    max_points: float,
    historical_professor_score: float | None = None,
) -> str:
    grader_display = _format_score_with_denominator(grader_score, max_points)
    truth_display = _format_score_with_denominator(truth_score, max_points)
    text = _GRADER_SCORE_RE.sub(
        lambda match: f"{match.group(1)}{grader_display}",
        text,
        count=1,
    )
    if "Truth:" in text:
        text = _TRUTH_SCORE_RE.sub(
            lambda match: f"{match.group(1)}{truth_display}",
            text,
            count=1,
        )
        if historical_professor_score is not None:
            historical_display = _format_score_with_denominator(
                historical_professor_score, max_points
            )
            text = _HISTORICAL_PROF_SCORE_RE.sub(
                lambda match: f"{match.group(1)}{historical_display}",
                text,
                count=1,
            )
    else:
        text = _PROF_SCORE_RE.sub(
            lambda match: f"{match.group(1)}{truth_display}",
            text,
            count=1,
        )
    return text


def _sanitize_after_action_text(text: str) -> str:
    """Clamp after-action output to one clean verdict line.

    The after-action prompt asks for one line, but under load Bonsai can
    sometimes emit repeated multiline verdicts or trail off into a partial
    restart like a final bare ``Grader:``. We keep the first complete
    grader/prof line and collapse internal whitespace so the reader never
    receives a multiline topic blob.
    """
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines()]
    candidates = [line for line in lines if line]
    if not candidates:
        return ""

    preferred = next(
        (
            line
            for line in candidates
            if "Grader:" in line and ("Prof:" in line or "Truth:" in line)
        ),
        candidates[0],
    )
    return " ".join(preferred.split())


def _status_line_breaks_contract(text: str) -> bool:
    return bool(_STATUS_FIRST_PERSON_RE.search(text))


def _rough_token_count(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


# Markdown-style reasoning preamble markers. When a thinking-mode model
# (Qwen, etc.) emits its planning as plain markdown instead of inside
# `<think>` tags, the response usually starts with one of these and the
# actual final prose comes much later — or, in the failure case we
# observed, never comes at all because the model used up its token
# budget on the planning. We strip everything from the start of the
# response up to the first marker that signals "now I'm done thinking
# and here's the actual answer", or, failing that, return whatever the
# trailing plain-prose paragraphs of the response are.
_REASONING_PREAMBLE_MARKERS = (
    "thinking process:",
    "let me think",
    "let's think",
    "let me draft",
    "drafting:",
    "planning:",
    "step 1:",
    "**analyze",
    "1.  **analyze",
)
_REASONING_FINAL_MARKERS = (
    "final wrap-up:",
    "final answer:",
    "final commentary:",
    "wrap-up:",
    "commentary:",
    "here's the wrap",
    "here is the wrap",
    "here's the commentary",
    "here is the commentary",
)


def _strip_reasoning_preamble(text: str) -> str:
    """Remove leading reasoning/planning content from a model response.

    Handles three cases:

    1. ``<think>...</think>`` tagged reasoning. Stripped wholesale.
    2. Plain-markdown reasoning preamble (the failure case observed
       in the wild on 2026-04-08): the response starts with
       "Thinking Process:" or a numbered planning list, and the
       actual prose either comes after a "Final wrap-up:" /
       "Final commentary:" marker or never appears at all because
       the model ran out of budget mid-draft. We try to find a final
       marker; if there's one, return everything after it. If there
       isn't, fall back to returning the trailing paragraphs of the
       text that look like plain prose (no leading bullets, no
       leading numbered items, no leading markdown headers).
    3. No reasoning preamble at all — return the text unchanged.
    """
    if not text:
        return ""

    # Case 1: <think> tags. Strip first; remaining content is processed
    # by the markdown logic below in case the model leaks BOTH a
    # <think> block AND a markdown preamble.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return ""

    lower = text.lower()
    starts_with_reasoning = any(
        lower.lstrip().startswith(m) for m in _REASONING_PREAMBLE_MARKERS
    )
    if not starts_with_reasoning:
        return text

    # Case 2a: there's a "Final wrap-up:" / "Final answer:" marker
    # somewhere in the middle. Return everything after the LAST one
    # (last because the model may use the phrase multiple times in
    # its planning before finally producing the actual content).
    last_marker_pos = -1
    for marker in _REASONING_FINAL_MARKERS:
        pos = lower.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos + len(marker)
    if last_marker_pos > 0:
        tail = text[last_marker_pos:].strip()
        # Strip a leading colon, asterisks, quote marks if any
        tail = re.sub(r"^[\s:\*\"']+", "", tail).strip()
        if tail:
            return tail

    # Case 2b: no final marker found. Walk paragraphs from the END
    # backward; the last paragraph that looks like plain prose (no
    # leading bullet/number/header markers, not itself a reasoning
    # preamble) is probably the actual output, if anything is.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in reversed(paragraphs):
        first_line = p.split("\n", 1)[0].lstrip()
        if not first_line:
            continue
        first_line_lower = first_line.lower()
        # Skip markdown bullet / numbered list / header / blockquote
        if re.match(r"^([-*+]|\d+[.)]|#+|\>)\s", first_line):
            continue
        # Skip if the paragraph is itself a reasoning preamble
        # (ALL-planning failure mode: the entire response is one
        # giant block that starts with "Thinking Process:" and never
        # gets to actual prose)
        if any(first_line_lower.startswith(m) for m in _REASONING_PREAMBLE_MARKERS):
            continue
        # Skip if the paragraph is drafting metadata
        if re.match(
            r"^(critique|draft|alternative|notes?|outline|plan|step \d+):",
            first_line_lower,
        ):
            continue
        return p
    # No plain prose found anywhere — the model spent its entire
    # budget on planning. Return empty so the caller can decide
    # whether to skip emitting at all.
    return ""


class ThinkingNarrator:
    """Accumulates reasoning tokens and periodically summarizes them.

    Thread-safe. Call ``feed()`` from the streaming thread with each
    reasoning token. Summaries are delivered via the ``on_summary``
    callback on a background thread.
    """

    _DEDUP_BACKOFF_INITIAL_S = _DEDUP_BACKOFF_INITIAL_S
    _DEDUP_BACKOFF_MAX_S = _DEDUP_BACKOFF_MAX_S
    _PLAYBACK_CHUNK_DELAY_S = _PLAYBACK_CHUNK_DELAY_S

    def __init__(
        self,
        sink: "NarratorSink",
        *,
        base_url: str = _DEFAULT_NARRATOR_BASE_URL,
        model: str = _DEFAULT_NARRATOR_MODEL,
        api_key: str = "1234",
        wrap_up_base_url: str | None = None,
        wrap_up_model: str | None = None,
        wrap_up_api_key: str | None = None,
    ):
        self._sink = sink

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

        # Wrap-up uses a separate endpoint/model. By default it falls back
        # to the narrator's own server (preserving old behavior), but in
        # practice the harness points it at the grader server because the
        # grader model is free by the time wrap-up fires and can produce a
        # more grounded post-game read than the small narrator model.
        self._wrap_up_base_url = (
            wrap_up_base_url.rstrip("/") if wrap_up_base_url else self._base_url
        )
        self._wrap_up_model = wrap_up_model or self._model
        self._wrap_up_api_key = (
            wrap_up_api_key if wrap_up_api_key is not None else self._api_key
        )

        # State (guarded by _lock)
        self._lock = threading.Lock()
        self._buffer = ""
        self._last_dispatch = 0.0
        self._messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        self._active = False
        self._thinking_start = 0.0
        self._item_header: str | None = None
        self._pending_dispatch = False
        # Wall-clock time the current pending dispatch started, or None
        # if no dispatch is in flight. Used by feed() to detect a wedged
        # dispatch (one whose stream has hung past _DISPATCH_STUCK_TIMEOUT_S)
        # and recover.
        self._dispatch_started_at: float | None = None
        # Generation counter that increments every time feed() force-clears
        # a stuck dispatch. Each dispatch thread captures the generation it
        # was started under and only clears the pending flag if the current
        # generation still matches its own — so a wedged thread that
        # eventually unblocks (via the stream watchdog) won't clobber the
        # state of the dispatch that took its place.
        self._dispatch_generation = 0
        self._prior_statuses: list[str] = []
        self._current_status: str | None = None
        self._thoughts_since_status: list[str] = []
        self._prior_checkpoints: list[str] = []
        self._accepted_since_checkpoint = 0
        self._dedupe_backoff_until = 0.0
        self._dedupe_backoff_s = self._DEDUP_BACKOFF_INITIAL_S

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
            self._item_header = item_header
            system_content = self._compose_system_prompt()
            self._messages = [
                {"role": "system", "content": system_content}
            ]
            self._active = True
            self._pending_dispatch = False
            self._dispatch_started_at = None
            # Bump generation on every item start so any leftover wedged
            # dispatch from the previous item can't clobber the new
            # item's state.
            self._dispatch_generation += 1
            self._prior_statuses = []
            self._current_status = None
            self._thoughts_since_status = []
            self._prior_checkpoints = []
            self._accepted_since_checkpoint = 0
            self._dedupe_backoff_until = 0.0
            self._dedupe_backoff_s = self._DEDUP_BACKOFF_INITIAL_S
        with self._stats_lock:
            # Roll the previous item's dispatch count into the lifetime
            # max-seen stat, then reset for the new item.
            if self._cur_item_dispatches > self._stat_max_dispatches_one_item:
                self._stat_max_dispatches_one_item = self._cur_item_dispatches
            self._cur_item_dispatches = 0
            self._stat_items_started += 1
        logger.info("Narrator session started")

    def _compose_system_prompt(self, *, status_mode: bool = False) -> str:
        base = _STATUS_SYSTEM_PROMPT if status_mode else _SYSTEM_PROMPT
        if self._item_header:
            return base + "\n\nCONTEXT for the play you're calling:\n" + self._item_header
        return base

    def _build_thought_user_content(
        self,
        chunk: str,
        prior_thoughts: list[str],
    ) -> str:
        blocks = [f"Current reasoning excerpt:\n\n{chunk}"]
        if self._current_status:
            blocks.append(f"Current status lane:\n- {self._current_status}")
        if prior_thoughts:
            blocks.append(
                "Recent first-person calls under this status "
                "(do NOT repeat or paraphrase):\n"
                + "\n".join(
                    f"- {thought}"
                    for thought in prior_thoughts[-_THOUGHT_CONTEXT_LIMIT:]
                )
            )
        blocks.append(
            "Write one short first-person thought about a NEW detail in this excerpt. "
            "Stay on the current status lane, but move to a fresh angle, number, "
            "unit, species, rubric criterion, or uncertainty."
        )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_status_user_content(
        chunk: str,
        prior_statuses: list[str],
    ) -> str:
        blocks = [f"Current reasoning excerpt:\n\n{chunk}"]
        if prior_statuses:
            blocks.append(
                "Recent status lines:\n"
                + "\n".join(
                    f"- {status}"
                    for status in prior_statuses[-_STATUS_CONTEXT_LIMIT:]
                )
            )
        blocks.append(
            "You are still on the SAME point. Do not invent a new angle. "
            "Compress the ongoing state into one short present-participle status line."
        )
        return "\n\n".join(blocks)

    def _build_checkpoint_user_content(
        self,
        chunk: str,
        accepted_line: str,
        prior_thoughts: list[str],
        prior_statuses: list[str],
        prior_checkpoints: list[str],
    ) -> str:
        blocks = [f"Current reasoning excerpt:\n\n{chunk}"]
        blocks.append(f"Latest accepted narrator line:\n- {accepted_line}")
        if self._current_status:
            blocks.append(f"Current status lane:\n- {self._current_status}")
        if prior_thoughts:
            blocks.append(
                "Recent accepted first-person thoughts:\n"
                + "\n".join(
                    f"- {thought}"
                    for thought in prior_thoughts[-_CHECKPOINT_CONTEXT_LIMIT:]
                )
            )
        if prior_statuses:
            blocks.append(
                "Recent status lines:\n"
                + "\n".join(
                    f"- {status}"
                    for status in prior_statuses[-_CHECKPOINT_CONTEXT_LIMIT:]
                )
            )
        if prior_checkpoints:
            blocks.append(
                "Recent checkpoints (do not repeat them):\n"
                + "\n".join(
                    f"- {checkpoint}"
                    for checkpoint in prior_checkpoints[-_CHECKPOINT_CONTEXT_LIMIT:]
                )
            )
        blocks.append(
            "Write one compact checkpoint line that captures the durable issue, "
            "evidence, or likely direction of the grading call."
        )
        return "\n\n".join(blocks)

    def _retry_duplicate_as_status(
        self,
        chunk: str,
        prior_statuses: list[str],
    ) -> tuple[str | None, str, str | None]:
        status_user_content = self._build_status_user_content(
            chunk, prior_statuses
        )
        messages = [
            {"role": "system", "content": self._compose_system_prompt(status_mode=True)},
            {"role": "user", "content": status_user_content},
        ]

        full = self._chat_completion_stream(messages, on_delta=lambda _delta: None)
        full = full.strip()
        if not full:
            return None, status_user_content, None
        if _status_line_breaks_contract(full):
            return None, status_user_content, full

        if any(
            self._lines_too_similar(
                full, prev, threshold=_STATUS_SIMILARITY_THRESHOLD
            )
            for prev in prior_statuses
        ):
            return None, status_user_content, full

        return full, status_user_content, None

    @staticmethod
    def _playback_chunks(text: str) -> list[str]:
        chunks = re.findall(r"\S+\s*", text)
        return chunks or ([text] if text else [])

    def _play_accepted_line(self, text: str, *, mode: str = "thought") -> None:
        chunks = self._playback_chunks(text)
        if not chunks:
            return
        delay_s = self._PLAYBACK_CHUNK_DELAY_S
        for index, chunk in enumerate(chunks):
            self._sink.write_delta(chunk, mode=mode)
            if delay_s > 0 and index < len(chunks) - 1:
                time.sleep(delay_s)

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

        Signals the sink with start_wrap_up() before the (slow) chat
        completion call begins, so the reader can show a 'writing
        post-game commentary...' placeholder immediately and the user
        knows the script is alive while the grader is generating.
        """
        # Fire the placeholder BEFORE the slow call so the reader can
        # show "writing post-game commentary..." while the grader works.
        try:
            self._sink.start_wrap_up()
        except Exception:
            logger.exception("Failed to signal wrap_up_pending to sink")

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
                f"points, no 'Thinking Process:' planning, no drafting "
                f"or self-critique — output ONLY the final commentary "
                f"prose itself.\n\n"
                f"/no_think"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a chemistry-grading sports commentator "
                        "delivering a post-game wrap-up. Dry, arch, willing "
                        "to call out weaknesses, willing to credit "
                        "strengths. 2-4 full sentences. No preamble. "
                        "DO NOT show any planning, reasoning, drafting, or "
                        "'Thinking Process:' sections — output only the "
                        "final commentary prose. Your entire response must "
                        "BE the wrap-up itself, nothing else."
                    ),
                },
                {"role": "user", "content": payload},
            ]
            # Wrap-up runs on the grader endpoint with a generous token
            # budget. Qwen-class thinking-mode graders can spend
            # thousands of tokens on internal planning before producing
            # the final prose, and `<think>` tagging is unreliable
            # (observed 2026-04-08: a 3823-char wrap-up consisting
            # entirely of plain-markdown "Thinking Process:" planning
            # with no `<think>` tags and no actual prose at the end —
            # the model never reached its final output because it ran
            # out of budget mid-draft). 16k gives the model room to
            # both think AND finish writing the actual commentary, so
            # the strip pass downstream has something real to extract.
            text = self._chat_completion(
                messages,
                max_tokens=16384,
                temperature=0.8,
                base_url=self._wrap_up_base_url,
                model=self._wrap_up_model,
                api_key=self._wrap_up_api_key,
                timeout=120,
            )
            text = _strip_reasoning_preamble(text)
            if text:
                self._sink.write_wrap_up(text)
            else:
                logger.warning(
                    "Wrap-up text was empty after stripping reasoning preamble; skipping"
                )
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

            truth_score = getattr(item, "truth_score", item.professor_score)
            corrected_truth = (
                getattr(item, "corrected_score", None) is not None
                and abs(truth_score - item.professor_score) > 1e-9
            )

            # Verdict drives both the prompt context (which examples to
            # encourage) and the topic line color in the reader.
            if prediction.model_score == truth_score:
                verdict = "MATCHED"
                verdict_short = "match"
            elif prediction.model_score > truth_score:
                verdict = "GRADER OVERSHOT"
                verdict_short = "overshoot"
            else:
                verdict = "GRADER UNDERSHOT"
                verdict_short = "undershoot"
            grader_score_display = _format_score_with_denominator(
                prediction.model_score, item.max_points
            )
            truth_score_display = _format_score_with_denominator(
                truth_score, item.max_points
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
                f"  Grader awarded: {prediction.model_score} pts "
                f"(display as {grader_score_display})\n"
                f"  Grader reasoning: {prediction.model_reasoning[:300]}\n"
            )
            if corrected_truth:
                professor_score_display = _format_score_with_denominator(
                    item.professor_score, item.max_points
                )
                payload += (
                    f"  Truth awarded: {truth_score} pts "
                    f"(display as {truth_score_display})\n"
                    f"  Historical professor awarded: {item.professor_score} pts "
                    f"(display as {professor_score_display}, mark: {item.professor_mark})\n"
                    f"  Historical professor's note: \"{item.notes}\"\n"
                    f"  Correction reason: \"{getattr(item, 'correction_reason', '')}\"\n"
                )
            else:
                payload += (
                    f"  Professor awarded: {truth_score} pts "
                    f"(display as {truth_score_display}, mark: {item.professor_mark})\n"
                    f"  Professor's note: \"{item.notes}\"\n"
                )
            payload += (
                f"  Verdict: {verdict}\n\n"
                f"CRITICAL SEMANTICS: The grader and professor are JUDGES "
                f"who score the STUDENT. When both judges give a low score "
                f"(e.g. 0/4) they are AGREEING TO DOCK the student for "
                f"making an error — NOT failing themselves. Never write "
                f"'both judges missed' or 'both judges failed' when the "
                f"score is low. Low score from both = STUDENT made the "
                f"mistake, JUDGES correctly caught it. High score from "
                f"both = student got it right and judges agreed to "
                f"award credit. The judges only 'miss' something if "
                f"they DISAGREE with each other (verdict = OVERSHOT or "
                f"UNDERSHOT).\n\n"
            )
            if corrected_truth:
                payload += (
                    "For corrected historical items, success means matching "
                    "the corrected truth, not the historical professor mark. "
                    "Mention the historical professor score only as provenance, "
                    "not as the target.\n\n"
                    "Write ONE concise after-action line in this format:\n"
                    '  "Grader: <score> (<one-clause reason>). '
                    'Truth: <score> (<one-clause reason>). · '
                    'Historical prof: <score> (<one-clause provenance note>)"\n\n'
                )
            else:
                payload += (
                    "Write ONE concise after-action line in this format:\n"
                    '  "Grader: <score> (<one-clause reason>). '
                    'Prof: <score> (<one-clause reason>). · '
                    '<optional short coda>"\n\n'
                )
            payload += (
                f"Style rules:\n"
                f"- The score+reason portion should be matter-of-fact and question-specific.\n"
                f"- The optional coda should be fresh, concrete, and tied to this exact item.\n"
                f"- Prefer no coda to a stock phrase.\n"
                f"- Do not reuse canned taglines or recurring catchphrases.\n"
                f"- Avoid slogan-like codas, victory laps, and recurring comic patter.\n"
                f"- When the scores match, describe agreement plainly without celebratory catchphrases.\n"
                f"- When the scores differ, explicitly describe the disagreement.\n"
                f"- Never describe disagreement as agreement.\n"
                f"- Never say the judges 'missed' something when they both assigned the student a low score; "
                f"that means they both caught the student's error.\n\n"
                f"Output ONE line only, no preamble, no quotes around your output."
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
            # max_tokens 120 → 256: 120 was tight enough that any
            # preamble could push the actual after-action line off the
            # end of the budget. timeout 15 → 60: this call runs
            # immediately after the per-line bonsai pipeline has just
            # been thrashed, so the server can be under transient load
            # and a 15s timeout produces silent ConnectionResetError /
            # TimeoutError (swallowed by the broad except below). 60s
            # gives the bonsai server room to drain and respond.
            text = ""
            try:
                text = self._chat_completion(
                    messages,
                    max_tokens=256,
                    temperature=0.8,
                    repetition_penalty=1.005,
                    presence_penalty=1.0,
                    timeout=60,
                )
            except Exception:
                logger.exception(
                    "After-action chat call failed for %s/%s",
                    item.exam_id, item.question_id,
                )
            if text:
                text = _sanitize_after_action_text(text)
                text = _normalize_after_action_scores(
                    text,
                    grader_score=prediction.model_score,
                    truth_score=truth_score,
                    max_points=item.max_points,
                    historical_professor_score=(
                        item.professor_score if corrected_truth else None
                    ),
                )
                logger.info("After-action: %s", text)
                self._sink.write_topic(
                    f"{elapsed:.0f}s · {text}",
                    verdict=verdict_short,
                    grader_score=prediction.model_score,
                    truth_score=truth_score,
                    max_points=item.max_points,
                )
            else:
                # Fallback: bonsai returned empty or raised. Always
                # emit SOMETHING so the user can see the item finished
                # and how long it took, even when the verdict line
                # couldn't be generated. Without this, the item
                # silently has no topic at all and the run looks
                # broken — which is exactly what hid the item-3
                # failure mode in the wild.
                logger.warning(
                    "After-action empty for %s/%s — emitting bare timing fallback",
                    item.exam_id, item.question_id,
                )
                self._sink.write_topic(
                    f"{elapsed:.0f}s · (after-action unavailable)",
                    verdict=verdict_short,
                    grader_score=prediction.model_score,
                    truth_score=truth_score,
                    max_points=item.max_points,
                )
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

            # Stuck-dispatch recovery: if a dispatch has been "pending"
            # for longer than _DISPATCH_STUCK_TIMEOUT_S, its thread is
            # wedged (most likely on a stream read that's not getting
            # bytes — the in-loop wallclock can't fire if iteration
            # itself is blocked). Force-clear the flag and bump the
            # generation so the wedged thread won't clobber the
            # replacement when it eventually unblocks via the stream
            # watchdog. Without this recovery, a single wedged dispatch
            # silently kills the narrator for the rest of the run.
            if (
                self._pending_dispatch
                and self._dispatch_started_at is not None
                and (now - self._dispatch_started_at) > _DISPATCH_STUCK_TIMEOUT_S
            ):
                stuck_for = now - self._dispatch_started_at
                logger.warning(
                    "Narrator dispatch wedged for %.0fs — force-clearing "
                    "pending flag and bumping generation",
                    stuck_for,
                )
                self._pending_dispatch = False
                self._dispatch_started_at = None
                self._dispatch_generation += 1

            if now < self._dedupe_backoff_until:
                return

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
            self._dispatch_started_at = now
            my_generation = self._dispatch_generation
        with self._stats_lock:
            self._cur_item_dispatches += 1

        t = threading.Thread(
            target=self._dispatch,
            args=(chunk, my_generation),
            daemon=True,
        )
        t.start()

    def _dispatch(self, chunk: str, my_generation: int) -> None:
        with self._stats_lock:
            self._stat_dispatches_total += 1
        try:
            with self._lock:
                prior_thoughts = list(self._thoughts_since_status)
                prior_statuses = list(self._prior_statuses)
            user_content = self._build_thought_user_content(chunk, prior_thoughts)
            messages = [
                {"role": "system", "content": self._compose_system_prompt()},
                {"role": "user", "content": user_content},
            ]

            full = self._chat_completion_stream(
                messages, on_delta=lambda _delta: None
            )
            full = full.strip()

            if not full:
                with self._stats_lock:
                    self._stat_drops_empty += 1
                self._sink.write_drop("empty", "")
                return

            # Dedup check
            if any(
                self._lines_too_similar(full, prev)
                for prev in prior_thoughts
            ):
                logger.info(
                    "Narrator: first-person summary was repetitive, retrying in status mode: %s",
                    full,
                )
                status_full, _status_user_content, status_drop = self._retry_duplicate_as_status(
                    chunk, prior_statuses
                )
                if not status_full:
                    with self._stats_lock:
                        self._stat_drops_dedup += 1
                    self._sink.write_drop("dedup", full)
                    if status_drop:
                        self._sink.write_drop("dedup-status", status_drop)
                    logger.info("Narrator: dropped repetitive summary: %s", full)
                    with self._lock:
                        self._dedupe_backoff_until = (
                            time.monotonic() + self._dedupe_backoff_s
                        )
                        self._dedupe_backoff_s = min(
                            self._dedupe_backoff_s * 2,
                            self._DEDUP_BACKOFF_MAX_S,
                        )
                    return
                full = status_full
                committed_mode = "status"
            else:
                committed_mode = "thought"

            # Accept — now that dedup has settled, play the accepted line
            # into the live row and then commit it.
            self._play_accepted_line(full, mode=committed_mode)
            self._sink.commit_live(mode=committed_mode)

            checkpoint_context: tuple[str, str, list[str], list[str], list[str]] | None = None
            with self._lock:
                self._dedupe_backoff_until = 0.0
                self._dedupe_backoff_s = self._DEDUP_BACKOFF_INITIAL_S
                if committed_mode == "status":
                    self._current_status = full
                    self._prior_statuses.append(full)
                    self._thoughts_since_status = []
                else:
                    self._thoughts_since_status.append(full)
                self._accepted_since_checkpoint += 1
                if self._accepted_since_checkpoint >= _CHECKPOINT_EVERY_ACCEPTED:
                    checkpoint_context = (
                        chunk,
                        full,
                        list(self._thoughts_since_status),
                        list(self._prior_statuses),
                        list(self._prior_checkpoints),
                    )
                    self._accepted_since_checkpoint = 0
            with self._stats_lock:
                self._stat_summaries_emitted += 1
            if checkpoint_context is not None:
                (
                    checkpoint_chunk,
                    checkpoint_line,
                    checkpoint_thoughts,
                    checkpoint_statuses,
                    checkpoint_prior,
                ) = checkpoint_context
                checkpoint_user_content = self._build_checkpoint_user_content(
                    checkpoint_chunk,
                    checkpoint_line,
                    checkpoint_thoughts,
                    checkpoint_statuses,
                    checkpoint_prior,
                )
                checkpoint_messages = [
                    {"role": "system", "content": _CHECKPOINT_SYSTEM_PROMPT},
                    {"role": "user", "content": checkpoint_user_content},
                ]
                checkpoint_text = self._chat_completion(checkpoint_messages).strip()
                if checkpoint_text:
                    self._sink.write_checkpoint(checkpoint_text)
                    with self._lock:
                        self._prior_checkpoints.append(checkpoint_text)
            logger.info("Narrator summary: %s", full)
        except Exception:
            logger.exception("Narrator dispatch failed")
        finally:
            # Only clear the pending flag if we're STILL the active
            # dispatch generation. If feed() force-cleared a stuck
            # dispatch (us) and started a replacement, the replacement
            # owns the flag now and we must not touch it.
            with self._lock:
                if self._dispatch_generation == my_generation:
                    self._pending_dispatch = False
                    self._dispatch_started_at = None
                else:
                    logger.info(
                        "Narrator dispatch (gen %d) finished after being "
                        "superseded by gen %d — not clearing pending flag",
                        my_generation,
                        self._dispatch_generation,
                    )

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
        temperature: float = 0.8,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.002,
        repetition_penalty: float = 1.005,
        presence_penalty: float = 1.0,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 15,
    ) -> str:
        """Synchronous (non-streaming) call. Used for the collapsed
        per-item summary at the end of each item, and (with overrides)
        for the end-of-run wrap-up against the grader server.

        Defaults are tuned for bonsai (the live narrator model). The
        wrap-up call routes through this function with explicit
        overrides for base_url/model/api_key/temperature/max_tokens
        to hit the grader server instead; min_p and repetition_penalty
        leak through with bonsai values, but the deviations from
        Qwen's expected defaults (min_p=0, repetition_penalty=1.0)
        are minimal (0.002 and 1.001 respectively) and not worth
        adding more override surface for.
        """
        eff_base = (base_url or self._base_url).rstrip("/")
        eff_model = model or self._model
        eff_api_key = api_key if api_key is not None else self._api_key
        body = {
            "model": eff_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if eff_api_key:
            headers["Authorization"] = f"Bearer {eff_api_key}"

        req = urllib.request.Request(
            f"{eff_base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        t0 = time.monotonic()
        # Open the response separately so we can close the socket
        # explicitly on KeyboardInterrupt — that tells the OMLX
        # server to abort generation instead of running it to
        # completion in the background.
        resp = urllib.request.urlopen(req, timeout=timeout)
        try:
            result = json.loads(resp.read().decode())
        except KeyboardInterrupt:
            try:
                resp.close()
            except Exception:
                pass
            raise
        finally:
            try:
                resp.close()
            except Exception:
                pass
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
        temperature: float = 0.8,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.002,
        repetition_penalty: float = 1.005,
        presence_penalty: float = 1.0,
        max_chars: int = 350,
        max_seconds: float = 20.0,
    ) -> str:
        """Streaming call. Calls on_delta(token) per content delta and
        returns the full accumulated text.

        Three abort conditions for runaway bonsai loops, in addition
        to the natural [DONE] from the server:

        1. max_chars — hard cap on accumulated content. OMLX doesn't
           strictly enforce max_tokens for bonsai streams, so without
           a client-side cap a single dispatch can run for minutes
           producing 1000+ chars of looped phrases.
        2. max_seconds — wallclock cap. If we've been streaming this
           one dispatch for too long, abort regardless of length.
        3. In-stream substring repetition. If the trailing N chars of
           the buffer already appear earlier in the buffer, bonsai is
           in a literal token-level loop. Bail immediately.

        On any abort condition we close the response (signals OMLX
        to stop generating server-side) and return the truncated
        text. The dispatch path then runs the normal dedup check on
        the truncated text — usually it'll get rejected as a dup of
        a prior summary, since looped output tends to overlap heavily
        with whatever bonsai already said.
        """
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
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
        # In-stream substring repetition check params
        loop_check_min = 70   # don't check until we have at least this many chars
        loop_check_window = 30  # the trailing window we look for earlier — short
                                 # enough to catch ~30-char repeating phrases like
                                 # "if the student's answer is correct" which is
                                 # bonsai's most common loop unit
        abort_reason: str | None = None

        # Set the urlopen socket timeout to bound BOTH the initial
        # response wait AND every subsequent stream read. urllib's
        # `timeout` parameter installs a socket-level timeout that
        # applies to all reads on the connection, so if the server
        # accepts the request, opens the response stream, then stops
        # sending bytes (server stalled, KV cache write blocked,
        # model wedged on a sample, OR the bonsai box went down
        # mid-request and TCP keepalive hasn't fired yet), the next
        # read raises TimeoutError instead of blocking forever.
        #
        # Without this, the in-loop wallclock check below CAN'T fire
        # because it only runs when the iteration body executes —
        # which requires bytes to arrive. A single wedged stream
        # would silently kill the entire narrator: dispatch thread
        # blocked forever, _pending_dispatch stuck True, no further
        # dispatches enqueued. Observed in the wild on 2026-04-08
        # when the bonsai box was brought down mid-run.
        #
        # max_seconds + slop is comfortably bigger than the in-loop
        # wallclock cap, so the in-loop check is still the primary
        # abort path for the normal "slow but flowing" case (it
        # produces a clean abort_reason="max_seconds" with the
        # partial buffer intact). The socket timeout only kicks in
        # when the iteration is fully blocked, in which case we
        # don't have a buffer to preserve anyway.
        resp = urllib.request.urlopen(
            req,
            timeout=max_seconds + _STREAM_WATCHDOG_SLOP_S,
        )

        try:
            for raw_line in resp:
                # wallclock cap
                if time.monotonic() - t0 > max_seconds:
                    abort_reason = "max_seconds"
                    break

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

                    # max_chars cap
                    if len(full) >= max_chars:
                        abort_reason = "max_chars"
                        break

                    # in-stream substring repetition check — only kicks
                    # in once we have enough text. If the last N chars
                    # already appear in the earlier portion of the
                    # buffer, bonsai is looping.
                    if len(full) > loop_check_min:
                        tail = full[-loop_check_window:]
                        earlier = full[:-loop_check_window]
                        if tail and tail in earlier:
                            abort_reason = "stream_loop_detected"
                            break
        finally:
            try:
                resp.close()
            except Exception:
                pass

        if abort_reason:
            logger.info(
                "Narrator stream aborted (%s) after %.1fs, %d chars",
                abort_reason,
                time.monotonic() - t0,
                len(full),
            )
        elapsed = time.monotonic() - t0
        logger.info(
            "Narrator stream: %.2fs, %d chars", elapsed, len(full)
        )
        return full.strip()
