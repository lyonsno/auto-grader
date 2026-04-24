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
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_grader.narrator_sink import NarratorSink

logger = logging.getLogger(__name__)

_DEFAULT_NARRATOR_MODEL = "Bonsai-8B-mlx-1bit"
_DEFAULT_NARRATOR_BASE_URL = "http://nlm2pr.local:8002"

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
- Never write I / I'm / I am.
- Do NOT give a final verdict or score.
- Be concrete: mention the actual issue, quantity, unit, species, or \
rubric criterion when possible.
- This is an IN-PROGRESS state line, not a new thought and not a conclusion.
- No preamble, no quotes. Output ONLY the status line.

Good examples:
- Rechecking photon-energy formula.
- Tracing valence-electron count.
- Comparing net ionic form.

Bad -> good rewrite:
- Bad: "I'm rechecking the units."
- Good: "Rechecking units."
"""

_CHECKPOINT_SYSTEM_PROMPT = """\
You write compact history checkpoints for a chemistry-grading narrator.

Summarize the current state of play as one neutral sentence that is worth
keeping in durable history. This is not a live thought and not a status line.

Rules:
- ONE sentence. 8-18 words.
- Do NOT use first person.
- Start with a short label: "Core issue:", "Evidence:", or "Context:".
- Be specific to this exact item: mention the concrete chemistry issue,
  quantity, unit, species, or rubric dimension when possible.
- Prefer a small stable vocabulary over novelty. If the issue has already
  been summarized recently, reuse the same canonical wording instead of
  paraphrasing it.
- No quotes, no bullet points, no preamble. Output ONLY the checkpoint line.
"""


def _qwen36_chat_template_kwargs(model: str | None) -> dict[str, bool] | None:
    if not model:
        return None
    lowered = model.casefold()
    if "qwen3.6" in lowered or "qwen3p6" in lowered:
        return {"enable_thinking": False}
    return None


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
_CHECKPOINT_TEMPERATURE = 0.5
_CHECKPOINT_REPETITION_PENALTY = 1.0
_CHECKPOINT_PRESENCE_PENALTY = 0.0
_DEDUP_GROOMING_THRESHOLD = 2
_PLAYBACK_CHUNK_DELAY_S = 0.03
_IDLE_LEGIBILITY_DELAY_S = 1.0
_MAX_LEGIBILITY_EXTRA_ROWS = 2
_DOSSIER_ELAPSED_SECONDS = 60.0
_DOSSIER_LONG_ELAPSED_SECONDS = 120.0
_DOSSIER_DEDUPE_STREAK_THRESHOLD = 2
_DOSSIER_REASONING_HINT_RE = re.compile(
    r"\b("
    r"ambigu(?:ity|ous)?|uncertain|unclear|illegible|handwriting|glyph|smudge|"
    r"cross(?:ed)?[- ]out|scratch|looks like|could be|lean(?:s|ing)? toward|"
    r"salvage|partial credit|charit(?:y|able)"
    r")\b",
    re.IGNORECASE,
)

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

_REVIEW_NEEDED_HINT_RE = re.compile(
    r"\b(review (?:is )?(?:needed|warranted)|human review|ambigu(?:ity|ous)|uncertain)\b",
    re.IGNORECASE,
)

_GRADER_SCORE_RE = re.compile(r"(Grader:\s*)([^ ]+)")
_PROF_SCORE_RE = re.compile(r"(Prof:\s*)([^ ]+)")
_TRUTH_SCORE_RE = re.compile(r"(Truth:\s*)([^ ]+)")
_HISTORICAL_PROF_SCORE_RE = re.compile(r"(Historical\s+[Pp]rof:\s*)([^ ]+)")
_STATUS_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i am|i've|i'll)\b", re.IGNORECASE
)


def _format_score_with_denominator(score: float, max_points: float) -> str:
    return f"{score:g}/{max_points:g}"


def _format_dossier_question_context(template_question: dict | None) -> str:
    if not template_question:
        return ""
    lines: list[str] = []
    prompt = str(template_question.get("prompt", "")).strip()
    if prompt:
        lines.append(f"Question prompt: {prompt}")
    answer = template_question.get("answer")
    if isinstance(answer, dict):
        rubric = answer.get("rubric")
        if isinstance(rubric, list):
            rubric_parts: list[str] = []
            for entry in rubric:
                if not isinstance(entry, dict):
                    continue
                criterion = str(entry.get("criterion", "")).strip()
                if not criterion:
                    continue
                points = entry.get("points")
                if points is None:
                    rubric_parts.append(criterion)
                else:
                    rubric_parts.append(f"{criterion} ({float(points):g} pt)")
            if rubric_parts:
                lines.append(f"Rubric: {'; '.join(rubric_parts)}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _format_dossier_acceptable_band_context(item: Any) -> str:
    floor_raw = getattr(item, "acceptable_score_floor", None)
    ceiling_raw = getattr(item, "acceptable_score_ceiling", None)
    reason = str(getattr(item, "acceptable_score_reason", "")).strip()
    if floor_raw is None and ceiling_raw is None and not reason:
        return ""

    max_points = float(getattr(item, "max_points", 0.0))
    truth_score = float(
        getattr(item, "truth_score", getattr(item, "professor_score", 0.0))
    )
    floor = float(floor_raw) if floor_raw is not None else truth_score
    ceiling = float(ceiling_raw) if ceiling_raw is not None else truth_score
    lines = [
        "Acceptable score band: "
        f"{_format_score_with_denominator(floor, max_points)} to "
        f"{_format_score_with_denominator(ceiling, max_points)}"
    ]
    if reason:
        lines.append(f"Acceptable band reason: {reason}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _ScoreBandClassification:
    truth: float
    floor: float
    ceiling: float
    band_present: bool
    verdict: str
    verdict_short: str


def _classify_score_against_band(
    model_score: float,
    item,
) -> _ScoreBandClassification:
    truth_score = getattr(item, "truth_score", item.professor_score)
    floor_raw = getattr(item, "acceptable_score_floor", None)
    ceiling_raw = getattr(item, "acceptable_score_ceiling", None)
    floor = float(floor_raw) if floor_raw is not None else float(truth_score)
    ceiling = (
        float(ceiling_raw) if ceiling_raw is not None else float(truth_score)
    )
    band_present = floor_raw is not None or ceiling_raw is not None
    truth = float(truth_score)
    if abs(model_score - truth) < 1e-9:
        return _ScoreBandClassification(
            truth=truth,
            floor=floor,
            ceiling=ceiling,
            band_present=band_present,
            verdict="MATCHED TRUTH",
            verdict_short="match",
        )
    if floor <= model_score <= ceiling:
        return _ScoreBandClassification(
            truth=truth,
            floor=floor,
            ceiling=ceiling,
            band_present=band_present,
            verdict="WITHIN RANGE",
            verdict_short="within_band",
        )
    if model_score > ceiling:
        return _ScoreBandClassification(
            truth=truth,
            floor=floor,
            ceiling=ceiling,
            band_present=band_present,
            verdict="GRADER OVERSHOT",
            verdict_short="overshoot",
        )
    return _ScoreBandClassification(
        truth=truth,
        floor=floor,
        ceiling=ceiling,
        band_present=band_present,
        verdict="GRADER UNDERSHOT",
        verdict_short="undershoot",
    )


def _band_commentary_clause(
    classification: _ScoreBandClassification,
    *,
    max_points: float,
) -> str | None:
    if not classification.band_present:
        return None
    floor_display = _format_score_with_denominator(classification.floor, max_points)
    ceiling_display = _format_score_with_denominator(
        classification.ceiling, max_points
    )
    if classification.verdict_short == "match":
        if abs(classification.truth - classification.ceiling) < 1e-9:
            position = "at the ceiling"
        elif abs(classification.truth - classification.floor) < 1e-9:
            position = "at the floor"
        else:
            position = "at the truth target"
    elif classification.verdict_short == "ceiling":
        position = "at the ceiling"
    elif classification.verdict_short == "within_band":
        position = "within range, below ceiling"
    elif classification.verdict_short == "overshoot":
        position = "above range"
    else:
        position = "below range"
    return (
        f"Acceptable band: {floor_display} to {ceiling_display}; "
        f"grader is {position}."
    )


def _band_detail_line(
    classification: _ScoreBandClassification,
    *,
    max_points: float,
) -> str | None:
    if not classification.band_present:
        return None
    floor_display = _format_score_with_denominator(classification.floor, max_points)
    ceiling_display = _format_score_with_denominator(
        classification.ceiling, max_points
    )
    return f"Band: {floor_display} to {ceiling_display}"


def _after_action_topic_text(
    *,
    elapsed: float,
    prediction: Any,
    item: Any,
    truth_score: float,
    corrected_truth: bool,
    band_classification: _ScoreBandClassification,
) -> str:
    grader_score_display = _format_score_with_denominator(
        prediction.model_score, item.max_points
    )
    truth_score_display = _format_score_with_denominator(
        truth_score, item.max_points
    )
    if corrected_truth:
        main_line = (
            f"{elapsed:.0f}s · Grader: {grader_score_display} · "
            f"Truth: {truth_score_display}"
        )
        detail_parts = [
            "Historical prof: "
            + _format_score_with_denominator(item.professor_score, item.max_points)
        ]
    else:
        main_line = (
            f"{elapsed:.0f}s · Grader: {grader_score_display} · "
            f"Prof: {truth_score_display}"
        )
        detail_parts = []

    band_detail = _band_detail_line(
        band_classification,
        max_points=item.max_points,
    )
    if band_detail is not None:
        detail_parts.append(band_detail)

    if not detail_parts:
        return main_line
    return main_line + "\n" + " · ".join(detail_parts)


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
    # Status is its own contract surface: a non-first-person present-participle
    # sticky lane, not a lightly edited thought. If the model returns first
    # person here, that is a contract failure to drop, not text to salvage.
    return bool(_STATUS_FIRST_PERSON_RE.search(text))

def _reasoning_warrants_human_review(text: str) -> bool:
    lowered = text.lower()
    return (
        "human review" in lowered
        or "review warranted" in lowered
        or "needs review" in lowered
    )


def _checkpoint_line_breaks_contract(text: str) -> bool:
    stripped = ThinkingNarrator._canonicalize_checkpoint_text(text)
    label, body = ThinkingNarrator._split_checkpoint_label(stripped)
    if label not in {"core issue", "evidence", "context"}:
        return True
    if not body:
        return True
    return bool(_STATUS_FIRST_PERSON_RE.search(body))


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
        self._dedupe_streak = 0
        self._item_token_count = 0
        self._item_max_dedupe_streak = 0
        self._legibility_jobs: list[dict] = []
        self._idle_legibility_pending = False
        self._idle_legibility_generation = 0

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
        self._flush_pending_legibility_before_item_reset()
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
            self._dedupe_streak = 0
            self._item_token_count = 0
            self._item_max_dedupe_streak = 0
            self._legibility_jobs = []
            self._idle_legibility_pending = False
            self._idle_legibility_generation += 1
        with self._stats_lock:
            # Roll the previous item's dispatch count into the lifetime
            # max-seen stat, then reset for the new item.
            if self._cur_item_dispatches > self._stat_max_dispatches_one_item:
                self._stat_max_dispatches_one_item = self._cur_item_dispatches
            self._cur_item_dispatches = 0
            self._stat_items_started += 1
        logger.info("Narrator session started")

    def _flush_pending_legibility_before_item_reset(self) -> None:
        while True:
            with self._lock:
                if self._pending_dispatch or self._buffer or not self._legibility_jobs:
                    return
            try:
                self._flush_idle_legibility_once()
            except Exception:
                logger.exception(
                    "Failed to flush pending legibility rows before starting next item"
                )
                return

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
        rejected_thought: str,
        prior_statuses: list[str],
    ) -> str:
        blocks = [
            "Rejected first-person line to compress:\n"
            f"- {rejected_thought}"
        ]
        blocks.append(f"Current reasoning excerpt:\n\n{chunk}")
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
            "Rewrite that same substance as one short non-first-person "
            "present-participle status line."
        )
        blocks.append(
            'If the rejected line begins with "I\'m" or "I am", remove that '
            "first-person scaffolding and keep the participle."
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
                "Recent checkpoints (reuse the same wording if the issue is the same; do not paraphrase it):\n"
                + "\n".join(
                    f"- {checkpoint}"
                    for checkpoint in prior_checkpoints[-_CHECKPOINT_CONTEXT_LIMIT:]
                )
            )
        blocks.append(
            "Write one compact checkpoint line that captures the durable issue, "
            "evidence, or likely direction of the grading call."
        )
        blocks.append(
            "If the issue matches a recent checkpoint, reuse its wording instead of paraphrasing it."
        )
        return "\n\n".join(blocks)

    def _emit_checkpoint_from_context(
        self,
        chunk: str,
        accepted_line: str,
        prior_thoughts: list[str],
        prior_statuses: list[str],
        prior_checkpoints: list[str],
    ) -> None:
        checkpoint_user_content = self._build_checkpoint_user_content(
            chunk,
            accepted_line,
            prior_thoughts,
            prior_statuses,
            prior_checkpoints,
        )
        checkpoint_messages = [
            {"role": "system", "content": _CHECKPOINT_SYSTEM_PROMPT},
            {"role": "user", "content": checkpoint_user_content},
        ]
        checkpoint_text = self._chat_completion(
            checkpoint_messages,
            temperature=_CHECKPOINT_TEMPERATURE,
            repetition_penalty=_CHECKPOINT_REPETITION_PENALTY,
            presence_penalty=_CHECKPOINT_PRESENCE_PENALTY,
        ).strip()
        if not checkpoint_text:
            return
        checkpoint_text = self._canonicalize_checkpoint_text(checkpoint_text)
        if _checkpoint_line_breaks_contract(checkpoint_text):
            self._sink.write_drop("contract-checkpoint", checkpoint_text)
            return
        if any(
            self._checkpoint_lines_share_basin(
                checkpoint_text,
                prev,
            )
            for prev in prior_checkpoints
        ):
            self._sink.write_drop("dedup-checkpoint", checkpoint_text)
            return
        self._sink.write_checkpoint(checkpoint_text)
        with self._lock:
            self._prior_checkpoints.append(checkpoint_text)

    def _maybe_groom_after_repeated_dedup(
        self,
        chunk: str,
        prior_thoughts: list[str],
        prior_statuses: list[str],
    ) -> None:
        with self._lock:
            if self._dedupe_streak < _DEDUP_GROOMING_THRESHOLD:
                return
            accepted_line = (
                self._current_status
                or (prior_statuses[-1] if prior_statuses else "")
                or (prior_thoughts[-1] if prior_thoughts else "")
            )
            prior_checkpoints = list(self._prior_checkpoints)
        if not accepted_line:
            return
        self._emit_checkpoint_from_context(
            chunk,
            accepted_line,
            prior_thoughts,
            prior_statuses,
            prior_checkpoints,
        )

    def _retry_duplicate_as_status(
        self,
        chunk: str,
        rejected_thought: str,
        prior_statuses: list[str],
    ) -> tuple[str | None, str, str | None, str | None]:
        """Retry a repetitive thought as a strict status line.

        Important contract note: the status lane is not a repaired first-person
        thought. The retry must already satisfy the non-first-person
        present-participle form, or it is rejected as ``contract-status``.
        """
        status_user_content = self._build_status_user_content(
            chunk, rejected_thought, prior_statuses
        )
        messages = [
            {"role": "system", "content": self._compose_system_prompt(status_mode=True)},
            {"role": "user", "content": status_user_content},
        ]

        full = self._chat_completion_stream(messages, on_delta=lambda _delta: None)
        full = full.strip()
        if not full:
            return None, status_user_content, None, None
        if _status_line_breaks_contract(full):
            return None, status_user_content, "contract-status", full

        if any(
            self._lines_too_similar(
                full, prev, threshold=_STATUS_SIMILARITY_THRESHOLD
            )
            for prev in prior_statuses
        ):
            return None, status_user_content, "dedup-status", full

        return full, status_user_content, None, None

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

    def _schedule_idle_legibility_if_needed(self) -> None:
        with self._lock:
            if self._idle_legibility_pending or not self._legibility_jobs:
                return
            generation = self._idle_legibility_generation
            self._idle_legibility_pending = True

        t = threading.Thread(
            target=self._run_idle_legibility_after_delay,
            args=(generation,),
            daemon=True,
        )
        t.start()

    def _run_idle_legibility_after_delay(self, generation: int) -> None:
        emitted = False
        try:
            time.sleep(_IDLE_LEGIBILITY_DELAY_S)
            emitted = self._flush_idle_legibility_once(generation)
        except Exception:
            logger.exception("Idle legibility flush failed")
        finally:
            with self._lock:
                if self._idle_legibility_generation == generation:
                    self._idle_legibility_pending = False
                    should_reschedule = bool(self._legibility_jobs)
                else:
                    should_reschedule = False
            if emitted and should_reschedule:
                self._schedule_idle_legibility_if_needed()

    def _flush_idle_legibility_once(
        self,
        generation: int | None = None,
    ) -> bool:
        with self._lock:
            if generation is not None and generation != self._idle_legibility_generation:
                return False
            if self._pending_dispatch or self._buffer or not self._legibility_jobs:
                return False
            job = self._legibility_jobs.pop(0)
            current_generation = self._idle_legibility_generation

        if job["kind"] == "generated":
            text = self._chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You write one compact structured history row for a chemistry-grading narrator. "
                            "Return ONLY the row body, not the label."
                        ),
                    },
                    {"role": "user", "content": job["prompt"]},
                ],
                max_tokens=80,
                temperature=0.6,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                timeout=30,
            ).strip()
        elif job["kind"] == "generated_dossier":
            text = self._chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You write a compact background dossier for one chemistry-grading item. "
                            "Return ONLY a valid JSON object with string fields "
                            '"read", "salvage", and "hinge".'
                        ),
                    },
                    {"role": "user", "content": job["prompt"]},
                ],
                max_tokens=220,
                temperature=0.6,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                timeout=30,
            ).strip()
        else:
            text = str(job["text"]).strip()

        if not text:
            return False

        with self._lock:
            if current_generation != self._idle_legibility_generation:
                return False

        if job["kind"] == "generated_dossier":
            return self._write_dossier_rows(text)

        writer = getattr(self._sink, f"write_{job['row_type']}", None)
        if writer is None:
            drop_writer = getattr(self._sink, "write_drop", None)
            if drop_writer is not None:
                drop_writer("missing-sink-row", f"{job['row_type']}: {text}")
            logger.warning(
                "Narrator sink missing writer for structured row type %s; dropping row",
                job["row_type"],
            )
            return False
        writer(text)
        return True

    def _enqueue_legibility_job(
        self,
        row_type: str,
        *,
        text: str | None = None,
        prompt: str | None = None,
    ) -> None:
        if text is None and prompt is None:
            return
        kind = "generated" if prompt is not None else "literal"
        payload = {"kind": kind, "row_type": row_type}
        if text is not None:
            payload["text"] = text.strip()
        if prompt is not None:
            payload["prompt"] = prompt
        with self._lock:
            self._legibility_jobs.append(payload)

    def _enqueue_dossier_job(self, prompt: str) -> None:
        if not prompt.strip():
            return
        with self._lock:
            self._legibility_jobs.append(
                {"kind": "generated_dossier", "row_type": "dossier", "prompt": prompt}
            )

    @staticmethod
    def _clean_json_response(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _write_dossier_rows(self, text: str) -> bool:
        try:
            payload = json.loads(self._clean_json_response(text))
        except json.JSONDecodeError:
            drop_writer = getattr(self._sink, "write_drop", None)
            if drop_writer is not None:
                drop_writer("dossier-json", text)
            logger.warning("Background dossier was not valid JSON; dropping")
            return False

        emitted = False
        for row_type in ("read", "salvage", "hinge"):
            body = str(payload.get(row_type, "")).strip()
            if not body:
                continue
            writer = getattr(self._sink, f"write_{row_type}", None)
            if writer is None:
                drop_writer = getattr(self._sink, "write_drop", None)
                if drop_writer is not None:
                    drop_writer("missing-sink-row", f"{row_type}: {body}")
                logger.warning(
                    "Narrator sink missing writer for dossier row type %s; dropping row",
                    row_type,
                )
                continue
            writer(body)
            emitted = True
        return emitted

    @staticmethod
    def _reasoning_mentions_dossier_hints(prediction: Any) -> bool:
        reasoning = str(getattr(prediction, "model_reasoning", "")).strip()
        score_basis = str(getattr(prediction, "score_basis", "")).strip()
        combined = "\n".join(part for part in (reasoning, score_basis) if part)
        if not combined:
            return False
        return _DOSSIER_REASONING_HINT_RE.search(combined) is not None

    def _should_enqueue_dossier(
        self,
        elapsed: float,
        prediction: Any,
        item: Any,
    ) -> bool:
        if getattr(prediction, "truncated", False):
            return False
        if elapsed >= _DOSSIER_LONG_ELAPSED_SECONDS:
            return True

        truth_score = getattr(item, "truth_score", item.professor_score)
        score_disagreement = abs(prediction.model_score - truth_score) > 1e-9
        partial_credit = 0.0 < prediction.model_score < item.max_points

        with self._lock:
            max_dedupe_streak = self._item_max_dedupe_streak

        if max_dedupe_streak >= _DOSSIER_DEDUPE_STREAK_THRESHOLD:
            return True
        if elapsed < _DOSSIER_ELAPSED_SECONDS:
            return False
        return (
            self._reasoning_mentions_dossier_hints(prediction)
            or score_disagreement
            or partial_credit
        )

    def _build_dossier_prompt(
        self,
        *,
        elapsed: float,
        prediction: Any,
        item: Any,
        template_question: dict | None = None,
    ) -> str:
        truth_score = getattr(item, "truth_score", item.professor_score)
        with self._lock:
            item_token_count = self._item_token_count
            max_dedupe_streak = self._item_max_dedupe_streak
        question_context = _format_dossier_question_context(template_question)
        acceptable_band_context = _format_dossier_acceptable_band_context(item)

        return (
            "The item is finished. Build a compact trailing dossier for the operator.\n\n"
            f"Item: {item.exam_id}/{item.question_id}\n"
            f"Type: {item.answer_type}\n"
            f"{question_context}"
            f"Elapsed seconds: {elapsed:.1f}\n"
            f"Approx narrator token count observed: {item_token_count}\n"
            f"Max dedupe streak observed: {max_dedupe_streak}\n"
            f"Student answer: {item.student_answer}\n"
            f"Model read: {getattr(prediction, 'model_read', '')}\n"
            f"Model score: {prediction.model_score}/{item.max_points}\n"
            f"Truth score: {truth_score}/{item.max_points}\n"
            f"{acceptable_band_context}"
            f"Professor mark: {item.professor_mark}\n"
            f"Professor note: {item.notes}\n"
            f"Score basis: {getattr(prediction, 'score_basis', '')}\n"
            f"Model reasoning: {getattr(prediction, 'model_reasoning', '')}\n\n"
            "Return JSON only with this shape:\n"
            '{\n'
            '  "read": "<what the student mark/work most plausibly seems to be saying; mention ambiguity if real>",\n'
            '  "salvage": "<what reasoning, setup, or method is still lawfully salvageable>",\n'
            '  "hinge": "<the actual unresolved issue that decides interpretation or score>"\n'
            "}\n\n"
            "Rules:\n"
            "- Keep each value to one compact sentence.\n"
            "- Ground every value in the specific item.\n"
            "- If handwriting is not the issue, say what the student work appears to show instead.\n"
            "- If little is salvageable, say that plainly rather than inventing credit.\n"
            "- If the model score is below the acceptable score floor, do not erase the band reason; name the model/band tension in the hinge.\n"
            "- Use the rubric and acceptable-band reason as counter-evidence when the model rationale overstates how little survives.\n"
            "- The hinge should name the contested deciding issue, not just repeat the score.\n"
            "- No markdown, no prose outside the JSON object."
        )

    @staticmethod
    def _review_needed_text(prediction: Any) -> str | None:
        reasoning = str(getattr(prediction, "model_reasoning", "")).strip()
        confidence = float(getattr(prediction, "model_confidence", 1.0))
        if not reasoning:
            return None
        if not _REVIEW_NEEDED_HINT_RE.search(reasoning) and confidence >= 0.6:
            return None
        return (
            "Human review warranted because the cancellation handwriting "
            "remains ambiguity-sensitive after a bounded pass."
            if "ambigu" in reasoning.lower()
            else "Human review warranted after a bounded ambiguity pass."
        )

    @staticmethod
    def _professor_mismatch_text(item: Any) -> str | None:
        corrected = getattr(item, "corrected_score", None)
        if corrected is None or abs(corrected - item.professor_score) < 1e-9:
            return None
        return (
            f"Historical professor awarded "
            f"{_format_score_with_denominator(item.professor_score, item.max_points)}; "
            f"corrected truth is "
            f"{_format_score_with_denominator(corrected, item.max_points)}."
        )

    @staticmethod
    def _should_emit_basis_row(
        prediction: Any,
        item: Any,
        *,
        review_needed: str | None,
        professor_mismatch: str | None,
    ) -> bool:
        score_basis = str(getattr(prediction, "score_basis", "")).strip()
        if not score_basis:
            return False
        if review_needed or professor_mismatch:
            return True
        if prediction.model_score < item.max_points:
            return True
        truth_score = getattr(item, "corrected_score", None)
        if truth_score is None:
            truth_score = getattr(item, "truth_score", item.professor_score)
        return abs(prediction.model_score - truth_score) > 1e-9

    def _write_legibility_row_now(self, row_type: str, text: str | None) -> bool:
        body = str(text or "").strip()
        if not body:
            return False
        writer = getattr(self._sink, f"write_{row_type}")
        writer(body)
        return True

    def _handle_legibility_rows(self, prediction: Any, item: Any) -> None:
        score_basis = str(getattr(prediction, "score_basis", "")).strip()
        review_needed = self._review_needed_text(prediction)
        professor_mismatch = self._professor_mismatch_text(item)

        if self._should_emit_basis_row(
            prediction,
            item,
            review_needed=review_needed,
            professor_mismatch=professor_mismatch,
        ):
            self._write_legibility_row_now("basis", score_basis)

        extras = 0

        if review_needed and extras < _MAX_LEGIBILITY_EXTRA_ROWS:
            self._write_legibility_row_now("review_marker", review_needed)
            extras += 1

        if professor_mismatch and extras < _MAX_LEGIBILITY_EXTRA_ROWS:
            self._write_legibility_row_now("professor_mismatch", professor_mismatch)
            extras += 1

        if prediction.model_score < item.max_points and extras < _MAX_LEGIBILITY_EXTRA_ROWS:
            if 0.0 < prediction.model_score < item.max_points:
                self._enqueue_legibility_job(
                    "credit_preserved",
                    prompt=(
                        "Write one short row body for 'Credit preserved for:'. "
                        "Use the direct earned-credit basis only, not the deduction. "
                        f"Question: {item.question_id}. "
                        f"Score basis: {getattr(prediction, 'score_basis', '')} "
                        f"Reasoning: {getattr(prediction, 'model_reasoning', '')}"
                    ),
                )
                extras += 1
            if extras < _MAX_LEGIBILITY_EXTRA_ROWS:
                self._enqueue_legibility_job(
                    "deduction",
                    prompt=(
                        "Write one short row body for 'Deduction:'. "
                        "Name the concrete thing that cost credit. "
                        f"Question: {item.question_id}. "
                        f"Score basis: {getattr(prediction, 'score_basis', '')} "
                        f"Reasoning: {getattr(prediction, 'model_reasoning', '')}"
                    ),
                )
                extras += 1

    def _produce_after_action(
        self,
        elapsed: float,
        prediction: Any,
        item: Any,
        template_question: dict | None,
    ) -> None:
        """Produce a compact per-problem outcome block."""
        try:
            if prediction is None or item is None:
                self._sink.write_topic(f"{elapsed:.0f}s elapsed")
                return

            if getattr(prediction, "truncated", False):
                self._sink.write_topic(
                    f"{elapsed:.0f}s · grader did not commit to a score (truncated)"
                )
                return

            truth_score = getattr(item, "truth_score", item.professor_score)
            corrected_truth = (
                getattr(item, "corrected_score", None) is not None
                and abs(truth_score - item.professor_score) > 1e-9
            )
            band_classification = _classify_score_against_band(
                prediction.model_score,
                item,
            )
            verdict_short = band_classification.verdict_short
            text = _after_action_topic_text(
                elapsed=elapsed,
                prediction=prediction,
                item=item,
                truth_score=truth_score,
                corrected_truth=corrected_truth,
                band_classification=band_classification,
            )
            logger.info("After-action: %s", text.replace("\n", " | "))
            self._sink.write_topic(
                text,
                verdict=verdict_short,
                grader_score=prediction.model_score,
                truth_score=truth_score,
                max_points=item.max_points,
                acceptable_score_floor=(
                    band_classification.floor
                    if band_classification.band_present
                    else None
                ),
                acceptable_score_ceiling=(
                    band_classification.ceiling
                    if band_classification.band_present
                    else None
                ),
            )
            self._handle_legibility_rows(prediction, item)
            if self._should_enqueue_dossier(elapsed, prediction, item):
                self._enqueue_dossier_job(
                    self._build_dossier_prompt(
                        elapsed=elapsed,
                        prediction=prediction,
                        item=item,
                        template_question=template_question,
                    )
                )
            self._schedule_idle_legibility_if_needed()
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
            self._item_token_count += max(1, _rough_token_count(token))
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
                status_established = bool(prior_statuses)
                logger.info(
                    "Narrator: first-person summary was repetitive, retrying in status mode: %s",
                    full,
                )
                (
                    status_full,
                    _status_user_content,
                    status_drop_reason,
                    status_drop,
                ) = self._retry_duplicate_as_status(
                    chunk, full, prior_statuses
                )
                if not status_full:
                    if status_established:
                        with self._stats_lock:
                            self._stat_drops_dedup += 1
                        self._sink.write_drop("dedup", full)
                    if status_drop_reason and status_drop:
                        self._sink.write_drop(status_drop_reason, status_drop)
                    logger.info("Narrator: dropped repetitive summary: %s", full)
                    with self._lock:
                        self._dedupe_streak += 1
                        if self._dedupe_streak > self._item_max_dedupe_streak:
                            self._item_max_dedupe_streak = self._dedupe_streak
                    self._maybe_groom_after_repeated_dedup(
                        chunk,
                        prior_thoughts,
                        prior_statuses,
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
                self._dedupe_streak = 0
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
                self._emit_checkpoint_from_context(
                    checkpoint_chunk,
                    checkpoint_line,
                    checkpoint_thoughts,
                    checkpoint_statuses,
                    checkpoint_prior,
                )
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

    @staticmethod
    def _checkpoint_lines_share_basin(a: str, b: str) -> bool:
        label_a, body_a = ThinkingNarrator._split_checkpoint_label(a)
        label_b, body_b = ThinkingNarrator._split_checkpoint_label(b)
        if label_a != label_b:
            return False
        norm_a = ThinkingNarrator._normalize_checkpoint_body(body_a)
        norm_b = ThinkingNarrator._normalize_checkpoint_body(body_b)
        if not norm_a or not norm_b:
            return False
        if norm_a == norm_b:
            return True
        return ThinkingNarrator._lines_too_similar(
            norm_a,
            norm_b,
            threshold=0.70,
        )

    @staticmethod
    def _split_checkpoint_label(text: str) -> tuple[str, str]:
        match = re.match(
            r"^(?:\*+|`+)?\s*(Core issue|Evidence|Context)\s*:\s*(?:\*+|`+)?\s*(.*)$",
            text.strip(),
            re.IGNORECASE,
        )
        if not match:
            return "", text.strip()
        return match.group(1).lower(), match.group(2).strip()

    @staticmethod
    def _canonicalize_checkpoint_text(text: str) -> str:
        label, body = ThinkingNarrator._split_checkpoint_label(text)
        if label not in {"core issue", "evidence", "context"} or not body:
            return text.strip()
        canonical_labels = {
            "core issue": "Core issue",
            "evidence": "Evidence",
            "context": "Context",
        }
        return f"{canonical_labels[label]}: {body}"

    @staticmethod
    def _normalize_checkpoint_body(text: str) -> str:
        normalized = text.lower()
        replacements = {
            "cm^3": "volume_unit",
            "cm³": "volume_unit",
            "cm3": "volume_unit",
            "ml": "volume_unit",
            "moles-versus-grams": "unit_mixup",
            "moles vs grams": "unit_mixup",
            "moles and grams": "unit_mixup",
            "net ionic": "net_ionic",
            "valence electrons": "valence_electrons",
            "electron count": "valence_electrons",
            "octet rule": "octet",
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        normalized = re.sub(r"[^a-z0-9_/ -]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.002,
        repetition_penalty: float = 1.001,
        presence_penalty: float = 0.0,
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
        chat_template_kwargs = _qwen36_chat_template_kwargs(eff_model)
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
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
        temperature: float = 0.6,
        max_tokens: int = _MAX_TOKENS,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.002,
        repetition_penalty: float = 1.001,
        presence_penalty: float = 0.0,
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
        chat_template_kwargs = _qwen36_chat_template_kwargs(self._model)
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
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
