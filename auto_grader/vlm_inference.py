"""VLM inference wrapper for grading exam scans.

Thin layer between the eval harness and an OpenAI-compatible VLM server.
Sends scan page images + rubric context, parses structured scoring responses
into Prediction objects that the eval harness can score.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import yaml

from auto_grader.eval_harness import EvalItem, Prediction


@dataclass
class ServerConfig:
    """Connection config for an OpenAI-compatible VLM server.

    Sampling defaults follow Alibaba's official recommendations for
    Qwen3.5 *thinking-mode coding* tasks (the "precise coding tasks
    e.g. WebDev" row of the model card sampling table):

        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0

    Why coding-mode and not general-mode: our task is thinking-mode
    reasoning with **structured JSON output**, which is structurally
    closer to coding than to free-form prose. The thinking-mode
    *general* row recommends presence_penalty=1.5, but JSON output is
    highly repetitive (`{`, `"`, `}`, field names) and presence
    penalty actively fights against valid JSON emission. Confirmed
    empirically on 2026-04-08 16:43: with presence_penalty=1.5 the
    model emitted ~160 coherent reasoning_content tokens (squinting
    at handwriting, doing the actual division 95/13.6, considering
    misprints) and then the stream closed without ever producing a
    content delta. Reasoning prose was fine, JSON emission was
    blocked. Switching to the coding preset with a low, bounded
    presence penalty
    fixes the structured-output phase without giving up the
    exploration that the original temperature=0.1 was missing.

    History of regimes tried:

      1. temperature=0.1, presence_penalty=0 (pre-2026-04-08): caused
         repetitive 200+ second reasoning loops on items like fr-5b.
         Low temperature collapsed exploration, no anti-repetition
         gradient meant the model could chew on the same loop forever.
      2. temperature=1.0, presence_penalty=1.5 (Alibaba thinking-mode
         general, tried 2026-04-08 16:43): reasoning prose was clean
         and grounded, but JSON output was blocked entirely. Empty
         content, parser fail on every item.
      3. temperature=0.6, presence_penalty=0.75 (temporary bounded
         experiment on top of Alibaba thinking-mode coding): kept the
         coding preset structure, but added a moderate novelty gradient
         to see whether Qwen escaped repetitive local loops without
         collapsing JSON emission the way 1.5 did.
      4. temperature=0.6, presence_penalty=0.0 (current): revert to the
         coding preset after the bounded experiment. Structured JSON
         emission remains the primary constraint, so we prefer the
         simpler known-good zero-presence regime until there is clear
         evidence the higher value helps more than it harms.

    We still do NOT jump to the general-mode 1.5 presence penalty for
    structured output.

    max_tokens=8192 keeps headroom for legitimately hard items without
    letting one bad stall chew through the old 16384-token runway. The
    longest useful traces we have seen were already in the ~7-8K token
    band, so 8192 is still generous, but no longer gives pathological
    items an enormous extra burn window.
    """

    base_url: str  # e.g. "http://192.168.68.128:8001"
    api_key: str = "1234"
    model: str = "qwen3p5-35B-A3B"
    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------


#: DPI used for page images handed to the VLM grader. Kept modest so
#: request payloads (base64-encoded PNGs) stay well under typical
#: endpoint limits and rasterization latency stays tolerable per item.
GRADER_PAGE_DPI = 200

#: DPI used for page images handed to the focus-preview pipeline. The
#: primary preview renderer on image-capable terminals (WezTerm, iTerm2)
#: is the iTerm2 inline image path, which hands the full-resolution PNG
#: directly to the terminal for pixel-for-pixel rasterization. That
#: path benefits from maximum source resolution — the terminal will
#: produce a sharper rendered image the more pixels it has to work with.
#: The half-block fallback renderer (for terminals without image
#: support) downsamples to terminal cells regardless of source DPI and
#: doesn't benefit from the extra pixels, but doesn't suffer either.
#: The grader does NOT see this higher-resolution image — it's
#: rasterized into a separate cache at GRADER_PAGE_DPI.
PREVIEW_PAGE_DPI = 800


def extract_page_image(pdf_path: Path, page_num: int, dpi: int = GRADER_PAGE_DPI) -> bytes:
    """Extract a single page from a PDF as a PNG image (bytes).

    page_num is 1-indexed (matching ground truth schema).
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _image_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Grading philosophy, in project terms:
#
# - The grader's job is to award the highest score justified by the work
#   actually on the page, not merely to mark errors.
# - "Generous" here means lawful rescue, not speculative charity: actively
#   search for rubric-grounded setup credit, method credit, carry-forward
#   credit, and Lewis-structure partials, but do not imagine missing work.
# - Answered-form requirements still bound rescue. If the requested form is the
#   thing being graded, nearby chemical ingredients do not magically become the
#   requested answer.
# - Ambiguity should not turn into endless overthinking. After one careful pass,
#   choose the best-supported reading, record uncertainty in model_reasoning,
#   lower confidence when warranted, and stop.
#
# Keep this explanation next to the system prompt so future prompt edits do not
# quietly drift back toward vague "be charitable" language.
_SYSTEM_PROMPT = """\
You are grading a chemistry exam.
Award the highest score justified by the student's written work under the rubric.
Actively rescue as much lawful partial credit as possible.
If the student's work supports a lawful full-credit interpretation, take it and stop.
Be charitable toward handwriting and notation: if a student's marks admit a reasonable reading as correct, read them that way.
Be strict toward errors you see. An error you notice is an error you grade, even if the student "demonstrated the core concept" — that is abandoning the rubric, not charity.
Use is_obviously_fully_correct = true only for clearly correct answers needing no human rescue.
Use is_obviously_wrong = true only for clearly wrong answers with no lawful rescue path.
Do not use is_obviously_wrong = true if any lawful partial-credit path remains.
Treat mL and cm³ as equivalent unless the question explicitly tests form.
If the student shows correct method but makes an arithmetic slip, award partial credit for the method.
If setup is chemically correct and the only miss is small arithmetic, truncation, or rounding, award full credit unless exact rounding or significant figures are being tested.
Right relation but later execution or unit miss: preserve nonzero setup credit unless the setup itself is wrong.
Wrong-concept vs wrong-execution: preserve method credit for right approach with bad arithmetic or units, but not for a wrong approach that only shares surface symbols with the right one.
If the student's approach would still be wrong with perfect execution, do not award method credit.
Internal consistency: if this part depends on an earlier wrong answer but the student applies their own earlier result correctly here, award full credit for the method in this part.
On Lewis-structure questions, rescue partial credit for correct connectivity, valence electrons, or bond order even if octets, formal charges, or resonance are incomplete.
Do not collapse a Lewis-structure answer to zero when connectivity or the valence-electron basis is clearly right and only bonding or octet completion is wrong.
Grade what is written, not a more favorable answer you can imagine.
If two readings are plausible and neither is clearly better supported, choose the best-supported reading and move on.
After one careful pass, if ambiguity still affects the score, choose the best-supported reading, say in model_reasoning that human review is warranted, lower model_confidence, and stop.
score_basis = short literal basis for the awarded score: credit earned vs lost.
model_reasoning = broader reasoning only: ambiguity, OCR, rescue logic, or review handoff.
Do not restate score_basis in model_reasoning.
Answered-form rule: grade the form the question asked for. A net ionic equation means net ionic only; molecular and full ionic equations answer a different question.
When the requested form is the thing being graded, do not award rescue credit for nearby ingredients unless the rubric explicitly does so.
If the requested answer form is plainly missing, stop and score only what is on the page.
Use upstream_dependency = "none" unless carry-forward is clear.
Respond with only the JSON object below. upstream_dependency and if_dependent_then_consistent are required fields and must be populated before model_score:
{
  "model_read": "<what the student wrote, verbatim>",
  "model_score": <number>,
  "model_confidence": <0 to 1>,
  "model_reasoning": "<brief justification>",
  "upstream_dependency": "<earlier part id or 'none'>",
  "if_dependent_then_consistent": <true | false | null>,
  "score_basis": <string>,
  "is_obviously_fully_correct": <true | false | null>,
  "is_obviously_wrong": <true | false | null>
}
"""

GRADING_PROMPT_VERSION = "2026-04-11-positive-sweep-v1"

def _build_grading_prompt(item: EvalItem, template_question: dict | None) -> str:
    """Build the user-facing grading prompt for one question."""
    parts = [
        f"Grade question {item.question_id} on this page.",
        f"Answer type: {item.answer_type}",
        f"Maximum points: {item.max_points}",
    ]

    if template_question:
        if "prompt" in template_question:
            parts.append(f"Question text: {template_question['prompt']}")
        if "correct" in template_question:
            correct = template_question["correct"]
            if isinstance(correct, dict):
                if "value" in correct:
                    parts.append(f"Correct answer: {correct['value']}")
                if "expression" in correct:
                    parts.append(f"Answer expression: {correct['expression']}")
                if "accept" in correct:
                    parts.append(f"Also accept: {correct['accept']}")
            else:
                parts.append(f"Correct answer: {correct}")
        if "rubric" in template_question.get("correct", {}):
            rubric = template_question["correct"]["rubric"]
            rubric_text = "; ".join(
                f"{r.get('criterion', '?')} ({r.get('points', '?')} pts)"
                for r in rubric
            )
            parts.append(f"Rubric: {rubric_text}")

    parts.append(
        "\nRespond with ONLY the JSON object, no markdown fences or other text."
    )
    return "\n".join(parts)


def grading_prompt_metadata() -> dict[str, str]:
    """Return operator-legible prompt identity for run manifests."""
    content_hash = hashlib.sha256(
        (
            _SYSTEM_PROMPT
            + "\n---build_grading_prompt---\n"
            + inspect.getsource(_build_grading_prompt)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": GRADING_PROMPT_VERSION,
        "content_hash": content_hash,
    }


# ---------------------------------------------------------------------------
# Template question lookup
# ---------------------------------------------------------------------------


def load_template_questions(template_path: Path) -> dict[str, dict]:
    """Load template and return a dict mapping question_id -> question dict."""
    with open(template_path) as f:
        template = yaml.safe_load(f)

    questions: dict[str, dict] = {}
    for section in template.get("sections", []):
        for q in section.get("questions", []):
            qid = q.get("id", "")
            questions[qid] = q
            for part in q.get("parts", []):
                pid = part.get("id", "")
                if pid:
                    questions[pid] = part
    return questions


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _parse_vlm_response(text: str) -> dict[str, Any]:
    """Parse the VLM's JSON response, tolerating markdown fences, thinking,
    and various formatting quirks from different models."""
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try to find a JSON object with nested content (model_reasoning may have quotes)
    # Use a greedy match from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try fixing common issues: single quotes -> double quotes
        fixed = candidate.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Last resort: extract fields with regex
    score_m = re.search(r'"?model_score"?\s*:\s*([\d.]+)', text)
    conf_m = re.search(r'"?model_confidence"?\s*:\s*([\d.]+)', text)
    read_m = re.search(r'"?model_read"?\s*:\s*"([^"]*)"', text)
    score_basis_m = re.search(r'"?score_basis"?\s*:\s*"([^"]*)"', text)
    reason_m = re.search(r'"?model_reasoning"?\s*:\s*"([^"]*)"', text)
    obvious_full_m = re.search(
        r'"?is_obviously_fully_correct"?\s*:\s*(true|false|null)',
        text,
        flags=re.IGNORECASE,
    )
    obvious_wrong_m = re.search(
        r'"?is_obviously_wrong"?\s*:\s*(true|false|null)',
        text,
        flags=re.IGNORECASE,
    )

    if score_m:
        def _boolish(match):
            if not match:
                return None
            value = match.group(1).lower()
            if value == "true":
                return True
            if value == "false":
                return False
            return None

        return {
            "model_score": float(score_m.group(1)),
            "model_confidence": float(conf_m.group(1)) if conf_m else 0.5,
            "model_read": read_m.group(1) if read_m else "",
            "score_basis": score_basis_m.group(1) if score_basis_m else "",
            "model_reasoning": reason_m.group(1) if reason_m else "",
            "is_obviously_fully_correct": _boolish(obvious_full_m),
            "is_obviously_wrong": _boolish(obvious_wrong_m),
        }

    raise ValueError(f"Could not parse VLM response as JSON: {text[:300]}")


def _failure_prediction(
    item: EvalItem,
    *,
    message: str,
    raw_assistant: str,
    raw_reasoning: str,
) -> Prediction:
    """Return a structured grader failure as a Prediction.

    Runs should degrade on one bad item rather than crashing the whole
    batch. We encode the failure as a zero-confidence, zero-score
    prediction and preserve the raw payloads for later inspection.
    """
    return Prediction(
        exam_id=item.exam_id,
        question_id=item.question_id,
        model_score=0.0,
        model_confidence=0.0,
        score_basis="",
        model_reasoning=message,
        model_read="",
        raw_assistant=raw_assistant,
        raw_reasoning=raw_reasoning,
        is_obviously_fully_correct=None,
        is_obviously_wrong=None,
        upstream_dependency="none",
        if_dependent_then_consistent=None,
    )


def grade_single_item(
    item: EvalItem,
    page_image: bytes,
    config: ServerConfig,
    template_question: dict | None = None,
    on_reasoning_delta: Any = None,
) -> Prediction:
    """Send one item to the VLM and return a Prediction.

    Always uses streaming mode so the SSE iterator is interruptible
    by SIGINT (Ctrl-C). When on_reasoning_delta is provided, reasoning
    tokens are pumped through it for the live narrator; otherwise the
    stream is consumed silently and we just need the final content.
    Closing the response mid-stream tells OMLX to abort the inference.
    """
    import urllib.request

    prompt_text = _build_grading_prompt(item, template_question)
    image_url = _image_to_data_url(page_image)

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "presence_penalty": config.presence_penalty,
        "repetition_penalty": config.repetition_penalty,
        "stream": True,
    }

    body = json.dumps(payload).encode()

    def _build_request():
        return urllib.request.Request(
            f"{config.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
        )

    last_err = None
    content = ""
    reasoning = ""
    finish_reason: str | None = None
    for attempt in range(3):
        try:
            req = _build_request()
            resp = urllib.request.urlopen(req, timeout=600)
            try:
                content, reasoning, finish_reason = _consume_streaming_response(
                    resp, on_reasoning_delta
                )
            except KeyboardInterrupt:
                # User hit Ctrl-C — close the socket so OMLX cancels
                # the in-flight inference, then re-raise.
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
            break
        except KeyboardInterrupt:
            raise
        except (TimeoutError, OSError) as e:
            last_err = e
            if attempt < 2:
                import time
                time.sleep(2)
    else:
        raise TimeoutError(
            f"VLM request failed after 3 attempts for "
            f"{item.exam_id}/{item.question_id}: {last_err}"
        )

    try:
        parsed = _parse_vlm_response(content)
    except ValueError:
        if finish_reason == "length":
            message = (
                "Grader output was truncated at the max token limit before it "
                "finished the required JSON."
            )
        else:
            message = (
                "Grader output could not be parsed as the required JSON."
            )
        return _failure_prediction(
            item,
            message=message,
            raw_assistant=content,
            raw_reasoning=reasoning,
        )

    # Coerce upstream-dependency fields tolerantly. Older grader models or
    # gemma-4 may not honor the new schema; default to "none" / null so
    # downstream code can detect "grader did not declare" vs "grader said
    # no dependency" by checking for the literal "none" sentinel.
    raw_dep = parsed.get("upstream_dependency", "none")
    upstream_dependency = (
        str(raw_dep).strip() if raw_dep is not None else "none"
    ) or "none"
    raw_consistent = parsed.get("if_dependent_then_consistent", None)
    if isinstance(raw_consistent, bool):
        if_dependent_then_consistent: bool | None = raw_consistent
    elif isinstance(raw_consistent, str):
        # Models sometimes emit "true"/"false"/"null" as strings.
        s = raw_consistent.strip().lower()
        if s == "true":
            if_dependent_then_consistent = True
        elif s == "false":
            if_dependent_then_consistent = False
        else:
            if_dependent_then_consistent = None
    else:
        if_dependent_then_consistent = None

    def _coerce_optional_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "true":
                return True
            if s == "false":
                return False
        return None

    is_obviously_fully_correct = _coerce_optional_bool(
        parsed.get("is_obviously_fully_correct")
    )
    is_obviously_wrong = _coerce_optional_bool(
        parsed.get("is_obviously_wrong")
    )
    if is_obviously_fully_correct and is_obviously_wrong:
        is_obviously_fully_correct = None
        is_obviously_wrong = None

    return Prediction(
        exam_id=item.exam_id,
        question_id=item.question_id,
        model_score=float(parsed.get("model_score", 0)),
        model_confidence=float(parsed.get("model_confidence", 0.5)),
        score_basis=str(parsed.get("score_basis", "")),
        model_reasoning=str(parsed.get("model_reasoning", "")),
        model_read=str(parsed.get("model_read", "")),
        raw_assistant=content,
        raw_reasoning=reasoning,
        is_obviously_fully_correct=is_obviously_fully_correct,
        is_obviously_wrong=is_obviously_wrong,
        upstream_dependency=upstream_dependency,
        if_dependent_then_consistent=if_dependent_then_consistent,
    )


def _consume_streaming_response(
    resp, on_reasoning_delta
) -> tuple[str, str, str | None]:
    """Read SSE chunks from the VLM stream. Pumps reasoning_content deltas
    through the callback as they arrive (when provided); returns
    (assistant_content, reasoning_content) for parsing AND for the
    post-hoc critic pass.

    The reasoning trace is the verbatim <think> span — much longer than
    the curated model_reasoning field in the parsed JSON, and the only
    place where consistency-rule violations are observable.

    SIGINT propagates through the iterator naturally — Python checks for
    signals between chunk reads, so Ctrl-C interrupts within ~one chunk.

    Accumulates content and reasoning into lists and joins at the end
    rather than `s += chunk` in the loop. CPython has an in-place
    optimization for the single-reference case that often makes `+=`
    amortized O(N), but it is not guaranteed and silently degrades to
    O(N^2) under conditions that are easy to trip (e.g. another
    reference taken transiently for logging). For 200+ second
    reasoning streams with thousands of deltas this matters.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    fallback_scan = ""
    waiting_for_reasoning_colon = False
    waiting_for_reasoning_quote = False
    capturing_reasoning_string = False
    pending_escape = False
    pending_unicode_digits = 0
    saw_reasoning_channel = False

    def _extract_fallback_reasoning(content_delta: str) -> str:
        nonlocal fallback_scan
        nonlocal waiting_for_reasoning_colon
        nonlocal waiting_for_reasoning_quote
        nonlocal capturing_reasoning_string
        nonlocal pending_escape
        nonlocal pending_unicode_digits

        if not content_delta:
            return ""

        extracted: list[str] = []
        for ch in content_delta:
            if capturing_reasoning_string:
                if pending_unicode_digits > 0:
                    pending_unicode_digits -= 1
                    continue
                if pending_escape:
                    if ch in {"n", "r", "t"}:
                        extracted.append(" ")
                    elif ch == "u":
                        pending_unicode_digits = 4
                    else:
                        extracted.append(ch)
                    pending_escape = False
                    continue
                if ch == "\\":
                    pending_escape = True
                    continue
                if ch == '"':
                    capturing_reasoning_string = False
                    continue
                extracted.append(ch)
                continue

            fallback_scan = (fallback_scan + ch)[-64:]
            if waiting_for_reasoning_colon:
                if ch == ":":
                    waiting_for_reasoning_colon = False
                    waiting_for_reasoning_quote = True
                continue
            if waiting_for_reasoning_quote:
                if ch == '"':
                    waiting_for_reasoning_quote = False
                    capturing_reasoning_string = True
                elif ch in " \t\r\n":
                    continue
                else:
                    waiting_for_reasoning_quote = False
                continue
            if fallback_scan.endswith('"model_reasoning"'):
                waiting_for_reasoning_colon = True

        return "".join(extracted)

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
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice.get("finish_reason"))
        # Reasoning tokens — accumulate for the critic, and pump to
        # narrator if wired.
        rc_delta = delta.get("reasoning_content", "")
        if rc_delta:
            if rc_delta.strip():
                saw_reasoning_channel = True
                reasoning_parts.append(rc_delta)
            elif not reasoning_parts:
                # Ignore whitespace-only pseudo-deltas. Some streams emit
                # a bare newline on the reasoning channel even though the
                # actual reasoning only arrives inside the final JSON
                # model_reasoning field. Treat that as "no real reasoning
                # channel yet" so the fallback parser can still engage.
                rc_delta = ""
        if rc_delta and rc_delta.strip():
            if on_reasoning_delta is not None:
                try:
                    on_reasoning_delta(rc_delta)
                except Exception:
                    pass
        # Final assistant content — accumulate for parsing
        c_delta = delta.get("content", "")
        if c_delta:
            content_parts.append(c_delta)
            if not saw_reasoning_channel:
                fallback_reasoning = _extract_fallback_reasoning(c_delta)
                if fallback_reasoning:
                    reasoning_parts.append(fallback_reasoning)
                    if on_reasoning_delta is not None:
                        try:
                            on_reasoning_delta(fallback_reasoning)
                        except Exception:
                            pass
    return "".join(content_parts), "".join(reasoning_parts), finish_reason


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

_EXAM_PDF_MAP = {
    "15-blue": "15 blue.pdf",
    "27-blue-2023": "27 blue 2023.pdf",
    "34-blue": "34 blue.pdf",
    "39-blue": "39 blue_Redacted 1.pdf",
    # 39-blue-redacted is the FERPA-name-redacted PDF with grading marks
    # ALSO removed by hand. This is currently the only contamination-free
    # exam in the eval set; everything else still has the prof's
    # checkmarks and margin scores leaking signal into the grader. Use
    # this exam_id for any experiment whose accuracy claim is meant to
    # be trusted. See the contamination spike on 2026-04-08 for context.
    "39-blue-redacted": "39 blue_Redacted_grading_marks_removed.pdf",
}


def _report_focus_preview_failure(sink: Any, item: EvalItem, exc: Exception) -> None:
    """Surface preview-pipeline failures without killing the grading run.

    The focus preview is an operator affordance, not a reason to abort scoring.
    But failures in that pipeline must not vanish silently, because the human
    is actively evaluating the surface live in Paint Dry.
    """
    msg = (
        f"{item.exam_id}/{item.question_id} · "
        f"focus preview unavailable ({type(exc).__name__})"
    )
    try:
        sink.write_drop("focus-preview-error", msg)
    except Exception:
        print(f"[focus-preview-error] {msg}", file=sys.stderr)


def grade_all_items(
    ground_truth: list[EvalItem],
    scans_dir: Path,
    config: ServerConfig,
    template_path: Path | None = None,
    progress_callback: Any = None,
    narrator: Any = None,
    sink: Any = None,
    focus_preview_callback: Any = None,
) -> list[Prediction]:
    """Grade all ground truth items against VLM, returning predictions.

    Caches page images to avoid re-extracting the same page for multiple
    questions on that page.

    If a ThinkingNarrator + NarratorSink are provided, the VLM call switches
    to streaming mode and reasoning_content tokens are pumped through the
    narrator for live play-by-play. The sink also gets per-item header and
    topic events.
    """
    template_questions = (
        load_template_questions(template_path) if template_path else {}
    )

    # Two separate caches at two separate resolutions. The grader cache
    # feeds the VLM at GRADER_PAGE_DPI (payload-friendly). The preview
    # cache feeds the focus-preview terminal renderer at PREVIEW_PAGE_DPI
    # (high enough to survive the half-block Nyquist ceiling). Held one
    # page at a time per exam, so memory overhead is bounded.
    page_cache: dict[tuple[str, int], bytes] = {}
    preview_page_cache: dict[tuple[str, int], bytes] = {}
    predictions: list[Prediction] = []

    for i, item in enumerate(ground_truth):
        cache_key = (item.exam_id, item.page)
        if cache_key not in page_cache:
            pdf_name = _EXAM_PDF_MAP.get(item.exam_id)
            if not pdf_name:
                raise ValueError(f"No PDF mapping for exam_id: {item.exam_id}")
            pdf_path = scans_dir / pdf_name
            if not pdf_path.exists():
                raise FileNotFoundError(f"Scan PDF not found: {pdf_path}")
            page_cache[cache_key] = extract_page_image(
                pdf_path, item.page, dpi=GRADER_PAGE_DPI
            )
            preview_page_cache[cache_key] = extract_page_image(
                pdf_path, item.page, dpi=PREVIEW_PAGE_DPI
            )

        tq = template_questions.get(item.question_id)

        # Per-item narrator lifecycle
        if narrator is not None and sink is not None:
            header = (
                f"[item {i + 1}/{len(ground_truth)}] "
                f"{item.exam_id}/{item.question_id} "
                f"({item.answer_type}, {item.max_points} pts)"
            )
            # Build a richer context for the narrator: actual question prompt
            # from the template + the student's answer + the professor's
            # ground-truth score. Bonsai needs grounding or it hallucinates
            # chemistry from the system-prompt examples.
            narrator_context_parts = [header]
            if tq:
                if "prompt" in tq:
                    qp = str(tq["prompt"]).strip().replace("\n", " ")
                    narrator_context_parts.append(f"Question prompt: {qp[:200]}")
                if "correct" in tq:
                    correct = tq["correct"]
                    if isinstance(correct, dict) and "value" in correct:
                        narrator_context_parts.append(
                            f"Expected answer: {correct['value']}"
                        )
                    elif not isinstance(correct, dict):
                        narrator_context_parts.append(
                            f"Expected answer: {correct}"
                        )
            narrator_context_parts.append(
                f"Student wrote: \"{item.student_answer}\""
            )
            narrator_context_parts.append(
                f"Professor scored: {item.professor_score}/{item.max_points} "
                f"(mark: {item.professor_mark})"
            )
            narrator_context = "\n".join(narrator_context_parts)
            sink.write_header(header)
            if focus_preview_callback is not None:
                try:
                    focus_preview_callback(
                        item=item,
                        page_image=preview_page_cache[cache_key],
                        template_question=tq,
                    )
                except Exception as exc:
                    _report_focus_preview_failure(sink, item, exc)
            narrator.start(item_header=narrator_context)
            on_delta = narrator.feed
        else:
            on_delta = None

        pred = grade_single_item(
            item, page_cache[cache_key], config, tq,
            on_reasoning_delta=on_delta,
        )
        predictions.append(pred)

        if narrator is not None:
            narrator.stop_and_summarize(
                prediction=pred,
                item=item,
                template_question=tq,
            )

        if progress_callback:
            progress_callback(i + 1, len(ground_truth), item, pred)

    return predictions
