"""VLM inference wrapper for grading exam scans.

Thin layer between the eval harness and an OpenAI-compatible VLM server.
Sends scan page images + rubric context, parses structured scoring responses
into Prediction objects that the eval harness can score.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import yaml

from auto_grader.eval_harness import EvalItem, Prediction


@dataclass
class ServerConfig:
    """Connection config for an OpenAI-compatible VLM server."""

    base_url: str  # e.g. "http://192.168.68.128:8001"
    api_key: str = "1234"
    model: str = "qwen3p5-35B-A3B"
    max_tokens: int = 2048
    temperature: float = 0.1


# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------


def extract_page_image(pdf_path: Path, page_num: int, dpi: int = 200) -> bytes:
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

_SYSTEM_PROMPT = """\
You are grading a chemistry exam. You will be shown a scanned page from a \
student's exam and asked to grade a specific question.

Grading philosophy: BE CHARITABLE. Give the student the benefit of the doubt.
- If the student's work is internally consistent but builds on a wrong answer \
from a previous part, award full credit for the current part.
- If the student shows correct methodology but makes an arithmetic error, \
award partial credit for the method.
- If the answer is ambiguous but a reasonable reading supports correctness, \
give credit.
- When in doubt, find a way to award credit within the rubric rather \
than withholding it.

For each question, you must:
1. Read what the student wrote
2. Compare it to the correct answer / rubric
3. Award a score, erring on the side of generosity

Respond in EXACTLY this JSON format (no other text):
{
  "model_read": "<what the student wrote, verbatim>",
  "model_score": <numeric score you award>,
  "model_confidence": <0.0 to 1.0, your confidence in the score>,
  "model_reasoning": "<brief explanation of your grading>"
}
"""


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
    reason_m = re.search(r'"?model_reasoning"?\s*:\s*"([^"]*)"', text)

    if score_m:
        return {
            "model_score": float(score_m.group(1)),
            "model_confidence": float(conf_m.group(1)) if conf_m else 0.5,
            "model_read": read_m.group(1) if read_m else "",
            "model_reasoning": reason_m.group(1) if reason_m else "",
        }

    raise ValueError(f"Could not parse VLM response as JSON: {text[:300]}")


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
    for attempt in range(3):
        try:
            req = _build_request()
            resp = urllib.request.urlopen(req, timeout=600)
            try:
                content = _consume_streaming_response(
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

    parsed = _parse_vlm_response(content)

    return Prediction(
        exam_id=item.exam_id,
        question_id=item.question_id,
        model_score=float(parsed.get("model_score", 0)),
        model_confidence=float(parsed.get("model_confidence", 0.5)),
        model_reasoning=str(parsed.get("model_reasoning", "")),
        model_read=str(parsed.get("model_read", "")),
    )


def _consume_streaming_response(resp, on_reasoning_delta) -> str:
    """Read SSE chunks from the VLM stream. Pumps reasoning_content deltas
    through the callback as they arrive (when provided); returns the
    assistant content (the JSON response) for parsing.

    SIGINT propagates through the iterator naturally — Python checks for
    signals between chunk reads, so Ctrl-C interrupts within ~one chunk."""
    content_full = ""
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
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        # Reasoning tokens — pump to narrator if wired
        rc_delta = delta.get("reasoning_content", "")
        if rc_delta and on_reasoning_delta is not None:
            try:
                on_reasoning_delta(rc_delta)
            except Exception:
                pass
        # Final assistant content — accumulate for parsing
        c_delta = delta.get("content", "")
        if c_delta:
            content_full += c_delta
    return content_full


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

_EXAM_PDF_MAP = {
    "15-blue": "15 blue.pdf",
    "27-blue-2023": "27 blue 2023.pdf",
    "34-blue": "34 blue.pdf",
    "39-blue": "39 blue_Redacted 1.pdf",
}


def grade_all_items(
    ground_truth: list[EvalItem],
    scans_dir: Path,
    config: ServerConfig,
    template_path: Path | None = None,
    progress_callback: Any = None,
    narrator: Any = None,
    sink: Any = None,
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

    page_cache: dict[tuple[str, int], bytes] = {}
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
            page_cache[cache_key] = extract_page_image(pdf_path, item.page)

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
            narrator.stop_and_summarize()

        if progress_callback:
            progress_callback(i + 1, len(ground_truth), item, pred)

    return predictions
