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
    blocked. Switching to the coding preset with presence_penalty=0
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
      3. temperature=0.6, presence_penalty=0 (current — Alibaba
         thinking-mode coding): the right combination for our task
         shape. 6x more exploration than (1) so reasoning loops
         should dissolve, no presence penalty fighting JSON emission.

    If (3) still produces repetitive reasoning, the next move is
    raising temperature toward 0.8-1.0 while keeping presence_penalty
    at 0. We do NOT raise presence_penalty for structured output.

    max_tokens=16384 is intentionally generous. Picking an intermediate
    value (e.g. 4096) is the worst of all worlds: too low to reliably
    accommodate the longest legitimate reasoning we have observed (fr-5b
    was ~30K chars ≈ 7-8K tokens of reasoning_content), too high to
    "fail fast" on a runaway loop. The cost of setting max_tokens high
    is asymmetric — it is only paid when the model actually uses the
    budget — so a high ceiling catches the long-reasoning case without
    penalizing short-reasoning items. 16384 gives comfortable headroom
    above the worst observed legitimate reasoning while still being a
    finite safety net for true infinite loops. The optimization target
    is "every item gets a definitive answer"; per-item wall clock up
    to 3-4 minutes on the hardest items is acceptable for a 12-item
    curated test set.
    """

    base_url: str  # e.g. "http://192.168.68.128:8001"
    api_key: str = "1234"
    model: str = "qwen3p5-35B-A3B"
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


# ---------------------------------------------------------------------------
# Per-model-family sampling presets
# ---------------------------------------------------------------------------

_MODEL_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "qwen": (
        "qwen/",
        "qwen3.5",
        "qwen3-",
        "qwen3p5-",
    ),
    "gemma-4": (
        "gemma-4-",
        "google/gemma-4-",
    ),
}

_TASK_SAMPLING_PRESETS: dict[str, dict[str, dict[str, float | int]]] = {
    "grading": {
        "neutral": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "qwen": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "gemma-4": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    },
    "describe": {
        "neutral": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "qwen": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "gemma-4": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    },
}


def known_model_families() -> tuple[str, ...]:
    return ("qwen", "gemma-4", "neutral")


def resolve_model_family(model: str, requested_family: str | None = None) -> str:
    """Resolve a model into a registered family or raise loudly."""
    families = known_model_families()
    if requested_family is not None:
        family = requested_family.strip().lower()
        if family not in families:
            valid = ", ".join(families)
            raise ValueError(
                f"Unknown model-family '{requested_family}'. Valid values: {valid}."
            )
        return family

    normalized = model.strip().lower()
    for family, prefixes in _MODEL_FAMILY_PATTERNS.items():
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return family

    valid = ", ".join(families)
    raise ValueError(
        f"Unregistered model '{model}'. Pass --model-family "
        f"{{{valid}}} to choose explicit sampling defaults."
    )


def apply_model_sampling_preset(
    config: ServerConfig,
    model: str | None = None,
    *,
    family: str | None = None,
    task: str = "grading",
) -> ServerConfig:
    """Return a new ServerConfig with explicit family-based sampling."""
    import dataclasses

    if task not in _TASK_SAMPLING_PRESETS:
        raise ValueError(f"Unknown sampling task '{task}'")
    name = model or config.model
    resolved_family = resolve_model_family(name, family)
    preset = _TASK_SAMPLING_PRESETS[task][resolved_family]
    return dataclasses.replace(config, **preset)


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

Grading philosophy:
- Be charitable. If a reasonable reading supports correctness, give credit.
- If the student shows correct method but makes an arithmetic slip, award \
partial credit for the method.
- Internal consistency rule: if this part depends on an earlier wrong answer \
but the student applies their own earlier result correctly here, award full \
credit for the method in this part. Do not double-penalize one earlier error.
- Answered-form rule: if the question asks for a specific form, grade the \
requested form. Example: a net ionic equation must actually be net ionic; a \
full molecular or full ionic equation does not satisfy that criterion.

For each question, you must:
1. Read what the student wrote
2. Compare it to the correct answer / rubric
3. Decide whether this question depends on an earlier part of the same \
problem. If yes, name that earlier part. If no, say "none".
4. If it does depend on an earlier part, decide whether the student's work \
here is internally consistent with their own earlier result.
5. Award a score, erring on the side of generosity while respecting the \
requested answer form.

Respond in EXACTLY this JSON format (no other text). The
upstream_dependency and if_dependent_then_consistent fields are
REQUIRED — the grading process is not complete without them, and they
must be filled in before model_score:

{
  "model_read": "<what the student wrote, verbatim>",
  "upstream_dependency": "<earlier part id this depends on, e.g. '5(a)', or 'none'>",
  "if_dependent_then_consistent": <true | false | null if upstream_dependency is 'none'>,
  "model_score": <numeric score you award>,
  "model_confidence": <0.0 to 1.0, your confidence in the score>,
  "model_reasoning": "<brief explanation of your grading, including how the upstream dependency was handled>"
}
"""

GRADING_PROMPT_VERSION = "2026-04-08-condensed-v1"
DESCRIBE_ONLY_PROMPT_VERSION = "2026-04-11-describe-only-v2"
DESCRIBE_ONLY_PROMPT = (
    "This is a page from a student's chemistry exam. The page may "
    "contain multiple questions; describe everything visually "
    "present on it.\n\n"
    "For each distinct piece of writing or drawing, transcribe what "
    "the student wrote as literally as possible — numbers, "
    "equations, diagrams, marginal notes — and note where it sits "
    "on the page (near which question number, in which margin). If "
    "any handwriting or drawing is ambiguous and admits more than "
    "one reasonable reading, list the alternatives.\n\n"
    "Do not grade. Do not apply chemistry knowledge to judge "
    "correctness. Describe what is visually present, nothing more."
)


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


def describe_prompt_metadata() -> dict[str, str]:
    content_hash = hashlib.sha256(
        DESCRIBE_ONLY_PROMPT.encode("utf-8")
    ).hexdigest()
    return {
        "version": DESCRIBE_ONLY_PROMPT_VERSION,
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
    reason_m = re.search(r'"?model_reasoning"?\s*:\s*"([^"]*)"', text)

    if score_m:
        return {
            "model_score": float(score_m.group(1)),
            "model_confidence": float(conf_m.group(1)) if conf_m else 0.5,
            "model_read": read_m.group(1) if read_m else "",
            "model_reasoning": reason_m.group(1) if reason_m else "",
        }

    raise ValueError(f"Could not parse VLM response as JSON: {text[:300]}")


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _extract_reasoning_tokens(delta: dict[str, Any]) -> list[str]:
    """Flatten provider-specific reasoning delta shapes into plain text."""
    details = delta.get("reasoning_details")
    if isinstance(details, list) and details:
        tokens: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = detail.get("text")
            if isinstance(text, str) and text:
                tokens.append(text)
                continue
            summary = detail.get("summary")
            if isinstance(summary, str) and summary:
                tokens.append(summary)
        if tokens:
            return tokens

    for field_name in ("reasoning_content", "reasoning"):
        value = delta.get(field_name)
        if isinstance(value, str) and value:
            return [value]

    return []


def stream_vision_completion(
    *,
    config: ServerConfig,
    prompt_text: str,
    page_image: bytes | None = None,
    image_data_url: str | None = None,
    system_prompt: str | None = None,
    on_reasoning_delta: Any = None,
    extra_body: dict[str, Any] | None = None,
    raw_dump_path: Path | None = None,
    timeout: float = 600,
    retries: int = 3,
    failure_context: str | None = None,
) -> tuple[str, str]:
    """Shared OpenAI-compatible vision call path for grading and smokes."""
    import time
    import urllib.error
    import urllib.request

    if image_data_url is None:
        if page_image is None:
            raise ValueError("page_image or image_data_url is required")
        image_data_url = _image_to_data_url(page_image)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    )

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "presence_penalty": config.presence_penalty,
        "repetition_penalty": config.repetition_penalty,
        "stream": True,
    }
    if extra_body:
        payload.update(extra_body)

    body = json.dumps(payload).encode()
    url = _chat_completions_url(config.base_url)

    def _build_request():
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return urllib.request.Request(
            url,
            data=body,
            headers=headers,
        )

    last_err: Exception | None = None
    content = ""
    reasoning = ""
    for attempt in range(retries):
        try:
            req = _build_request()
            resp = urllib.request.urlopen(req, timeout=timeout)
            try:
                content, reasoning = _consume_streaming_response(
                    resp,
                    on_reasoning_delta=on_reasoning_delta,
                    raw_dump_path=raw_dump_path,
                )
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
            break
        except KeyboardInterrupt:
            raise
        except (TimeoutError, OSError) as e:
            last_err = e
            if isinstance(e, urllib.error.HTTPError):
                try:
                    body_text = e.read().decode(
                        "utf-8", errors="replace"
                    )[:4096]
                except Exception:
                    body_text = "<could not read response body>"
                last_err = RuntimeError(
                    f"HTTP Error {e.code}: {e.reason} — server body: {body_text}"
                )
            if attempt < retries - 1:
                time.sleep(2)
    else:
        context = f" for {failure_context}" if failure_context else ""
        raise TimeoutError(
            f"VLM request failed after {retries} attempts{context}: {last_err}"
        )

    return content, reasoning


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
    prompt_text = _build_grading_prompt(item, template_question)
    content, reasoning = stream_vision_completion(
        config=config,
        prompt_text=prompt_text,
        page_image=page_image,
        system_prompt=_SYSTEM_PROMPT,
        on_reasoning_delta=on_reasoning_delta,
        failure_context=f"{item.exam_id}/{item.question_id}",
    )

    parsed = _parse_vlm_response(content)

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

    return Prediction(
        exam_id=item.exam_id,
        question_id=item.question_id,
        model_score=float(parsed.get("model_score", 0)),
        model_confidence=float(parsed.get("model_confidence", 0.5)),
        model_reasoning=str(parsed.get("model_reasoning", "")),
        model_read=str(parsed.get("model_read", "")),
        raw_assistant=content,
        raw_reasoning=reasoning,
        upstream_dependency=upstream_dependency,
        if_dependent_then_consistent=if_dependent_then_consistent,
    )


def _consume_streaming_response(
    resp, on_reasoning_delta=None, raw_dump_path: Path | None = None
) -> tuple[str, str]:
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
    dump_fh = open(raw_dump_path, "a") if raw_dump_path else None
    try:
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
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta", {}) or {}
            if dump_fh is not None:
                dump_fh.write(json.dumps(delta, ensure_ascii=False) + "\n")
            for token in _extract_reasoning_tokens(delta):
                reasoning_parts.append(token)
                if on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(token)
                    except Exception:
                        pass
            # Final assistant content — accumulate for parsing
            c_delta = delta.get("content", "")
            if c_delta:
                content_parts.append(c_delta)
    finally:
        if dump_fh is not None:
            dump_fh.close()
    return "".join(content_parts), "".join(reasoning_parts)


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
            narrator.stop_and_summarize(
                prediction=pred,
                item=item,
                template_question=tq,
            )

        if progress_callback:
            progress_callback(i + 1, len(ground_truth), item, pred)

    return predictions
