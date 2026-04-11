#!/usr/bin/env python3
"""smoke_probe.py — diagnostic perception probe for VLMs.

Purpose: measure the *optical* ceiling of each vision model we have
access to, decoupled from the grading task. Instead of grading a
chemistry exam, the model is asked to *describe what is visually
present on the page* with no chemistry reasoning, no rubric, no
scoring. The outputs let us compare raw perception across models
and diagnose whether grading failures upstream are perception
failures (model can't see it) or reasoning failures (model sees
it but rationalizes a wrong conclusion).

Supports two backends:
  - local: an OMLX-compatible OpenAI endpoint on the LAN
  - openrouter: https://openrouter.ai/api/v1

OpenRouter backend mirrors the pattern used by spoke's command.py
(operation wallet moth-storm): bearer-token auth, streaming, the
server-specific `reasoning: {enabled: true}` body flag for thinking
models, and a flattener that handles all three reasoning-delta
shapes OR emits.

Outputs one JSONL row per (model, item) call to:
  ~/dev/auto-grader-runs/<ts>-<model>-probe/probe.jsonl

Each row is self-describing — includes the prompt, the model id,
the backend, elapsed time, the description, the reasoning trace,
and any error. Downstream analysis is a matter of reading the
JSONL, not re-deriving state from a run dir.

This is a diagnostic instrument, not a grading pass. It deliberately
does NOT:
  - load or compare against ground_truth scores
  - run a narrator
  - apply rubrics
  - call compare_runs or any grading-aggregation code
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Reuse the existing image extraction / data-URL helpers rather than
# duplicating them. smoke_probe is a peer of smoke_vlm in scripts/;
# both import from auto_grader.
from auto_grader.vlm_inference import (
    _EXAM_PDF_MAP,
    _image_to_data_url,
    extract_page_image,
)

_GROUND_TRUTH = Path(__file__).resolve().parent.parent / "eval" / "ground_truth.yaml"
_SCANS_DIR = Path.home() / "dev" / "auto-grader-assets" / "scans"
_DEFAULT_RUNS_ROOT = Path.home() / "dev" / "auto-grader-runs"

# Probe prompt v0 — describe-only, no chemistry reasoning, no grading.
# Deliberately asks for ambiguous-reading enumeration so downstream
# analysis can see where perception is uncertain vs where it commits.
PROBE_PROMPT_VERSION = "2026-04-11-describe-only-v2"
PROBE_PROMPT = (
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


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------


@dataclass
class ProbeBackend:
    """Connection info for a single backend call.

    base_url: the API root, WITHOUT the /chat/completions suffix. We
        append /v1/chat/completions (or just /chat/completions if the
        URL already carries a version prefix).
    api_key: bearer token. Empty string allowed for open local servers.
    is_openrouter: controls OR-specific body fields (reasoning flag) and
        the reasoning-delta flattener's OR path.
    """

    name: str  # "local" or "openrouter"
    base_url: str
    api_key: str = ""
    is_openrouter: bool = False


def _build_backend(args: argparse.Namespace) -> ProbeBackend:
    if args.backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "--backend openrouter requires OPENROUTER_API_KEY in the "
                "environment. Set it and re-run."
            )
        return ProbeBackend(
            name="openrouter",
            base_url=args.base_url or "https://openrouter.ai/api/v1",
            api_key=key,
            is_openrouter=True,
        )
    # local
    return ProbeBackend(
        name="local",
        base_url=args.base_url or "http://macbook-pro-2.local:8001",
        api_key=os.environ.get("OMLX_SERVER_API_KEY", "1234"),
        is_openrouter=False,
    )


# ---------------------------------------------------------------------------
# HTTP + streaming
# ---------------------------------------------------------------------------


def _extract_reasoning_tokens(delta: dict[str, Any]) -> list[str]:
    """Flatten provider-specific reasoning delta shapes into plain text.

    Providers emit reasoning in one of three shapes, sometimes more
    than one simultaneously on the same delta:
      - `reasoning_content` (plain string) — OMLX / some OpenAI-compat
        servers
      - `reasoning` (plain string) — OpenRouter legacy compat field
      - `reasoning_details` (structured list with per-item `text` /
        `summary` fields) — OpenRouter canonical shape

    OpenRouter emits the SAME token in both `reasoning` and
    `reasoning_details[].text` on every chunk. Naively reading both
    doubles every token and produces 'IIII nneeeedd' output.
    Diagnosed 2026-04-11 on google/gemma-4-26b-a4b-it, 4069/4071
    deltas had the dual shape.

    Fix: treat the three shapes as a priority chain, not an
    accumulation. Prefer reasoning_details (most typed, canonical on
    OR), fall back to reasoning_content, fall back to reasoning.
    Stop at the first non-empty source. This is safe because all
    three express the same underlying reasoning stream — no provider
    is known to split reasoning ACROSS shapes on a single delta.
    """
    details = delta.get("reasoning_details")
    if isinstance(details, list) and details:
        tokens: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = detail.get("text")
            if isinstance(text, str) and text:
                tokens.append(text)
            else:
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


def _consume_stream(resp, raw_dump_path: Path | None = None) -> tuple[str, str]:
    """Read an SSE stream from an OpenAI-compatible chat completion.

    Returns (content_text, reasoning_text). Both are plain strings
    accumulated from deltas. Mirrors the shape of
    auto_grader.vlm_inference._consume_streaming_response but without
    the narrator hooks — the probe has no narrator.

    If raw_dump_path is given, every delta dict is appended to that
    file as a JSONL row before flattening. This is a diagnostic
    affordance used to inspect provider-specific reasoning shapes;
    remove or leave unused in normal runs.
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
            # Some providers (notably OpenRouter) send non-choice events
            # mid-stream — usage summaries, keep-alives, provider metadata
            # — with an empty or absent `choices` array. Naive
            # chunk["choices"][0] crashes on those. Guard explicitly.
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta", {}) or {}
            if dump_fh is not None:
                dump_fh.write(json.dumps(delta, ensure_ascii=False) + "\n")
            for tok in _extract_reasoning_tokens(delta):
                reasoning_parts.append(tok)
            c = delta.get("content", "")
            if isinstance(c, str) and c:
                content_parts.append(c)
    finally:
        if dump_fh is not None:
            dump_fh.close()
    return "".join(content_parts), "".join(reasoning_parts)


def _call(
    backend: ProbeBackend,
    model: str,
    prompt: str,
    image_data_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: float = 600,
    raw_dump_path: Path | None = None,
) -> tuple[str, str, float]:
    """Single chat-completion call, streaming.

    Returns (content, reasoning, elapsed_sec). Raises on HTTP /
    transport errors — caller decides whether to retry or record as a
    probe failure.

    temperature defaults to 0.3 (low) because the task is
    description-only and we want stable perception, not exploration.
    max_tokens default 4096 is generous for a description task;
    perception responses should be much shorter than grading
    reasoning.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # OpenRouter wants an explicit reasoning flag to turn thinking on
    # for reasoning-capable models. Without it OR suppresses the
    # reasoning stream even on models that have it natively.
    if backend.is_openrouter:
        body["reasoning"] = {"enabled": True}

    # Compose URL. Local OMLX wants /v1/chat/completions; OpenRouter's
    # base already ends with /v1, so just append /chat/completions.
    if backend.base_url.rstrip("/").endswith("/v1"):
        url = f"{backend.base_url.rstrip('/')}/chat/completions"
    else:
        url = f"{backend.base_url.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    try:
        content, reasoning = _consume_stream(resp, raw_dump_path=raw_dump_path)
    finally:
        try:
            resp.close()
        except Exception:
            pass
    elapsed = time.time() - t0
    return content, reasoning, elapsed


# ---------------------------------------------------------------------------
# Item selection (reuses ground_truth.yaml just for exam/page mapping)
# ---------------------------------------------------------------------------


@dataclass
class ProbeItem:
    exam_id: str
    question_id: str
    page: int
    # Informational only — we do NOT compare against this
    student_answer: str = ""


def _load_items(pick_spec: str) -> list[ProbeItem]:
    """Parse --pick 'exam_id:question_id,exam_id:question_id,...' and
    resolve each to a ProbeItem by consulting ground_truth.yaml for
    the page number. We use ground_truth ONLY for the page mapping —
    no score comparison happens anywhere in the probe.
    """
    with open(_GROUND_TRUTH) as f:
        gt_raw = yaml.safe_load(f)

    index: dict[tuple[str, str], dict] = {}
    for exam in gt_raw.get("exams", []):
        exam_id = exam.get("exam_id") or exam.get("id", "")
        for q in exam.get("items", []):
            qid = q.get("question_id", "")
            index[(exam_id, qid)] = q

    items: list[ProbeItem] = []
    pairs = [p.strip() for p in pick_spec.split(",") if p.strip()]
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"--pick entry missing colon: {pair!r}")
        exam_id, qid = pair.split(":", 1)
        q = index.get((exam_id, qid))
        if q is None:
            raise SystemExit(
                f"--pick: no match in ground_truth for {exam_id}:{qid}"
            )
        items.append(
            ProbeItem(
                exam_id=exam_id,
                question_id=qid,
                page=int(q.get("page", 0)),
                student_answer=str(q.get("student_answer", "")),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Run dir + writer
# ---------------------------------------------------------------------------


def _run_dir(model: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = model.replace("/", "_")
    return _DEFAULT_RUNS_ROOT / f"{stamp}-{safe}-probe"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic perception probe for VLMs (describe-only)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id as known to the backend (e.g. qwen3p5-35B-A3B for "
        "local OMLX, or google/gemma-3-27b-it for openrouter).",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "openrouter"),
        default="local",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the backend base URL. Defaults: local → "
        "http://macbook-pro-2.local:8001, openrouter → "
        "https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--pick",
        required=True,
        help="Comma-separated exam:question pairs, e.g. "
        "'15-blue:fr-12b,15-blue:fr-1,15-blue:fr-8'",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run dir; defaults to ~/dev/auto-grader-runs/"
        "<ts>-<model>-probe/",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Per-call max_tokens budget (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3 — low, stable perception)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned calls without sending them.",
    )
    parser.add_argument(
        "--raw-dump",
        default=None,
        help="Diagnostic: path to write raw SSE delta dicts as JSONL. "
        "Used for inspecting provider-specific reasoning shapes when "
        "the flattener produces unexpected output.",
    )
    args = parser.parse_args()

    backend = _build_backend(args)
    items = _load_items(args.pick)

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir(args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    probe_path = run_dir / "probe.jsonl"

    print(f"Backend:  {backend.name} ({backend.base_url})")
    print(f"Model:    {args.model}")
    print(f"Items:    {len(items)} — {[f'{i.exam_id}:{i.question_id}' for i in items]}")
    print(f"Run dir:  {run_dir}")
    print(f"Prompt:   {PROBE_PROMPT_VERSION}")
    print()

    if args.dry_run:
        print("[dry-run] not sending any calls.")
        return 0

    # Write a header row first so the file is self-describing even if
    # the first inference call fails mid-stream.
    with open(probe_path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "header",
                    "backend": backend.name,
                    "base_url": backend.base_url,
                    "model": args.model,
                    "prompt_version": PROBE_PROMPT_VERSION,
                    "prompt": PROBE_PROMPT,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "started": datetime.now().isoformat(timespec="seconds"),
                }
            )
            + "\n"
        )

    # Cache page images so multi-item calls on the same page don't
    # re-render. Key by (exam_id, page).
    page_cache: dict[tuple[str, int], bytes] = {}

    n_ok = 0
    n_err = 0
    for i, item in enumerate(items, start=1):
        key = (item.exam_id, item.page)
        if key not in page_cache:
            pdf_name = _EXAM_PDF_MAP.get(item.exam_id)
            if not pdf_name:
                print(
                    f"[{i}/{len(items)}] {item.exam_id}:{item.question_id}"
                    f" — no PDF mapping, skipping",
                    file=sys.stderr,
                )
                n_err += 1
                continue
            pdf_path = _SCANS_DIR / pdf_name
            if not pdf_path.exists():
                print(
                    f"[{i}/{len(items)}] {item.exam_id}:{item.question_id}"
                    f" — scan not found ({pdf_path}), skipping",
                    file=sys.stderr,
                )
                n_err += 1
                continue
            page_cache[key] = extract_page_image(pdf_path, item.page)
        image_bytes = page_cache[key]
        image_url = _image_to_data_url(image_bytes)

        print(
            f"[{i}/{len(items)}] {item.exam_id}:{item.question_id} "
            f"(page {item.page}) ...",
            flush=True,
        )
        row: dict[str, Any] = {
            "type": "probe",
            "exam_id": item.exam_id,
            "question_id": item.question_id,
            "page": item.page,
            "student_answer_for_reference": item.student_answer,
            "model": args.model,
            "backend": backend.name,
        }
        try:
            content, reasoning, elapsed = _call(
                backend,
                args.model,
                PROBE_PROMPT,
                image_url,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                raw_dump_path=Path(args.raw_dump) if args.raw_dump else None,
            )
            row["description"] = content
            row["reasoning"] = reasoning
            row["elapsed_sec"] = round(elapsed, 2)
            row["error"] = None
            n_ok += 1
            # Print a short preview so the operator can eyeball progress
            preview = (content or "").strip().splitlines()[0:3]
            for line in preview:
                print(f"    {line[:140]}")
            print(
                f"    ({elapsed:.1f}s, {len(content)} chars content, "
                f"{len(reasoning)} chars reasoning)"
            )
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            row["description"] = ""
            row["reasoning"] = ""
            row["elapsed_sec"] = None
            row["error"] = f"HTTPError {e.code}: {body}"
            n_err += 1
            print(f"    HTTPError {e.code}: {body[:200]}", file=sys.stderr)
        except Exception as e:
            row["description"] = ""
            row["reasoning"] = ""
            row["elapsed_sec"] = None
            row["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
            print(f"    {type(e).__name__}: {e}", file=sys.stderr)

        with open(probe_path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(probe_path, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "footer",
                    "ended": datetime.now().isoformat(timespec="seconds"),
                    "count_ok": n_ok,
                    "count_err": n_err,
                }
            )
            + "\n"
        )

    print()
    print(f"Wrote {probe_path}")
    print(f"ok={n_ok} err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
