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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Reuse the existing image extraction / data-URL helpers rather than
# duplicating them. smoke_probe is a peer of smoke_vlm in scripts/;
# both import from auto_grader.
from auto_grader.vlm_inference import (
    DESCRIBE_ONLY_PROMPT,
    DESCRIBE_ONLY_PROMPT_VERSION,
    _EXAM_PDF_MAP,
    ServerConfig,
    apply_model_sampling_preset,
    extract_page_image,
    known_model_families,
    resolve_model_family,
    stream_vision_completion,
)

_GROUND_TRUTH = Path(__file__).resolve().parent.parent / "eval" / "ground_truth.yaml"
_SCANS_DIR = Path.home() / "dev" / "auto-grader-assets" / "scans"
_DEFAULT_RUNS_ROOT = Path.home() / "dev" / "auto-grader-runs"

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
# Shared request config
# ---------------------------------------------------------------------------


def _build_request_config(
    *,
    backend: ProbeBackend,
    model: str,
    model_family: str,
    max_tokens: int,
    temperature_override: float | None,
) -> ServerConfig:
    config = ServerConfig(
        base_url=backend.base_url,
        api_key=backend.api_key,
        model=model,
        max_tokens=max_tokens,
    )
    config = apply_model_sampling_preset(
        config,
        family=model_family,
        task="describe",
    )
    if temperature_override is not None:
        config = replace(config, temperature=temperature_override)
    return config


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
        "--model-family",
        choices=known_model_families(),
        default=None,
        help=(
            "Explicit sampling family override for unregistered models. "
            "Required when --model does not match a known family prefix."
        ),
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
        default=None,
        help=(
            "Optional override for the family default temperature. "
            "Describe-only runs otherwise use low, stable perception settings."
        ),
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
    resolved_family = resolve_model_family(args.model, args.model_family)
    items = _load_items(args.pick)
    config = _build_request_config(
        backend=backend,
        model=args.model,
        model_family=resolved_family,
        max_tokens=args.max_tokens,
        temperature_override=args.temperature,
    )

    run_dir = Path(args.run_dir) if args.run_dir else _run_dir(args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    probe_path = run_dir / "probe.jsonl"

    print(f"Backend:  {backend.name} ({backend.base_url})")
    print(f"Model:    {args.model}")
    print(f"Family:   {resolved_family}")
    print(f"Items:    {len(items)} — {[f'{i.exam_id}:{i.question_id}' for i in items]}")
    print(f"Run dir:  {run_dir}")
    print(f"Prompt:   {DESCRIBE_ONLY_PROMPT_VERSION}")
    print(
        f"Sampling: temp={config.temperature} top_p={config.top_p} "
        f"top_k={config.top_k} min_p={config.min_p} "
        f"presence={config.presence_penalty} rep={config.repetition_penalty}"
    )
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
                    "model_family": resolved_family,
                    "prompt_version": DESCRIBE_ONLY_PROMPT_VERSION,
                    "prompt": DESCRIBE_ONLY_PROMPT,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
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
            t0 = time.time()
            content, reasoning = stream_vision_completion(
                config=config,
                prompt_text=DESCRIBE_ONLY_PROMPT,
                page_image=image_bytes,
                extra_body=(
                    {"reasoning": {"enabled": True}}
                    if backend.is_openrouter
                    else None
                ),
                raw_dump_path=Path(args.raw_dump) if args.raw_dump else None,
                failure_context=f"{item.exam_id}/{item.question_id}",
            )
            elapsed = time.time() - t0
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
