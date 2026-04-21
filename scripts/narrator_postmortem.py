from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable

from auto_grader.narrator_run import (
    load_recorded_messages,
    safe_filename_fragment,
    summarize_narrator_items,
)
from auto_grader.vlm_inference import _chat_completions_url


_SYSTEM_PROMPT = """\
You are writing a retrospective markdown postmortem for a completed chemistry
grading run.

Your job is to analyze the run at the run level, not to re-grade each item from
scratch. Focus on:
- where the grader diverged from truth
- what the saved narrator transcript reveals about the grader's reasoning shape
- concrete prompt or evaluation changes worth trying next

Write markdown with these sections:
1. Executive Summary
2. Error Patterns
3. Narrator Signals
4. Next Changes

Be concrete, concise, and grounded in the provided run artifact payload.
"""


def _resolve_run_dir(run_arg: str | Path) -> Path:
    path = Path(run_arg)
    if path.name == "narrator.jsonl":
        return path.parent
    return path


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text())


def _load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in (run_dir / "predictions.jsonl").read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("type") == "prediction":
            rows.append(row)
    return rows


def _truth_score(row: dict[str, Any]) -> float | None:
    corrected = row.get("corrected_score")
    if isinstance(corrected, (int, float)):
        return float(corrected)
    professor = row.get("professor_score")
    if isinstance(professor, (int, float)):
        return float(professor)
    return None


def _default_base_url(manifest: dict[str, Any], model: str) -> str:
    narrator_model = manifest.get("narrator_model")
    narrator_url = manifest.get("narrator_url")
    base_url = manifest.get("base_url")
    if narrator_model == model and isinstance(narrator_url, str) and narrator_url:
        return narrator_url
    if isinstance(base_url, str) and base_url:
        return base_url
    if isinstance(narrator_url, str) and narrator_url:
        return narrator_url
    return "http://127.0.0.1:8001"


def _resolve_api_key(base_url: str) -> str:
    if "openrouter.ai" in base_url.lower():
        return os.environ.get("OPENROUTER_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


def _build_prompt_payload(
    manifest: dict[str, Any],
    predictions: list[dict[str, Any]],
    narrator_items: list[dict[str, Any]],
    wrap_up_text: str | None,
) -> dict[str, Any]:
    narrator_by_label = {
        str(item.get("item_label", "")): item for item in narrator_items
    }
    items: list[dict[str, Any]] = []
    for row in predictions:
        label = f"{row.get('exam_id', '')}/{row.get('question_id', '')}"
        narrator_item = narrator_by_label.get(label, {})
        items.append(
            {
                "item_label": label,
                "answer_type": row.get("answer_type"),
                "student_answer": row.get("student_answer"),
                "model_score": row.get("model_score"),
                "truth_score": _truth_score(row),
                "max_points": row.get("max_points"),
                "score_basis": row.get("score_basis"),
                "model_reasoning": row.get("model_reasoning"),
                "upstream_dependency": row.get("upstream_dependency"),
                "if_dependent_then_consistent": row.get(
                    "if_dependent_then_consistent"
                ),
                "narrator_topic": narrator_item.get("topic"),
                "narrator_committed_lines": narrator_item.get(
                    "committed_lines", []
                ),
                "narrator_structured_rows": narrator_item.get(
                    "structured_rows", []
                ),
                "narrator_checkpoints": narrator_item.get("checkpoints", []),
            }
        )
    return {
        "run": {
            "run_id": manifest.get("run_id"),
            "source_model": manifest.get("model"),
            "prompt_version": manifest.get("prompt_version"),
            "test_set_id": manifest.get("test_set_id"),
            "status": manifest.get("status"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "git_commit": manifest.get("git_commit"),
            "git_branch": manifest.get("git_branch"),
        },
        "items": items,
        "saved_wrap_up": wrap_up_text,
    }


def _build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write the postmortem from this saved run payload.\n\n"
                + json.dumps(payload, indent=2)
            ),
        },
    ]


def _request_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 60.0,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.4,
        "max_tokens": 1200,
    }
    lowered_model = model.casefold()
    if "qwen3.6" in lowered_model or "qwen3p6" in lowered_model:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def write_postmortem(
    run_dir: str | Path,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    output_path: Path | None = None,
    completion_fn: Callable[..., str] | None = None,
) -> Path:
    resolved_run_dir = _resolve_run_dir(run_dir)
    manifest = _load_manifest(resolved_run_dir)
    predictions = _load_predictions(resolved_run_dir)
    narrator_messages = load_recorded_messages(resolved_run_dir / "narrator.jsonl")
    narrator_items, wrap_up_text = summarize_narrator_items(narrator_messages)
    payload = _build_prompt_payload(
        manifest,
        predictions,
        narrator_items,
        wrap_up_text,
    )
    messages = _build_messages(payload)
    resolved_base_url = base_url or _default_base_url(manifest, model)
    resolved_api_key = api_key if api_key is not None else _resolve_api_key(
        resolved_base_url
    )
    completion = (
        completion_fn or _request_completion
    )(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=model,
        messages=messages,
    )

    if output_path is None:
        output_path = (
            resolved_run_dir
            / f"postmortem-{safe_filename_fragment(model)}.md"
        )
    header = "\n".join(
        [
            "# Narrator Postmortem",
            "",
            f"- Run: {manifest.get('run_id', resolved_run_dir.name)}",
            f"- Source model: {manifest.get('model', '')}",
            f"- Postmortem model: {model}",
            f"- Prompt version: {manifest.get('prompt_version', '')}",
            f"- Test set: {manifest.get('test_set_id', '')}",
            "",
            "## Run Snapshot",
            "",
        ]
    )
    snapshot_lines = [
        f"- {item['item_label']}: model={item['model_score']} truth={item['truth_score']} basis={item['score_basis']}"
        for item in payload["items"]
    ]
    body = "\n".join(snapshot_lines + ["", "## Model Analysis", "", completion.strip(), ""])
    output_path.write_text(header + body)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a retrospective markdown postmortem for a saved narrator run."
    )
    parser.add_argument("run", help="Run directory or narrator.jsonl path")
    parser.add_argument("--model", required=True, help="Model id for the postmortem pass")
    parser.add_argument(
        "--base-url",
        help="Optional OpenAI-compatible base URL override for the postmortem model",
    )
    parser.add_argument(
        "--api-key",
        help="Optional API key override for the postmortem model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output markdown path",
    )
    args = parser.parse_args(argv)

    path = write_postmortem(
        args.run,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
