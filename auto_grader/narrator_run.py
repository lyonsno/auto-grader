from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_STRUCTURED_ROW_KINDS = {
    "basis",
    "ambiguity",
    "credit_preserved",
    "deduction",
    "review_marker",
    "professor_mismatch",
}
_HEADER_LABEL_RE = re.compile(r"^\[item \d+/\d+\]\s+(\S+)")


def load_recorded_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if isinstance(msg, dict):
            messages.append(msg)
    return messages


def safe_filename_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return fragment.strip("-") or "output"


def summarize_narrator_items(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    live_chunks: list[str] = []
    wrap_up_text: str | None = None

    def flush_live() -> None:
        nonlocal live_chunks
        if current is None:
            live_chunks = []
            return
        text = "".join(live_chunks).strip()
        if text:
            current["committed_lines"].append(text)
        live_chunks = []

    def start_item(header_text: str) -> dict[str, Any]:
        match = _HEADER_LABEL_RE.match(header_text)
        return {
            "header": header_text,
            "item_label": match.group(1) if match else header_text,
            "committed_lines": [],
            "structured_rows": [],
            "checkpoints": [],
            "topic": None,
        }

    for msg in messages:
        msg_type = str(msg.get("type", ""))
        if msg_type == "header":
            flush_live()
            current = start_item(str(msg.get("text", "")))
            items.append(current)
            continue
        if msg_type == "delta":
            live_chunks.append(str(msg.get("text", "")))
            continue
        if msg_type == "commit":
            flush_live()
            continue
        if current is None:
            if msg_type == "wrap_up":
                wrap_up_text = str(msg.get("text", "")).strip() or None
            continue
        if msg_type in _STRUCTURED_ROW_KINDS:
            current["structured_rows"].append(
                {"kind": msg_type, "text": str(msg.get("text", ""))}
            )
        elif msg_type == "checkpoint":
            current["checkpoints"].append(str(msg.get("text", "")))
        elif msg_type == "topic":
            current["topic"] = str(msg.get("text", ""))
        elif msg_type == "wrap_up":
            wrap_up_text = str(msg.get("text", "")).strip() or None

    flush_live()
    return items, wrap_up_text
