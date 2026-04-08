"""Rich-powered reader for the Project Paint Dry narrator stream.

Spawned by NarratorSink in a fresh Terminal.app window. Reads JSON-line
messages from a fifo and renders a live display:

    ╭ PROJECT PAINT DRY ─────────────────────────────╮
    │ qwen3p5-35B-A3B grading                        │
    ╰────────────────────────────────────────────────╯

      ▍ Splitting credit on the Lewis structure...        <- LIVE (top)

      ──────────────────────────────────────────────
      [item 12/38] 15-blue/fr-12a lewis_structure
        Catching the missing second resonance form        <- HISTORY
        Reading the ozone Lewis structure for double bonds
        Comparing electron pair geometry to the rubric
      [item 11/38] 15-blue/fr-11c
        Awarding the half-point for the correct count
        ...

The live line is pinned at top and updates char-by-char as bonsai
streams. Committed lines slot in at the top of the history (just
below live), pushing older lines down.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


_MAX_HISTORY_LINES = 60  # cap so we don't grow unbounded
_VISIBLE_HISTORY_LINES = 20  # how many to actually render


class PaintDryDisplay:
    """Maintains the live + history state and renders via rich."""

    def __init__(self):
        self.title = "PROJECT PAINT DRY"
        self.subtitle = "bonsai narrator · live"
        self.live_line = ""
        # History entries are tuples (kind, text):
        #   kind in {"line", "header", "topic"}
        self.history: Deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY_LINES)

    def render(self) -> Group:
        # Header panel
        header_text = Text()
        header_text.append(self.title, style="bold magenta")
        header_text.append("  ·  ", style="dim")
        header_text.append(self.subtitle, style="dim cyan")
        header = Panel(
            Align.left(header_text),
            border_style="magenta",
            padding=(0, 1),
        )

        # Live line — bright cyan with leading cursor glyph
        if self.live_line:
            live_text = Text()
            live_text.append("▍ ", style="bold cyan blink")
            live_text.append(self.live_line, style="bold white")
        else:
            live_text = Text("▍ ", style="dim cyan")
        live_panel = Panel(
            live_text,
            border_style="cyan",
            padding=(0, 1),
            title="[bold cyan]live[/bold cyan]",
            title_align="left",
        )

        # History panel — most recent first (top), older below
        history_lines = list(self.history)[-_VISIBLE_HISTORY_LINES:]
        history_lines.reverse()  # newest at top of panel
        history_text = Text()
        for i, (kind, text) in enumerate(history_lines):
            if i > 0:
                history_text.append("\n")
            if kind == "header":
                history_text.append(text, style="bold yellow")
            elif kind == "topic":
                history_text.append("  → ", style="dim green")
                history_text.append(text, style="green")
            else:
                history_text.append("  ", style="dim")
                history_text.append(text, style="white")
        if not history_lines:
            history_text = Text("(waiting for first commit...)", style="dim")

        history_panel = Panel(
            history_text,
            border_style="dim",
            padding=(0, 1),
            title="[dim]history[/dim]",
            title_align="left",
        )

        return Group(header, live_panel, history_panel)

    # -- mutators ----------------------------------------------------------

    def on_header(self, text: str) -> None:
        # Header goes into the history with a distinct style
        self.history.append(("header", text))

    def on_delta(self, text: str) -> None:
        self.live_line += text

    def on_commit(self) -> None:
        if self.live_line:
            self.history.append(("line", self.live_line))
            self.live_line = ""

    def on_topic(self, text: str) -> None:
        # Commit any in-flight live line first
        if self.live_line:
            self.history.append(("line", self.live_line))
            self.live_line = ""
        self.history.append(("topic", text))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: narrator_reader.py <fifo-path>", file=sys.stderr)
        return 2

    fifo_path = Path(sys.argv[1])
    if not fifo_path.exists():
        print(f"fifo not found: {fifo_path}", file=sys.stderr)
        return 2

    console = Console()
    display = PaintDryDisplay()

    # Open the fifo for reading. This blocks until the writer connects.
    fd = os.open(str(fifo_path), os.O_RDONLY)
    fifo = os.fdopen(fd, "r", buffering=1)

    try:
        with Live(
            display.render(),
            console=console,
            refresh_per_second=20,
            screen=False,
            auto_refresh=False,
        ) as live:
            buffer = ""
            while True:
                chunk = fifo.read(1)
                if not chunk:
                    # Writer closed
                    break
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "header":
                        display.on_header(msg.get("text", ""))
                    elif msg_type == "delta":
                        display.on_delta(msg.get("text", ""))
                    elif msg_type == "commit":
                        display.on_commit()
                    elif msg_type == "topic":
                        display.on_topic(msg.get("text", ""))
                    elif msg_type == "end":
                        live.update(display.render(), refresh=True)
                        # Wait for keypress before exiting so the user
                        # can read the final state.
                        console.print(
                            "\n[dim]session ended — press Enter to close...[/dim]"
                        )
                        try:
                            input()
                        except (EOFError, KeyboardInterrupt):
                            pass
                        return 0

                    live.update(display.render(), refresh=True)
    finally:
        try:
            fifo.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
