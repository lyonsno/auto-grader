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
        #   kind in {"line", "header", "topic", "drop"}
        self.history: Deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY_LINES)
        # Running counters
        self.stat_emitted = 0
        self.stat_dropped_dedup = 0
        self.stat_dropped_empty = 0

    def render(self) -> Group:
        # All chrome uses muted greys. The single accent color is cyan,
        # reserved for the live cursor and live text. Structural colors
        # (yellow for item headers, green for topics, red for drops) are
        # used only on dim/desaturated variants.

        # Header — title + running stats. Muted, single line.
        header_text = Text()
        header_text.append(self.title, style="bold bright_white")
        header_text.append("   ", style="dim")
        header_text.append(self.subtitle, style="grey50")
        header_text.append("   ", style="dim")
        header_text.append(
            f"emitted={self.stat_emitted}",
            style="green4" if self.stat_emitted > 0 else "grey50",
        )
        header_text.append("  ", style="dim")
        header_text.append(
            f"dedup={self.stat_dropped_dedup}",
            style="yellow4" if self.stat_dropped_dedup > 0 else "grey50",
        )
        header_text.append("  ", style="dim")
        header_text.append(
            f"empty={self.stat_dropped_empty}",
            style="red3" if self.stat_dropped_empty > 0 else "grey50",
        )
        header = Panel(
            Align.left(header_text),
            border_style="grey39",
            padding=(0, 1),
        )

        # Live line — the one place we use a real accent color (cyan).
        # The cursor glyph is solid cyan, the text is plain bright_white
        # for legibility against the dark background.
        if self.live_line:
            live_text = Text()
            live_text.append("▌ ", style="bright_cyan")
            live_text.append(self.live_line, style="bright_white")
        else:
            live_text = Text("▌ ", style="grey39")
        live_panel = Panel(
            live_text,
            border_style="grey39",
            padding=(0, 1),
            title="[grey50]live[/grey50]",
            title_align="left",
        )

        # History panel — newest at top, older below. All dimmed; only
        # item headers get a slight color lift to mark sections.
        history_lines = list(self.history)[-_VISIBLE_HISTORY_LINES:]
        history_lines.reverse()
        history_text = Text()
        for i, (kind, text) in enumerate(history_lines):
            if i > 0:
                history_text.append("\n")
            if kind == "header":
                # Section marker — slight amber, not blazing yellow
                history_text.append("─ ", style="grey39")
                history_text.append(text, style="bold orange3")
            elif kind == "topic":
                history_text.append("  · ", style="grey50")
                history_text.append(text, style="dark_sea_green4")
            elif kind == "drop":
                history_text.append("  ✗ ", style="grey39")
                history_text.append(text, style="grey39 strike")
            else:
                history_text.append("    ", style="dim")
                history_text.append(text, style="grey85")
        if not history_lines:
            history_text = Text(
                "(waiting for first summary...)", style="grey39"
            )

        history_panel = Panel(
            history_text,
            border_style="grey39",
            padding=(0, 1),
            title="[grey50]history[/grey50]",
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
            self.stat_emitted += 1

    def on_drop(self, reason: str, text: str) -> None:
        # Surface drop in history so user sees the narrator working
        label = text[:80] if text else f"<{reason}>"
        self.history.append(("drop", f"[{reason}] {label}"))
        self.live_line = ""  # clear any in-flight live content
        if reason == "dedup":
            self.stat_dropped_dedup += 1
        elif reason == "empty":
            self.stat_dropped_empty += 1

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
                    elif msg_type == "drop":
                        display.on_drop(
                            msg.get("reason", "unknown"),
                            msg.get("text", ""),
                        )
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
