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
import time
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

# Shimmer parameters — slow chyron sweep across the most recent committed
# line and each item header. The shimmer head is the brightest point
# (chyron leading character) and trails into a less-saturated pastel.
_SHIMMER_CYCLE_S = 1.8
_SHIMMER_WIDTH = 12  # how many characters wide the shimmer trail is


def _shimmer_style_for_distance(distance: float) -> str:
    """Pick a per-character style based on distance from the shimmer head.

    distance=0 is the chyron leading character (brightest), distance grows
    as we move backward along the trail; once past _SHIMMER_WIDTH the
    character is at the base style.
    """
    if distance < 0:
        # Shimmer hasn't reached this character yet
        return ""
    if distance < 1:
        return "bold bright_white"
    if distance < 3:
        return "bright_white"
    if distance < 6:
        return "light_cyan1"
    if distance < 9:
        return "light_steel_blue1"
    if distance < _SHIMMER_WIDTH:
        return "grey85"
    return ""


def _apply_shimmer(text_obj: Text, content: str, base_style: str) -> Text:
    """Append `content` to text_obj with a moving shimmer overlay.

    The shimmer head sweeps left-to-right across the content over
    _SHIMMER_CYCLE_S seconds. Characters not currently within the
    shimmer trail get base_style.
    """
    if not content:
        return text_obj

    now = time.monotonic()
    phase = (now % _SHIMMER_CYCLE_S) / _SHIMMER_CYCLE_S  # 0..1
    # Head sweeps from -_SHIMMER_WIDTH to len(content) so the trail
    # enters from the left and exits off the right.
    head = phase * (len(content) + _SHIMMER_WIDTH) - _SHIMMER_WIDTH

    for i, ch in enumerate(content):
        distance = head - i
        shimmer_style = _shimmer_style_for_distance(distance)
        text_obj.append(ch, style=shimmer_style or base_style)
    return text_obj


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
        # End-of-run wrap-up (color commentary)
        self.wrap_up_text: str = ""

    def __rich__(self) -> Group:
        return self.render()

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
        # The cursor glyph is solid cyan, the text is plain bright_white.
        # overflow="fold" forces rich to wrap long lines at panel width
        # instead of truncating them. The panel will grow vertically as
        # needed to fit a long bonsai line.
        if self.live_line:
            live_text = Text(no_wrap=False, overflow="fold")
            live_text.append("▌ ", style="bright_cyan")
            live_text.append(self.live_line, style="bright_white")
        else:
            live_text = Text("▌ ", style="grey39", overflow="fold")
        live_panel = Panel(
            live_text,
            border_style="grey39",
            padding=(0, 1),
            title="[grey50]live[/grey50]",
            title_align="left",
        )

        # History panel — newest at top, older below. The TOPMOST line
        # (whatever kind) gets a chyron shimmer, AND every "header" entry
        # gets a shimmer regardless of position. Other entries are static.
        history_lines = list(self.history)[-_VISIBLE_HISTORY_LINES:]
        history_lines.reverse()
        history_text = Text(no_wrap=False, overflow="fold")
        for i, (kind, text) in enumerate(history_lines):
            if i > 0:
                history_text.append("\n")
            is_top = (i == 0)
            apply_shimmer_here = is_top or kind == "header"

            if kind == "header":
                history_text.append("─ ", style="grey39")
                base_style = "bold orange3"
                if apply_shimmer_here:
                    _apply_shimmer(history_text, text, base_style)
                else:
                    history_text.append(text, style=base_style)
            elif kind == "topic":
                history_text.append("  · ", style="grey50")
                base_style = "dark_sea_green4"
                if apply_shimmer_here:
                    _apply_shimmer(history_text, text, base_style)
                else:
                    history_text.append(text, style=base_style)
            elif kind == "drop":
                history_text.append("  ✗ ", style="grey39")
                history_text.append(text, style="grey39 strike")
            else:
                history_text.append("    ", style="dim")
                base_style = "grey85"
                if apply_shimmer_here:
                    _apply_shimmer(history_text, text, base_style)
                else:
                    history_text.append(text, style=base_style)
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

        if self.wrap_up_text:
            wrap_text = Text(
                self.wrap_up_text,
                style="bright_white",
                no_wrap=False,
                overflow="fold",
            )
            wrap_panel = Panel(
                wrap_text,
                border_style="orange3",
                padding=(0, 1),
                title="[bold orange3]post-game[/bold orange3]",
                title_align="left",
            )
            return Group(header, live_panel, history_panel, wrap_panel)

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

    def on_rollback_live(self) -> None:
        """Discard the in-flight live line without committing.

        Used when a streaming summary gets dedup-rejected after the fact.
        The user briefly saw the typewriter, now it clears."""
        self.live_line = ""

    def on_wrap_up(self, text: str) -> None:
        """Final post-game commentary from bonsai."""
        self.wrap_up_text = text

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
        # auto_refresh=True so the shimmer animates between events.
        # The PaintDryDisplay implements __rich__ so each refresh tick
        # re-runs render() and gets a fresh shimmer phase.
        with Live(
            display,
            console=console,
            refresh_per_second=30,
            screen=False,
            auto_refresh=True,
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
                    elif msg_type == "rollback_live":
                        display.on_rollback_live()
                    elif msg_type == "topic":
                        display.on_topic(msg.get("text", ""))
                    elif msg_type == "drop":
                        display.on_drop(
                            msg.get("reason", "unknown"),
                            msg.get("text", ""),
                        )
                    elif msg_type == "wrap_up":
                        display.on_wrap_up(msg.get("text", ""))
                    elif msg_type == "end":
                        # auto_refresh is already running; one extra
                        # render tick will pick up the final state.
                        time.sleep(0.1)
                        # Stop the live so input() doesn't fight the
                        # background refresh thread.
                        live.stop()
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
