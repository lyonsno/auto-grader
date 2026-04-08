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
import threading
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

# Shimmer parameters — slow chyron sweep across the top N history lines.
# Each layer has a fixed phase offset relative to the one above it (so
# they're in stable orbit, not drifting), and intensity decays with
# layer position so older lines pulse dimmer than newer ones.
_SHIMMER_CYCLE_S = 1.8
_SHIMMER_WIDTH = 12          # how many characters wide the shimmer trail is
_SHIMMER_MAX_LAYERS = 6      # apply shimmer to the top N visible lines
_SHIMMER_LAYER_OFFSET = -0.06  # negative = wave appears to move downward
                              # through layers (top leads, lower lags)

# Base RGB colors per kind (for interpolation toward the shimmer peak)
_BASE_RGB = {
    "line": (190, 165, 195),    # soft mauve body — pinkish, distinct
    "topic": (110, 150, 110),   # dark_sea_green-ish
    "header": (200, 110, 30),   # orange3-ish
}
# Per-kind shimmer intensity multiplier — applied on top of layer_recency.
# Headers get cranked up so section markers really pulse, while normal
# lines get toned down so they're present but quiet (pinkish glow,
# subtle pulse). Topics stay at default.
_SHIMMER_KIND_INTENSITY = {
    "line": 0.45,    # quiet pulse on the body
    "topic": 1.00,
    "header": 1.40,  # cranked — section markers pop
}
# Shimmer peak — what each character's color is interpolated toward
# at the shimmer head. Warm yellow-orange for a fiery / ember
# aesthetic. Headers brighten toward gold, lines glow warm-pink,
# topics briefly flash chartreuse as the sweep passes.
_SHIMMER_PEAK_RGB = (255, 215, 120)


def _interp_rgb(
    base: tuple[int, int, int],
    peak: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linear interpolate from base toward peak by t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return (
        int(base[0] + (peak[0] - base[0]) * t),
        int(base[1] + (peak[1] - base[1]) * t),
        int(base[2] + (peak[2] - base[2]) * t),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _apply_shimmer(
    text_obj: Text,
    content: str,
    kind: str,
    layer_index: int,
    indent_width: int = 0,
    wrap_width: int | None = None,
) -> Text:
    """Append content to text_obj with a moving shimmer overlay.

    layer_index: 0 is the topmost (newest) line — full shimmer intensity
    and full chyron bold on the head. Each successive layer is dimmer
    (recency decay) and slightly phase-offset (each layer trails the one
    above by _SHIMMER_LAYER_OFFSET of a cycle, so the wave appears to
    ripple downward through the stack).

    indent_width: how many visual columns are taken by an indent already
    appended before this content (e.g. "    " is 4). Used to compute
    correct visual columns for wrap-aware shimmer.

    wrap_width: visual columns at which the panel wraps. When provided,
    the shimmer head sweeps based on VISUAL COLUMN (char_index modulo
    wrap_width) rather than absolute character index — this means
    chars at the same visual column on different visual rows of a
    wrapped logical line get the SAME shimmer treatment, so the wave
    appears as a vertical bar sweeping across all rows of a wrapped
    line in unison instead of scrolling along row 1 and then dropping
    to row 2. When None, falls back to character-index sweep.

    Past _SHIMMER_MAX_LAYERS the line is rendered static at base color.
    """
    if not content:
        return text_obj

    base_rgb = _BASE_RGB.get(kind, _BASE_RGB["line"])
    kind_intensity = _SHIMMER_KIND_INTENSITY.get(kind, 1.0)

    # Recency dimming — top is full, fades to zero at MAX_LAYERS
    if layer_index >= _SHIMMER_MAX_LAYERS:
        text_obj.append(content, style=_rgb_to_hex(base_rgb))
        return text_obj

    layer_recency = (1.0 - (layer_index / _SHIMMER_MAX_LAYERS)) * kind_intensity

    # Per-layer phase offset — each layer is shifted relative to the
    # one above by _SHIMMER_LAYER_OFFSET (negative = lags, so the wave
    # appears to flow downward through the stack)
    layer_phase_offset = (layer_index * _SHIMMER_LAYER_OFFSET) % 1.0

    now = time.monotonic()
    base_phase = (now % _SHIMMER_CYCLE_S) / _SHIMMER_CYCLE_S
    phase = (base_phase + layer_phase_offset) % 1.0

    if wrap_width is not None and wrap_width > _SHIMMER_WIDTH:
        # Visual-column sweep: head moves across the panel's interior
        # width and chars at the same visual column on different
        # visual rows of a wrapped line are in phase with each other.
        head = phase * (wrap_width + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        for i, ch in enumerate(content):
            absolute_col = indent_width + i
            visual_col = absolute_col % wrap_width
            distance = head - visual_col
            _append_shimmer_char(
                text_obj, ch, distance, base_rgb, layer_recency, layer_index
            )
    else:
        # Fallback character-index sweep (when no wrap_width given)
        head = phase * (len(content) + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        for i, ch in enumerate(content):
            distance = head - i
            _append_shimmer_char(
                text_obj, ch, distance, base_rgb, layer_recency, layer_index
            )

    return text_obj


def _append_shimmer_char(
    text_obj: Text,
    ch: str,
    distance: float,
    base_rgb: tuple[int, int, int],
    layer_recency: float,
    layer_index: int,
) -> None:
    """Append one shimmered character with the right style based on
    its distance from the shimmer head."""
    if distance < 0 or distance > _SHIMMER_WIDTH:
        color_rgb = base_rgb
        bold_head = False
    else:
        raw_intensity = 1.0 - (distance / _SHIMMER_WIDTH)
        intensity = raw_intensity * layer_recency
        color_rgb = _interp_rgb(base_rgb, _SHIMMER_PEAK_RGB, intensity)
        bold_head = (layer_index == 0 and -0.5 <= distance < 1.5)
    style = _rgb_to_hex(color_rgb)
    if bold_head:
        style = f"bold {style}"
    text_obj.append(ch, style=style)


class PaintDryDisplay:
    """Maintains the live + history state and renders via rich."""

    def __init__(self, console: Console | None = None):
        self._console = console
        self.title = "PROJECT PAINT DRY"
        self.subtitle = "bonsai narrator · live"
        self.live_line = ""
        # History entries are tuples (kind, text):
        #   kind in {"line", "header", "topic"}
        # Drops live in their own deque, rendered in a separate panel
        # below post-game so they don't clutter the narrative thread.
        self.history: Deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY_LINES)
        self.drops: Deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY_LINES)
        # Running counters
        self.stat_emitted = 0
        self.stat_dropped_dedup = 0
        self.stat_dropped_empty = 0
        # End-of-run wrap-up (color commentary)
        self.wrap_up_text: str = ""
        # When True, render() shows a "press Enter to close" footer.
        # The animation thread keeps running so the shimmer plays on
        # while the user reads.
        self.session_ended: bool = False

    def __rich__(self) -> Group:
        return self.render()

    def _compute_wrap_width(self) -> int | None:
        """Approximate visual width at which the history panel wraps.

        Panel chrome is borders (2) + padding (2) = 4 columns. The
        history Text is rendered inside that. Returns None if no
        console reference is available."""
        if self._console is None:
            return None
        try:
            term_width = self._console.size.width
        except Exception:
            return None
        # 2 borders + 2 padding cols = 4 chars of chrome
        usable = term_width - 4
        return max(20, usable)

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

        # History panel — newest at top, older below. Each visible line
        # gets a shimmer pass with intensity decaying by layer position
        # (top is brightest, fades to static at _SHIMMER_MAX_LAYERS).
        # Each layer is also slightly phase-offset so the sweep ripples
        # downward through the stack rather than all pulsing in lockstep.
        # Drops never shimmer — they're rejected, they should stay quiet.
        #
        # Wrap-aware shimmer: when a long line wraps inside the panel
        # to multiple visual rows, the shimmer is computed by VISUAL
        # COLUMN (modulo wrap_width) so the wave stays in phase across
        # the wrap — appears as a vertical bar sweeping all rows of a
        # wrapped line in unison.
        wrap_width = self._compute_wrap_width()

        history_lines = list(self.history)[-_VISIBLE_HISTORY_LINES:]
        history_lines.reverse()
        history_text = Text(no_wrap=False, overflow="fold")
        for i, (kind, text) in enumerate(history_lines):
            if i > 0:
                history_text.append("\n")

            if kind == "header":
                indent = "─ "
                history_text.append(indent, style="grey39")
                _apply_shimmer(
                    history_text, text, "header",
                    layer_index=i,
                    indent_width=len(indent),
                    wrap_width=wrap_width,
                )
            elif kind == "topic":
                indent = "  · "
                history_text.append(indent, style="grey50")
                _apply_shimmer(
                    history_text, text, "topic",
                    layer_index=i,
                    indent_width=len(indent),
                    wrap_width=wrap_width,
                )
            else:
                indent = "    "
                history_text.append(indent, style="dim")
                _apply_shimmer(
                    history_text, text, "line",
                    layer_index=i,
                    indent_width=len(indent),
                    wrap_width=wrap_width,
                )

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

        # Drops panel — below everything else, including post-game.
        # Pure debug surface; dim and quiet, no shimmer.
        drops_panel = None
        if self.drops:
            drops_text = Text(no_wrap=False, overflow="fold")
            visible_drops = list(self.drops)[-_VISIBLE_HISTORY_LINES:]
            for i, (reason, label) in enumerate(visible_drops):
                if i > 0:
                    drops_text.append("\n")
                drops_text.append("  ✗ ", style="grey39")
                drops_text.append(f"[{reason}] ", style="grey39")
                drops_text.append(label, style="grey42 strike")
            drops_panel = Panel(
                drops_text,
                border_style="grey30",
                padding=(0, 1),
                title=(
                    f"[grey42]rejected · "
                    f"dedup={self.stat_dropped_dedup} "
                    f"empty={self.stat_dropped_empty}[/grey42]"
                ),
                title_align="left",
            )

        wrap_panel = None
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

        # Order: header, live, history, post-game, drops, [footer]
        panels = [header, live_panel, history_panel]
        if wrap_panel is not None:
            panels.append(wrap_panel)
        if drops_panel is not None:
            panels.append(drops_panel)
        if self.session_ended:
            footer = Text(
                "  ▌ session ended — press Enter to close ▐",
                style="grey50 italic",
            )
            panels.append(footer)
        return Group(*panels)

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
        # Drops go to their own deque, rendered in a separate panel
        # below post-game. Keeps the history a clean read of what was
        # actually accepted while preserving the debug surface.
        label = text[:120] if text else f"<{reason}>"
        self.drops.append((reason, label))
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
    display = PaintDryDisplay(console=console)

    # Open the fifo for reading. This blocks until the writer connects.
    fd = os.open(str(fifo_path), os.O_RDONLY)
    fifo = os.fdopen(fd, "r", buffering=1)

    try:
        # auto_refresh=False — we drive the render manually from a
        # background timer thread. Tried auto_refresh=True with
        # display.__rich__ but rich's diff layer was treating
        # shimmer phase changes as no-op (because the underlying
        # text content was identical between frames, only the
        # per-character styles changed). Manual update + force
        # refresh works reliably.
        animation_stop = threading.Event()

        with Live(
            display.render(),
            console=console,
            refresh_per_second=30,
            screen=False,
            auto_refresh=False,
        ) as live:
            def _animation_tick():
                while not animation_stop.is_set():
                    try:
                        live.update(display.render(), refresh=True)
                    except Exception:
                        # Transient race with the message loop mutating
                        # display state — next tick will recover.
                        pass
                    time.sleep(1.0 / 30)

            anim_thread = threading.Thread(
                target=_animation_tick,
                name="paint-dry-animation",
                daemon=True,
            )
            anim_thread.start()

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
                        # Flag the display so render() shows a "press
                        # Enter" footer. Keep the animation thread
                        # running so the shimmer continues to play
                        # while the user reads the final state.
                        # Wait for stdin in a side thread.
                        display.session_ended = True
                        enter_event = threading.Event()

                        def _wait_enter():
                            try:
                                sys.stdin.readline()
                            except Exception:
                                pass
                            enter_event.set()

                        threading.Thread(
                            target=_wait_enter,
                            name="paint-dry-enter-wait",
                            daemon=True,
                        ).start()
                        enter_event.wait()
                        animation_stop.set()
                        anim_thread.join(timeout=0.5)
                        return 0
    finally:
        animation_stop.set()
        try:
            fifo.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
