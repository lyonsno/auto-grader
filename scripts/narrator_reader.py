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
import math
import os
import re
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


# Matches the elapsed-time prefix on after-action topic lines:
#   "47s · Grader: 2/2 (matched). Prof: 2/2. · Even the kid called this."
# Group 1: the elapsed time ("47s"), group 2: the rest after the bullet.
_TIME_PREFIX_RE = re.compile(r"^(\d+s)\s*·\s*(.*)$", re.DOTALL)


_MAX_HISTORY_LINES = 60  # cap so we don't grow unbounded
_VISIBLE_HISTORY_LINES = 20  # how many to actually render

# Shimmer parameters — slow chyron sweep across the top N history lines.
# Each layer has a fixed phase offset relative to the one above it (so
# they're in stable orbit, not drifting), and intensity decays with
# layer position so older lines pulse dimmer than newer ones.
_SHIMMER_DEFAULT_CYCLE_S = 2.7  # slowed 50% from prior 1.8 — most lines pulse calmly
_SHIMMER_RECENT_CYCLE_S = 1.2   # most-recently-committed line pulses 50% faster
                                 # than the original 1.8 — strong contrast against
                                 # the slowed default
_SHIMMER_WIDTH = 12          # how many characters wide the shimmer trail is
_SHIMMER_MAX_LAYERS = 16     # apply full-recency shimmer to the top N lines
_SHIMMER_LAYER_OFFSET = -0.04  # negative = wave appears to move downward
                              # through layers (top leads, lower lags).
                              # Smaller magnitude than before because we
                              # now span more layers.
# Headers and topics retain a faint shimmer FLOOR even past _SHIMMER_MAX_LAYERS,
# so the structural markers and verdict lines never go fully static.
_SHIMMER_FLOOR_RECENCY = 0.15  # 15% recency for floored kinds

# Base RGB colors per kind (for interpolation toward the shimmer peak)
_BASE_RGB = {
    "line": (190, 165, 195),     # soft mauve body — pinkish, distinct
    "line_alt": (225, 140, 170), # alternating warmer pink — punchier,
                                  # higher saturation, slight coral lean
    "topic": (110, 150, 110),    # dark_sea_green-ish
    "header": (200, 110, 30),    # orange3-ish
    "live": (240, 240, 248),     # bright off-white for the live field,
                                  # very slight cool cast so warm shimmer
                                  # peak pops against it
}
# Per-kind shimmer intensity multiplier — applied on top of layer_recency.
# Headers get cranked up so section markers really pulse, while normal
# lines get toned down so they're present but quiet (pinkish glow,
# subtle pulse). Topics stay at default. Live gets a subtle amplitude
# but bright peak color override below.
_SHIMMER_KIND_INTENSITY = {
    "line": 0.45,        # quiet pulse on the body
    "line_alt": 0.45,    # match line intensity for the alternate color
    "topic": 1.00,
    "header": 1.40,      # cranked — section markers pop
    "live": 0.55,        # subtle amplitude, but with vivid peak (below)
}
# Per-kind override of the shimmer peak color. Lives that aren't here
# fall through to the global _SHIMMER_PEAK_RGB.
_SHIMMER_KIND_PEAK_RGB = {
    "live": (255, 165, 40),   # vivid orange-amber for the live shimmer
}
# Kinds that retain a faint shimmer floor past _SHIMMER_MAX_LAYERS
_SHIMMER_FLOORED_KINDS = frozenset({"header", "topic"})

# Live-line undulation parameters — each character on the live line
# gets a per-position, per-time hue from a yellow→orange→red palette.
# Adjacent characters have slightly different hues (per-char phase
# offset) and the whole field undulates over a slow cycle.
_LIVE_UNDULATION_CYCLE_S = 3.5    # full hue cycle period
_LIVE_HUE_CENTER_DEG = 32          # center of the yellow-orange-red band
_LIVE_HUE_RANGE_DEG = 24           # +/- swing → 8°-56°, red-orange to yellow
_LIVE_PER_CHAR_PHASE_OFFSET = 0.18 # phase shift per character (radians)
_LIVE_BASE_SAT = 0.85              # base saturation
_LIVE_BASE_VAL = 0.92              # base value (brightness)
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


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (h in degrees, s/v in [0, 1]) to 8-bit RGB."""
    h = h % 360
    h_sector = int(h / 60) % 6
    f = (h / 60) - int(h / 60)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if h_sector == 0:
        r, g, b = v, t, p
    elif h_sector == 1:
        r, g, b = q, v, p
    elif h_sector == 2:
        r, g, b = p, v, t
    elif h_sector == 3:
        r, g, b = p, q, v
    elif h_sector == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (
        max(0, min(255, int(r * 255))),
        max(0, min(255, int(g * 255))),
        max(0, min(255, int(b * 255))),
    )


def _apply_shimmer(
    text_obj: Text,
    content: str,
    kind: str,
    layer_index: int,
    indent_width: int = 0,
    wrap_width: int | None = None,
    cycle_s: float | None = None,
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
    peak_rgb = _SHIMMER_KIND_PEAK_RGB.get(kind, _SHIMMER_PEAK_RGB)
    kind_intensity = _SHIMMER_KIND_INTENSITY.get(kind, 1.0)

    # Recency dimming — top is full, fades to zero at MAX_LAYERS.
    # Floored kinds (headers, topics) keep at least _SHIMMER_FLOOR_RECENCY
    # so structural markers and verdict lines never go fully static.
    raw_recency = max(0.0, 1.0 - (layer_index / _SHIMMER_MAX_LAYERS))
    if kind in _SHIMMER_FLOORED_KINDS:
        raw_recency = max(raw_recency, _SHIMMER_FLOOR_RECENCY)
    if raw_recency <= 0:
        # Past the dimming horizon and no floor — render fully static
        text_obj.append(content, style=_rgb_to_hex(base_rgb))
        return text_obj
    layer_recency = raw_recency * kind_intensity

    # Per-layer phase offset — each layer is shifted relative to the
    # one above by _SHIMMER_LAYER_OFFSET (negative = lags, so the wave
    # appears to flow downward through the stack). Cap the layer index
    # used for phase offset at MAX_LAYERS-1 so deep floored layers all
    # share a stable phase rather than drifting wildly.
    phase_layer = min(layer_index, _SHIMMER_MAX_LAYERS - 1)
    layer_phase_offset = (phase_layer * _SHIMMER_LAYER_OFFSET) % 1.0

    cycle = cycle_s if cycle_s is not None else _SHIMMER_DEFAULT_CYCLE_S
    now = time.monotonic()
    base_phase = (now % cycle) / cycle
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
                text_obj, ch, distance, base_rgb, peak_rgb,
                layer_recency, layer_index
            )
    else:
        # Fallback character-index sweep (when no wrap_width given)
        head = phase * (len(content) + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        for i, ch in enumerate(content):
            distance = head - i
            _append_shimmer_char(
                text_obj, ch, distance, base_rgb, peak_rgb,
                layer_recency, layer_index
            )

    return text_obj


def _render_live_undulating(
    text_obj: Text,
    content: str,
    indent_width: int,
    wrap_width: int | None,
    is_active: bool,
) -> Text:
    """Render the live line with per-character undulating warm colors
    (yellow / orange / red) AND a shimmer overlay on top.

    Each character has its own hue computed from time + char position,
    so adjacent characters land at slightly different points in the
    yellow-orange-red palette and the whole field undulates over a
    slow cycle. The shimmer head brightens characters near it and
    pushes their saturation down (toward white) for a heat-flicker
    feel.
    """
    if not content:
        return text_obj

    now = time.monotonic()
    undulation_phase_base = now * (2 * math.pi / _LIVE_UNDULATION_CYCLE_S)

    # Shimmer head — uses the recent cycle so the live field moves
    # at the same pace as the most recent line in history (the
    # other "live-feeling" element).
    cycle = _SHIMMER_RECENT_CYCLE_S
    shimmer_phase = (now % cycle) / cycle
    if wrap_width is not None and wrap_width > _SHIMMER_WIDTH:
        shimmer_head = (
            shimmer_phase * (wrap_width + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        )
    else:
        shimmer_head = (
            shimmer_phase * (len(content) + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        )

    # Frozen state has slightly muted colors so it reads as "settled"
    sat_mul = 1.0 if is_active else 0.7
    val_mul = 1.0 if is_active else 0.85

    for i, ch in enumerate(content):
        # Per-character undulating hue
        char_phase = undulation_phase_base + i * _LIVE_PER_CHAR_PHASE_OFFSET
        h = _LIVE_HUE_CENTER_DEG + _LIVE_HUE_RANGE_DEG * math.sin(char_phase)
        s = _LIVE_BASE_SAT * sat_mul
        v = _LIVE_BASE_VAL * val_mul

        # Shimmer overlay: brighten and de-saturate at the head
        if wrap_width is not None and wrap_width > _SHIMMER_WIDTH:
            visual_col = (indent_width + i) % wrap_width
            distance = shimmer_head - visual_col
        else:
            distance = shimmer_head - i

        bold_head = False
        if 0 <= distance < _SHIMMER_WIDTH:
            shimmer_intensity = 1.0 - (distance / _SHIMMER_WIDTH)
            # Push toward white-hot at the head (boost V, drop S)
            v = min(1.0, v + 0.08 * shimmer_intensity)
            s = max(0.0, s - 0.40 * shimmer_intensity)
            if -0.5 <= distance < 1.5:
                bold_head = is_active

        r, g, b = _hsv_to_rgb(h, s, v)
        style = f"#{r:02x}{g:02x}{b:02x}"
        if bold_head:
            style = f"bold {style}"
        text_obj.append(ch, style=style)

    return text_obj


def _append_shimmer_char(
    text_obj: Text,
    ch: str,
    distance: float,
    base_rgb: tuple[int, int, int],
    peak_rgb: tuple[int, int, int],
    layer_recency: float,
    layer_index: int,
) -> None:
    """Append one shimmered character with the right style based on
    its distance from the shimmer head. peak_rgb may be a per-kind
    override (e.g. orange for live) or the global yellow peak."""
    if distance < 0 or distance > _SHIMMER_WIDTH:
        color_rgb = base_rgb
        bold_head = False
    else:
        raw_intensity = 1.0 - (distance / _SHIMMER_WIDTH)
        intensity = raw_intensity * layer_recency
        color_rgb = _interp_rgb(base_rgb, peak_rgb, intensity)
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

        # Sticky live: two buffers. streaming_line is the in-progress
        # bonsai dispatch (the typewriter source). frozen_line is the
        # most recent committed bonsai line, which keeps showing in the
        # live panel until the next dispatch starts streaming new
        # content. Live panel shows streaming if non-empty, else frozen,
        # else just the cursor glyph.
        self.streaming_line: str = ""
        self.frozen_line: str = ""

        # History entries are 3-tuples (kind, text, parity):
        #   kind in {"line", "header", "topic"}
        #   parity is 0 or 1 for "line" entries (alternation), None for others
        # Drops live in their own deque, rendered in a separate panel
        # below post-game so they don't clutter the narrative thread.
        self.history: Deque[tuple[str, str, int | None]] = deque(maxlen=_MAX_HISTORY_LINES)
        self.drops: Deque[tuple[str, str]] = deque(maxlen=_MAX_HISTORY_LINES)

        # Per-line parity counter for mauve/pink alternation. Toggles
        # on each accepted "line" commit. Stored permanently per entry
        # so the alternation is stable as new lines arrive (no flicker).
        self._line_parity: int = 0

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

    def _build_display_entries(self) -> list[tuple[tuple, bool]]:
        """Group history into items, reverse so newest item is first,
        and within each group keep entries in chronological order so
        the header sits ABOVE its narrator lines and topic.

        Returns a list of (entry, is_most_recent) tuples in display
        order, capped to _VISIBLE_HISTORY_LINES. is_most_recent is
        True for exactly the entry at the back of the deque (the
        most-recently-committed thing), so callers can give it a
        faster shimmer cycle for visual contrast.
        """
        history_list = list(self.history)
        if not history_list:
            return []
        most_recent_idx = len(history_list) - 1

        # Forward-iterate, grouping at header boundaries, tracking
        # original deque indices so we can identify the most-recent.
        groups: list[list[tuple[tuple, int]]] = []
        current_group: list[tuple[tuple, int]] = []
        for idx, entry in enumerate(history_list):
            if entry[0] == "header":
                if current_group:
                    groups.append(current_group)
                current_group = [(entry, idx)]
            else:
                current_group.append((entry, idx))
        if current_group:
            groups.append(current_group)

        # Newest item on top, but entries within each group stay in
        # their natural (commit) order so header > lines > topic
        groups.reverse()

        display: list[tuple[tuple, bool]] = []
        for group in groups:
            for entry, idx in group:
                display.append((entry, idx == most_recent_idx))
                if len(display) >= _VISIBLE_HISTORY_LINES:
                    return display
        return display

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

        # Compute the wrap width once — used by both the live panel
        # shimmer and the history panel shimmer.
        wrap_width = self._compute_wrap_width()

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

        # Live line — sticky two-buffer model. Show streaming_line if
        # it's non-empty (active dispatch), otherwise show frozen_line
        # (the last committed line, waiting for the next dispatch to
        # start). Cursor glyph is bright cyan when actively streaming,
        # grey50 when only the frozen line is showing — subtle visual
        # cue that the field is settled vs. live.
        #
        # The displayed text gets a subtle yellow/orange shimmer
        # overlay (via _apply_shimmer with kind="live") so the live
        # field feels alive even between deltas. Subtle amplitude,
        # vivid peak (orange-amber).
        displayed_live = self.streaming_line or self.frozen_line
        is_active = bool(self.streaming_line)

        if displayed_live:
            live_text = Text(no_wrap=False, overflow="fold")
            cursor_style = "bright_cyan" if is_active else "grey50"
            live_text.append("▌ ", style=cursor_style)
            _render_live_undulating(
                live_text, displayed_live,
                indent_width=2,  # cursor glyph "▌ "
                wrap_width=wrap_width,
                is_active=is_active,
            )
        else:
            live_text = Text("▌ ", style="grey39", overflow="fold")
        live_panel = Panel(
            live_text,
            border_style="grey39",
            padding=(0, 1),
            title="[grey50]live[/grey50]",
            title_align="left",
        )

        # History panel — items grouped by header. Each item is a
        # group: header at the top, then narrator lines in chronological
        # order beneath it, then the topic at the bottom. Groups are
        # rendered newest-first, so the current item sits at the top
        # of the panel and older items sink below as new items start.
        #
        # Within each group entries are in their natural (commit) order,
        # so the header always sits ABOVE its own narrator lines.
        #
        # Layer index for shimmer is visual position (0 = topmost),
        # so the current item's header gets the brightest shimmer and
        # fades downward through the visible layers.
        #
        # Wrap-aware shimmer: when a long line wraps inside the panel
        # to multiple visual rows, the shimmer is computed by VISUAL
        # COLUMN (modulo wrap_width) so the wave stays in phase across
        # the wrap.
        display_entries = self._build_display_entries()
        history_text = Text(no_wrap=False, overflow="fold")
        for i, (entry, is_most_recent) in enumerate(display_entries):
            kind = entry[0]
            text = entry[1]
            parity = entry[2] if len(entry) > 2 else None
            if i > 0:
                history_text.append("\n")

            # Most-recently-committed entry pulses on the FAST cycle;
            # everything else uses the slowed default. Strong contrast.
            entry_cycle = (
                _SHIMMER_RECENT_CYCLE_S
                if is_most_recent
                else _SHIMMER_DEFAULT_CYCLE_S
            )

            if kind == "header":
                indent = "─ "
                history_text.append(indent, style="grey39")
                _apply_shimmer(
                    history_text, text, "header",
                    layer_index=i,
                    indent_width=len(indent),
                    wrap_width=wrap_width,
                    cycle_s=entry_cycle,
                )
            elif kind == "topic":
                indent = "  · "
                history_text.append(indent, style="grey50")
                # Pull out the elapsed-time prefix and color it as a
                # punchy warm accent (bold orange3 — same warm as the
                # post-game border and header base) so it pops out
                # from the rest of the line.
                m = _TIME_PREFIX_RE.match(text)
                if m:
                    time_prefix, rest = m.group(1), m.group(2)
                    history_text.append(time_prefix, style="bold orange3")
                    history_text.append("  ·  ", style="grey50")
                    extra_indent = len(time_prefix) + len("  ·  ")
                    _apply_shimmer(
                        history_text, rest, "topic",
                        layer_index=i,
                        indent_width=len(indent) + extra_indent,
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                    )
                else:
                    _apply_shimmer(
                        history_text, text, "topic",
                        layer_index=i,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                    )
            else:
                indent = "    "
                history_text.append(indent, style="dim")
                # Pick mauve or warmer pink based on the line's stored
                # parity. Stored per-entry (not computed from position)
                # so the alternation is stable as new lines arrive and
                # old lines fall off the deque.
                line_kind = "line_alt" if parity == 1 else "line"
                _apply_shimmer(
                    history_text, text, line_kind,
                    layer_index=i,
                    indent_width=len(indent),
                    wrap_width=wrap_width,
                    cycle_s=entry_cycle,
                )

        if not display_entries:
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
        # Header goes into the history. Doesn't touch the live buffers
        # — frozen_line keeps showing the previous committed dispatch
        # until the next bonsai dispatch starts streaming.
        self.history.append(("header", text, None))

    def on_delta(self, text: str) -> None:
        self.streaming_line += text

    def on_commit(self) -> None:
        # Push the just-finished streaming line into history (preserves
        # chronological order, so any subsequent topic/header lands
        # AFTER it), then promote it to frozen_line so it keeps showing
        # in the live panel until the next dispatch starts streaming.
        if self.streaming_line:
            self.history.append(
                ("line", self.streaming_line, self._line_parity)
            )
            self._line_parity = 1 - self._line_parity
            self.stat_emitted += 1
        self.frozen_line = self.streaming_line
        self.streaming_line = ""

    def on_drop(self, reason: str, text: str) -> None:
        # Drops go to their own deque, rendered in a separate panel
        # below post-game. Keeps the history a clean read of what was
        # actually accepted while preserving the debug surface.
        # rollback_live already cleared the streaming_line.
        label = text[:120] if text else f"<{reason}>"
        self.drops.append((reason, label))
        if reason == "dedup":
            self.stat_dropped_dedup += 1
        elif reason == "empty":
            self.stat_dropped_empty += 1

    def on_rollback_live(self) -> None:
        """Discard the in-flight streaming line without committing.

        Used when a streaming summary gets dedup-rejected. frozen_line
        is left UNCHANGED — the previous committed line keeps showing
        in the live panel, so the user sees a clean snap-back to the
        last accepted line instead of an empty live field."""
        self.streaming_line = ""

    def on_wrap_up(self, text: str) -> None:
        """Final post-game commentary from bonsai."""
        self.wrap_up_text = text

    def on_topic(self, text: str) -> None:
        # Topic (after-action) lands in history. Doesn't touch live
        # buffers — frozen_line keeps showing the last bonsai line.
        self.history.append(("topic", text, None))


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
