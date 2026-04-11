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
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Deque

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# scripts/ is not on sys.path by default when narrator_reader.py is
# spawned standalone. Add the repo root so the auto_grader package
# imports cleanly regardless of how the script is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from auto_grader.shimmer_phases import ShimmerPhaseState  # noqa: E402


# Matches the elapsed-time prefix on after-action topic lines:
#   "47s · Grader: 2/2 (matched). Prof: 2/2. · Even the kid called this."
# Group 1: the elapsed time ("47s"), group 2: the rest after the bullet.
_TIME_PREFIX_RE = re.compile(r"^(\d+s)\s*·\s*(.*)$", re.DOTALL)

# Splits an item header into the [item N/M] index marker and the rest
# of the title:
#   "[item 3/6] 15-blue/fr-5b (numeric, 2.0 pts)"
# Group 1 = "[item 3/6]", group 2 = "15-blue/fr-5b (numeric, 2.0 pts)".
# The index marker gets the cool steel-blue accent for an always-on
# cool note that's structural metadata, not status.
_HEADER_INDEX_RE = re.compile(r"^(\[item \d+/\d+\])\s*(.*)$", re.DOTALL)


_MAX_HISTORY_LINES = 90  # cap so we don't grow unbounded
_VISIBLE_HISTORY_LINES = 30  # how many to actually render

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
_SHIMMER_FLOOR_RECENCY = 0.40  # bumped from 0.15 — older headers and
                                # topics had decayed too much past the
                                # dimming horizon to read as alive; this
                                # keeps the structural pulse visible all
                                # the way down the stack instead of just
                                # on the most recent few items

# Base RGB colors per kind (for interpolation toward the shimmer peak).
# Sumi-e palette: a Japanese garden floor in two desaturated rows
# (sage moss + dust earth), with persimmon (柿色) headers, indigo
# (藍色) [item N/M] markers, and bright garden colors — celadon (青磁),
# vermilion (朱色), ochre (黄土) — landing as full-saturation accents
# on verdict topics. The narration alternation now carries the
# garden palette structurally (instead of bone-on-bone, which was
# too subtle to read), so celadon/vermilion/ochre exist on screen
# regardless of which verdicts the grader produces. Bone survives
# as the live-field background, the unknown-verdict topic fallback,
# and the global shimmer-peak highlight color.
_BASE_RGB = {
    "line": (135, 160, 145),     # muted celadon — sage moss row,
                                  # desaturated cousin of topic_match
                                  # so the verdict variant still pops
    "line_alt": (175, 160, 130), # muted ochre — dust earth row,
                                  # desaturated cousin of topic_undershoot
    "topic": (220, 205, 180),    # warm bone — fallback when verdict is
                                  # unknown / no prediction data. Bone's
                                  # structural home outside the live field
    "header": (228, 100, 50),    # persimmon (柿色) — vivid lacquer red,
                                  # the warm anchor of the painting.
                                  # Brighter than the muted version we
                                  # started with: real torii-gate /
                                  # tea-ceremony lacquer is bold, not
                                  # apologetic, and the cool indigo
                                  # axis was visually outweighing it
    "header_index": (90, 115, 180),    # indigo (藍色) — the [item N/M]
                                       # marker carries the cool axis
                                       # of the painting
    "live": (245, 240, 225),     # rice paper — warm off-white for the
                                  # live field, the brightest bone
                                  # surface in the composition
    # Topic verdict variants — full-saturation garden colors. The
    # narration rows above use desaturated cousins of these, so the
    # eye reads "muted family below, vivid accent here" and the
    # verdict still encodes meaning at a glance.
    "topic_match": (150, 208, 214),       # electric celadon — cooler,
                                          # brighter affirmative read.
                                          # Still garden-adjacent, but
                                          # pulled toward aqua so
                                          # agreement feels cleaner and
                                          # more "alive" than the old
                                          # mossy celadon
    "topic_overshoot": (210, 90, 65),     # vermilion (朱色) — too generous
    "topic_undershoot": (200, 150, 70),   # ochre (黄土) — too strict
    # Header dash — vermilion stroke at the start of every item header.
    # Gives vermilion a STRUCTURAL home (was the only verdict color
    # appearing purely as a verdict indicator) and pulses in sync with
    # the rest of the header so the painting reads as one stroke per
    # item: vermilion dash → indigo index → persimmon title.
    "header_dash": (210, 90, 65),
}
# Per-kind shimmer intensity multiplier — applied on top of layer_recency.
# Headers get cranked up so section markers really pulse, while normal
# lines get toned down so they're present but quiet (pinkish glow,
# subtle pulse). Topics stay at default. Live gets a subtle amplitude
# but bright peak color override below.
_SHIMMER_KIND_INTENSITY = {
    "line": 1.30,        # bumped from 1.10 — body rows still wave too
    "line_alt": 1.30,    # faintly to feel alive; this gives the
                          # coupled-oscillator phase ripple more
                          # presence on the largest visual surface
    "topic": 1.00,
    "topic_match": 1.10,        # slight extra shimmer lift so agreement
                                # gets its own pulse instead of reading
                                # like a neutral fallback
    "topic_overshoot": 1.00,
    "topic_undershoot": 1.00,
    "header": 1.40,      # cranked — section markers pop
    "header_index": 1.40,        # match header intensity for the cool index
    "header_dash": 1.40,         # match header intensity — the dash is
                                  # part of the same structural stroke
    "live": 0.55,        # subtle amplitude, but with vivid peak (below)
}
# Per-kind override of the shimmer peak color. Lives that aren't here
# fall through to the global _SHIMMER_PEAK_RGB.
_SHIMMER_KIND_PEAK_RGB = {
    # In-family peaks: every kind brightens within its own hue
    # family, so the shimmer wave reads as ink glistening (a single
    # hue getting brighter and back) rather than crossing through
    # off-palette midpoints toward a generic cream highlight. This
    # also avoids the persimmon→cream interpolation midpoint, which
    # passes through peachy pink and visually breaks the sumi-e
    # restraint when the wave is on a header.
    "live": (245, 155, 80),       # persimmon ember — live field warms
                                   # toward the same lacquer-red as the
                                   # headers as the wave passes
    "header": (255, 165, 95),     # fired persimmon — bright lacquer
                                   # in-family brightening, pushed to
                                   # match the brighter base
    "header_index": (185, 210, 240),  # rain-cleared sky blue — indigo
                                       # brightens toward the pale sky
                                       # after a storm wash painting
    "line": (175, 215, 180),      # glazed celadon — sage moss row
                                   # brightens toward kiln-glaze green
    "line_alt": (225, 200, 150),  # fired ochre — dust earth row
                                   # brightens toward kiln-fired earth
    "topic_match": (195, 232, 255),     # rain-lit sky celadon — borrows
                                        # the cooler blue family we
                                        # weren't using enough, so the
                                        # match shimmer reads electric
                                        # rather than ochre-warm
    "topic_overshoot": (250, 140, 105), # fired vermilion — bright
                                         # lacquer warning
    "topic_undershoot": (245, 195, 110), # fired ochre — bright earth
    "header_dash": (250, 140, 105),      # fired vermilion — the dash
                                          # brightens toward the same
                                          # bright lacquer that the
                                          # topic_overshoot verdict uses,
                                          # so vermilion has one
                                          # consistent identity across
                                          # both its structural and
                                          # indicator surfaces
}
# Kinds that retain a faint shimmer floor past _SHIMMER_MAX_LAYERS
_SHIMMER_FLOORED_KINDS = frozenset({
    "header",
    "header_index",
    "header_dash",
    "topic",
    "topic_match",
    "topic_overshoot",
    "topic_undershoot",
})

# Live panel reserves a fixed vertical footprint so it doesn't jitter
# the layout when bonsai produces a long line that wraps. The panel
# always shows _LIVE_PANEL_CONTENT_LINES rows of content (plus the
# top + bottom borders). When bonsai's output is longer than will
# fit in that area, we tail-truncate (keep the most recent chars).
_LIVE_PANEL_CONTENT_LINES = 3

# Live-line undulation parameters — each character on the live line
# gets a per-position, per-time hue from a warm orange-amber palette.
# Adjacent characters have slightly different hues (per-char phase
# offset) and the whole field undulates over a slow cycle.
# Pulled toward orange (away from yellow) and slightly desaturated
# from the previous values to harmonize with the rest of the sunset
# palette without losing fire feel.
_LIVE_UNDULATION_CYCLE_S = 6.0    # bumped from 3.5 (~1.7x) so the
                                   # hue undulation cycles slower per
                                   # unit time. Combined with the
                                   # per-hue luminance compensation,
                                   # this attacks the "hard on the
                                   # eyes" problem from the temporal
                                   # axis — fewer flicker cycles per
                                   # second = less perceptual fatigue
                                   # while reading the live line
_LIVE_HUE_CENTER_DEG = 18          # pulled toward persimmon red-orange
_LIVE_HUE_RANGE_DEG = 22           # widened swing → −4°-40°, slightly
                                    # more travel through the persimmon
                                    # / vermilion family so the per-char
                                    # undulation is actually visible
_LIVE_PER_CHAR_PHASE_OFFSET = 0.18 # phase shift per character (radians)
_LIVE_BASE_SAT = 0.80              # bumped from 0.62 — the live field
                                    # was washing out into static beige
                                    # because the saturation was too low
                                    # for the eye to read the undulation;
                                    # this restores warm pop without
                                    # crossing into neon territory
_LIVE_BASE_VAL = 0.95              # bright paper base
# Per-hue luminance compensation for the live undulation. At constant
# HSV V, pure red and pure yellow have very different perceived
# brightness (BT.709 luminance weights yellow ~4× higher than red),
# so the per-character hue undulation across the red-orange / amber
# range reads as a luminance flicker — your eye keeps trying to track
# the brightness change instead of just reading the text. We
# compensate by scaling V inversely with the per-char perceived
# luminance, normalized against the luminance at the hue center.
# Characters at the bright (amber/yellow) end of the swing get V
# dropped; characters at the dark (red) end stay at full V. Strength
# blends toward full equalization: 0.0 = no correction (raw HSV),
# 1.0 = fully flat perceived luminance. We use 0.65 — enough to
# kill the "darker red, lighter yellow" harshness without flattening
# the hue motion into a static orange band.
_LIVE_LUMINANCE_CORRECTION_STRENGTH = 0.65
# When a streaming dispatch finishes (on_commit), the live field
# stops updating but stays visible as the "frozen" line until the
# next dispatch starts. The settled state has slightly muted sat/val
# (0.70 / 0.85 multipliers) so it reads as past tense. Previously
# this transition was a step function — the bright "just-finished"
# pulse only existed for one frame. The fade lets the just-arrived
# line LINGER in its bright state and then slowly relax into the
# settled state, so the punch of completing a dispatch is something
# you can actually see and enjoy.
_LIVE_FREEZE_FADE_S = 2.5
_LIVE_FROZEN_SAT_MUL = 0.70
_LIVE_FROZEN_VAL_MUL = 0.85
# Shimmer peak — what each character's color is interpolated toward
# at the shimmer head. Pale moonlit gold (the highlight on a brush
# stroke as the wash dries), so the wave reads as a quiet brightening
# of the ink rather than a fire sweep.
_SHIMMER_PEAK_RGB = (235, 215, 175)


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
    phase_override: float | None = None,
) -> Text:
    """Append content to text_obj with a moving shimmer overlay.

    layer_index: 0 is the topmost (newest) line — full shimmer intensity
    and full chyron bold on the head. Each successive layer is dimmer
    (recency decay) and slightly phase-offset (each layer trails the one
    above by _SHIMMER_LAYER_OFFSET of a cycle, so the wave appears to
    ripple downward through the stack).

    phase_override: when supplied, use this phase directly instead of
    computing one from time.monotonic() and the per-layer offset. The
    coupled-oscillator phase state (ShimmerPhaseState) provides this
    so the layers can have weakly-coupled drifting periods that orbit
    the ideal stack instead of marching in lockstep. The override is
    expected to already account for the per-layer offset.

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

    if phase_override is not None:
        # Coupled-oscillator state already accounts for the per-layer
        # offset (and for per-layer period perturbations). Use the
        # provided phase verbatim.
        phase = phase_override % 1.0
    else:
        # Per-layer phase offset — each layer is shifted relative to
        # the one above by _SHIMMER_LAYER_OFFSET (negative = lags, so
        # the wave appears to flow downward through the stack). Cap
        # the layer index at MAX_LAYERS-1 so deep floored layers all
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
    char_offset: int = 0,
    freeze_age_s: float | None = None,
) -> Text:
    """Render the live line with per-character undulating warm colors
    (yellow / orange / red) AND a shimmer overlay on top.

    Each character has its own hue computed from time + char position,
    so adjacent characters land at slightly different points in the
    palette and the whole field undulates over a slow cycle. The
    shimmer head brightens characters near it and pushes their
    saturation down (toward white) for a heat-flicker feel.

    char_offset: number of characters that were tail-truncated off
    the front of `content` before passing in. Used to keep the
    per-character undulation phase stable across truncation — each
    character's phase is computed from its GLOBAL position in the
    full unfolded content, not its visible index, so the colors
    don't jump when a character falls off the front.

    freeze_age_s: when the line is frozen (is_active=False), this is
    seconds since freezing started. The renderer linearly interpolates
    saturation and value multipliers from 1.0 (fully bright, just
    finished streaming) toward _LIVE_FROZEN_SAT_MUL / _LIVE_FROZEN_VAL_MUL
    (settled past tense) over _LIVE_FREEZE_FADE_S. None when the line
    is actively streaming or there's no recorded freeze timestamp.
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

    # Sat/val multipliers fade smoothly from active brightness toward
    # the settled past-tense state over _LIVE_FREEZE_FADE_S after the
    # streaming dispatch finishes. While streaming, both stay at 1.0.
    if is_active or freeze_age_s is None:
        sat_mul = 1.0
        val_mul = 1.0
    else:
        fade = max(0.0, min(1.0, freeze_age_s / _LIVE_FREEZE_FADE_S))
        sat_mul = 1.0 - fade * (1.0 - _LIVE_FROZEN_SAT_MUL)
        val_mul = 1.0 - fade * (1.0 - _LIVE_FROZEN_VAL_MUL)

    # Reference luminance: the BT.709 perceived brightness at the hue
    # CENTER, computed at the current saturation. Used as the target
    # that all per-char luminances are scaled toward, so the hue
    # undulation no longer reads as a brightness flicker. Computed
    # once per frame outside the per-char loop.
    s_for_ref = _LIVE_BASE_SAT * sat_mul
    ref_r, ref_g, ref_b = _hsv_to_rgb(_LIVE_HUE_CENTER_DEG, s_for_ref, 1.0)
    ref_luminance = 0.2126 * ref_r + 0.7152 * ref_g + 0.0722 * ref_b

    for i, ch in enumerate(content):
        # Per-character undulating hue. Use the GLOBAL position
        # (visible index + char_offset) so the phase pattern stays
        # stable across truncation — characters don't change color
        # as the front of the buffer falls off.
        global_i = char_offset + i
        char_phase = (
            undulation_phase_base + global_i * _LIVE_PER_CHAR_PHASE_OFFSET
        )
        h = _LIVE_HUE_CENTER_DEG + _LIVE_HUE_RANGE_DEG * math.sin(char_phase)
        s = _LIVE_BASE_SAT * sat_mul
        v = _LIVE_BASE_VAL * val_mul

        # Per-hue luminance compensation. At constant V, BT.709 weights
        # mean amber/yellow chars are perceived ~3-4× brighter than
        # red chars, which makes the hue undulation read as a luminance
        # flicker that resists passive reading. Scale V inversely with
        # the per-char perceived luminance, blended toward neutral by
        # _LIVE_LUMINANCE_CORRECTION_STRENGTH. The result is the same
        # hue motion at roughly the same perceived brightness.
        test_r, test_g, test_b = _hsv_to_rgb(h, s, 1.0)
        test_luminance = (
            0.2126 * test_r + 0.7152 * test_g + 0.0722 * test_b
        )
        if test_luminance > 1:
            raw_correction = ref_luminance / test_luminance
            correction = 1.0 + (raw_correction - 1.0) * _LIVE_LUMINANCE_CORRECTION_STRENGTH
            v = max(0.0, min(1.0, v * correction))

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


class HistoryViewport:
    """Pure-logic scroll viewport for the Paint Dry history pane.

    Carries the entries the renderer should draw plus a scroll offset
    that is anchored to the *newest* visual row. Offset 0 means "the
    bottom of the visible window sits at the newest visual row" — the
    live edge. A positive offset means "the bottom of the window is
    `scroll_offset` visual rows above the newest visual row," which is
    what happens when the operator scrolls upward.

    Accounting is done in **visual rows**, not logical entries, so a
    wrapped long entry occupies its true on-screen footprint. Partial
    entries are never returned: if the remaining budget cannot contain
    a whole entry, the entry is dropped rather than sliced.

    Auto-follow semantics:

    * While `at_live_edge` (offset == 0), appending new history keeps
      the window pinned to newest — the operator always sees fresh
      rows.
    * While scrolled up (offset > 0), appending new history does NOT
      reset the offset. The viewport's offset is re-anchored relative
      to the new newest row so the same earlier slice stays visible.

    This class is intentionally free of Rich / terminal / input
    dependencies. Renderer wiring and raw key handling live elsewhere.
    """

    def __init__(self, visible_rows: int, wrap_width: int):
        if visible_rows <= 0:
            raise ValueError("visible_rows must be > 0")
        if wrap_width <= 0:
            raise ValueError("wrap_width must be > 0")
        self._visible_rows = visible_rows
        self._wrap_width = wrap_width
        self._entries: list[tuple[str, str, int | None]] = []
        # scroll_offset counts *visual rows above the newest visual row*.
        self._scroll_offset = 0

    # -- Visual row accounting --------------------------------------

    def _entry_visual_rows(self, entry: tuple[str, str, int | None]) -> int:
        text = entry[1] if len(entry) > 1 else ""
        if not text:
            return 1
        # Ceiling division so a 25-char line at wrap_width=10 is 3 rows.
        return max(1, (len(text) + self._wrap_width - 1) // self._wrap_width)

    def _total_visual_rows(self) -> int:
        return sum(self._entry_visual_rows(e) for e in self._entries)

    # -- Public state ------------------------------------------------

    @property
    def scroll_offset(self) -> int:
        return self._scroll_offset

    @property
    def at_live_edge(self) -> bool:
        return self._scroll_offset == 0

    def entries_snapshot(self) -> list[tuple[str, str, int | None]]:
        """Return a shallow copy of the entries currently held by the
        viewport in natural (oldest -> newest) order. Used by the
        sync layer to detect prefix divergence without touching
        private state."""
        return list(self._entries)

    # -- Mutation ----------------------------------------------------

    def append(self, entry: tuple[str, str, int | None]) -> None:
        """Add a new entry. If the viewport is at the live edge, the
        window auto-follows newest. If scrolled up, the offset is
        re-anchored so the same earlier slice stays visible.
        """
        self._entries.append(entry)
        if self._scroll_offset == 0:
            return
        # Scrolled up: re-anchor the offset so the same earlier content
        # remains on screen. Offset is "rows above newest visual row",
        # and a new entry added `k` visual rows to the bottom of the
        # stream pushes the previously-anchored slice up by `k`, so the
        # offset must grow by `k` to keep tracking the same slice.
        self._scroll_offset += self._entry_visual_rows(entry)
        self._clamp_offset()

    def scroll_up(self, n: int = 1) -> None:
        if n <= 0:
            return
        self._scroll_offset += n
        self._clamp_offset()

    def scroll_down(self, n: int = 1) -> None:
        if n <= 0:
            return
        self._scroll_offset = max(0, self._scroll_offset - n)

    def scroll_to_live_edge(self) -> None:
        self._scroll_offset = 0

    def _clamp_offset(self) -> None:
        # Maximum meaningful offset: push the bottom of the window all
        # the way up to the top of the oldest visual row. Beyond that
        # there is nothing more to reveal.
        total = self._total_visual_rows()
        max_offset = max(0, total - 1)
        if self._scroll_offset > max_offset:
            self._scroll_offset = max_offset

    # -- Windowing ---------------------------------------------------

    def visible_entries(self) -> list[tuple[str, str, int | None]]:
        """Return the entries whose visual rows fall inside the current
        window, in natural (oldest -> newest) order. Partial entries
        are never returned.
        """
        if not self._entries:
            return []

        # Compute the [bottom, top) visual-row window relative to the
        # newest visual row. Bottom row of the window is at visual-row
        # index `scroll_offset` from the newest; top row is
        # `scroll_offset + visible_rows`.
        rows_per_entry = [self._entry_visual_rows(e) for e in self._entries]
        total_rows = sum(rows_per_entry)

        # Walk entries from newest backwards, tracking how many visual
        # rows sit at-or-below the top of each entry, measured from the
        # newest visual row.
        # `rows_from_newest_top` = visual rows from the top of this
        # entry down to (and including) the newest visual row.
        bottom = self._scroll_offset
        top = self._scroll_offset + self._visible_rows

        # Build (entry, entry_bottom_from_newest, entry_top_from_newest)
        # where entry_bottom_from_newest is the visual-row distance from
        # the *bottom* (newest row) of the stream up to the bottom row
        # of this entry, and entry_top_from_newest is that distance to
        # the top row of the entry.
        spans: list[tuple[tuple[str, str, int | None], int, int]] = []
        cursor = 0  # rows accumulated from newest so far
        for entry, rows in zip(reversed(self._entries), reversed(rows_per_entry)):
            entry_bottom = cursor
            entry_top = cursor + rows
            spans.append((entry, entry_bottom, entry_top))
            cursor += rows
        # `spans` is newest -> oldest. Reverse back to natural order.
        spans.reverse()

        selected: list[tuple[str, str, int | None]] = []
        for entry, eb, et in spans:
            # Entry is fully inside the window iff its span [eb, et) is
            # contained in [bottom, top).
            if eb >= bottom and et <= top:
                selected.append(entry)
        # Guard: if total content is smaller than the visible budget,
        # selected already contains everything that fits. If nothing
        # fits (e.g. a single entry taller than the window), return
        # whatever empty slice the window currently sees — the caller
        # must tolerate an empty return rather than get a partial entry.
        _ = total_rows
        return selected


_SCROLL_PAGE_ROWS = 10


class HistoryScrollController:
    """Maps single keystrokes to PaintDryDisplay scroll actions.

    Pure logic: the live reader installs one of these and feeds it a
    character at a time from a cbreak-mode stdin reader thread. Key
    bindings are intentionally single-byte so the controller does not
    have to parse escape sequences for arrow keys — that's a future
    slice if vim-style `hjkl` + `0` proves insufficient.
    """

    def __init__(self, display: "PaintDryDisplay"):
        self._display = display

    def bindings(self) -> dict[str, str]:
        return {
            "k": "scroll history up one row",
            "j": "scroll history down one row",
            "u": f"scroll history up {_SCROLL_PAGE_ROWS} rows (page up)",
            "d": f"scroll history down {_SCROLL_PAGE_ROWS} rows (page down)",
            "0": "return to live edge (newest history)",
        }

    def handle_key(self, key: str) -> bool:
        """Route `key` to a scroll action. Returns True if the key
        was bound and an action was taken, False if the key is
        unbound (caller may swallow or forward it)."""
        if key == "k":
            self._display.scroll_history_up(1)
            return True
        if key == "j":
            self._display.scroll_history_down(1)
            return True
        if key == "u":
            self._display.scroll_history_up(_SCROLL_PAGE_ROWS)
            return True
        if key == "d":
            self._display.scroll_history_down(_SCROLL_PAGE_ROWS)
            return True
        if key == "0":
            self._display.scroll_history_to_live_edge()
            return True
        return False


class PaintDryDisplay:
    """Maintains the live + history state and renders via rich."""

    def __init__(self, console: Console | None = None):
        self._console = console
        self.title = "PROJECT PAINT DRY · sumi-e"
        self.subtitle = "bonsai narrator · live"

        # Sticky live: two buffers. streaming_line is the in-progress
        # bonsai dispatch (the typewriter source). frozen_line is the
        # most recent committed bonsai line, which keeps showing in the
        # live panel until the next dispatch starts streaming new
        # content. Live panel shows streaming if non-empty, else frozen,
        # else just the cursor glyph.
        self.streaming_line: str = ""
        self.frozen_line: str = ""
        # Timestamp at which the most recent dispatch finished streaming.
        # Used to drive the slow fade from "just-arrived bright" to
        # "settled past tense" colors over _LIVE_FREEZE_FADE_S seconds
        # so the punch of completing a dispatch is visible. None when
        # nothing has been frozen yet.
        self._freeze_started_at: float | None = None

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

        # Coupled-oscillator phase state for the default-cycle history
        # layers. Each visible layer slot has its own slightly
        # perturbed period; weak Kuramoto coupling keeps the inter-
        # layer offsets bounded so the stack visibly orbits the ideal
        # configuration without smearing into noise. Advanced once
        # per render() call. The most-recently-committed entry uses
        # the legacy fast cycle and bypasses this state.
        self._shimmer_phases = ShimmerPhaseState(
            num_layers=_VISIBLE_HISTORY_LINES,
            base_cycle_s=_SHIMMER_DEFAULT_CYCLE_S,
            layer_offset=_SHIMMER_LAYER_OFFSET,
        )
        self._last_phase_update_s: float | None = None

        # Running counters
        self.stat_emitted = 0
        self.stat_dropped_dedup = 0
        self.stat_dropped_empty = 0
        # End-of-run wrap-up (color commentary)
        self.wrap_up_text: str = ""
        # When wrap-up generation is in flight (between
        # start_wrap_up() and the actual wrap_up text arriving),
        # render() shows a "writing post-game commentary..." placeholder
        # in the post-game panel so the user knows the script is alive
        # and working on the wrap-up rather than hung.
        self.wrap_up_pending: bool = False
        self.wrap_up_pending_started: float = 0.0
        # When True, render() shows the post-session footer (scroll
        # keys remain live via HistoryScrollController). The animation
        # thread keeps running so the shimmer plays on while the
        # operator inspects the final state.
        self.session_ended: bool = False

        # In-pane history scroll viewport. Persistent across frames so
        # that `HistoryViewport.append()`'s re-anchoring logic carries
        # scroll state correctly when new history arrives while the
        # operator is scrolled up. Rebuilt only when the wrap width
        # changes (terminal resize) — scroll offset is preserved
        # across rebuilds. The viewport consumes a separately computed
        # flat list with essentials-first priority applied, so current
        # live-edge semantics are preserved (see
        # `_flat_display_entries`).
        self._viewport: HistoryViewport | None = None
        self._viewport_wrap_width: int | None = None
        # Number of flat entries already appended into `_viewport`, so
        # lazy sync can feed only the delta on each access.
        self._viewport_synced_len: int = 0
        # Optional explicit wrap width override for tests and for the
        # public viewport accessor when no console is attached. The
        # renderer still prefers `_compute_wrap_width()` from the live
        # console when present.
        self._wrap_width_override: int | None = None

    def __rich__(self) -> Group:
        return self.render()

    # -- History viewport (Crispy Drips) --------------------------------

    def _flat_display_entries(
        self,
    ) -> list[tuple[tuple[str, str, int | None], bool]]:
        """Return the priority-filtered flat list of history entries in
        natural chronological (oldest -> newest) order, with a boolean
        marking the most-recently-committed entry.

        Applies the essentials-first priority rule: headers and topics
        for every item always survive, and optional narrator lines
        are kept newest-first only until the visible-row budget
        (`_VISIBLE_HISTORY_LINES`) is exhausted. This preserves the
        legacy live-edge trim semantics — a long
        chatty item cannot push older items' headers/topics off the
        pane.

        Trade-off: narrator lines dropped here are also not available
        to scroll back to. That is the slice-2 compromise: the
        viewport wiring lands without changing live-edge behavior,
        and a follow-on can promote priority awareness INTO the
        viewport so scrolling up reveals hidden narrator lines too.
        """
        history_list = list(self.history)
        if not history_list:
            return []
        most_recent_idx = len(history_list) - 1

        # Group into items so priority fill can reason in item order.
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

        # Walk groups newest-first so "keep essentials newest-first
        # under budget" is deterministic, then restore natural order
        # before returning.
        groups_newest_first = list(reversed(groups))
        flat_newest_first: list[tuple[tuple, int]] = []
        for g in groups_newest_first:
            flat_newest_first.extend(g)

        budget = _VISIBLE_HISTORY_LINES
        keep_positions: set[int] = set()

        # Pass 1: essentials (headers + topics) in newest-first order.
        for pos, (entry, _idx) in enumerate(flat_newest_first):
            if entry[0] in ("header", "topic"):
                if len(keep_positions) >= budget:
                    break
                keep_positions.add(pos)

        # Pass 2: narrator lines, sorted by recency, filling what's
        # left of the budget.
        optionals = [
            (pos, entry, idx)
            for pos, (entry, idx) in enumerate(flat_newest_first)
            if entry[0] not in ("header", "topic")
        ]
        optionals.sort(key=lambda t: -t[2])  # newest first
        for pos, _entry, _idx in optionals:
            if len(keep_positions) >= budget:
                break
            keep_positions.add(pos)

        # Rebuild in natural (chronological) deque order so the
        # viewport sees entries oldest -> newest and can anchor its
        # "newest visual row" correctly.
        kept_by_idx = {
            flat_newest_first[pos][1]: flat_newest_first[pos][0]
            for pos in keep_positions
        }
        out: list[tuple[tuple[str, str, int | None], bool]] = []
        for idx in sorted(kept_by_idx):
            out.append((kept_by_idx[idx], idx == most_recent_idx))
        return out

    def _resolve_wrap_width(self) -> int:
        wrap = self._compute_wrap_width()
        if wrap is None:
            wrap = self._wrap_width_override or 80
        return wrap

    def _sync_viewport(self) -> HistoryViewport:
        """Ensure `self._viewport` reflects current history + wrap width.

        * Rebuild from scratch on first call, when the wrap width
          changes (terminal resize), or when the priority-filtered
          flat list diverges from what the viewport already contains.
        * Otherwise append only the delta so
          `HistoryViewport.append()`'s re-anchoring logic carries
          scroll state across new commits.
        """
        wrap_width = self._resolve_wrap_width()
        flat = self._flat_display_entries()
        flat_entries = [entry for entry, _ in flat]

        rebuild = (
            self._viewport is None
            or self._viewport_wrap_width != wrap_width
        )

        if not rebuild and self._viewport is not None:
            # Verify the prefix the viewport already holds still
            # matches the current priority-filtered flat list. If an
            # entry was filtered out (e.g. a narrator line dropped by
            # priority because newer lines took its budget slot), the
            # prefix diverges and we have to rebuild.
            current = self._viewport.entries_snapshot()
            prefix_len = min(len(current), len(flat_entries))
            if current[:prefix_len] != flat_entries[:prefix_len]:
                rebuild = True
            elif len(current) > len(flat_entries):
                rebuild = True

        if rebuild:
            held_offset = (
                self._viewport.scroll_offset if self._viewport is not None else 0
            )
            self._viewport = HistoryViewport(
                visible_rows=_VISIBLE_HISTORY_LINES,
                wrap_width=wrap_width,
            )
            self._viewport_wrap_width = wrap_width
            for entry in flat_entries:
                self._viewport.append(entry)
            self._viewport_synced_len = len(flat_entries)
            if held_offset > 0:
                self._viewport.scroll_up(held_offset)
        else:
            assert self._viewport is not None
            # Feed only the new entries so append()'s re-anchoring
            # logic carries scroll state forward.
            new_entries = flat_entries[self._viewport_synced_len :]
            for entry in new_entries:
                self._viewport.append(entry)
            self._viewport_synced_len = len(flat_entries)
        return self._viewport

    def history_viewport(self) -> HistoryViewport:
        """Public accessor: returns the synced viewport. Used by the
        render path and by the interactive input loop (slice 3)."""
        return self._sync_viewport()

    def _viewport_display_entries(
        self,
    ) -> list[tuple[tuple[str, str, int | None], bool]]:
        """Return the render-facing display entries for the history
        pane: the viewport's currently visible slice, reverse-grouped
        so the newest item's group sits at the top (natural Paint Dry
        layout), with entries within each group kept in chronological
        order so a header sits above its own narrator lines.
        """
        vp = self._sync_viewport()
        visible = vp.visible_entries()
        if not visible:
            return []

        # Identify the most-recently-committed entry by matching
        # against the last element of `self.history` (if any). This
        # preserves the fast-cycle shimmer for the newest entry even
        # when we're scrolled up and it's not in the visible slice.
        history_list = list(self.history)
        most_recent_entry = history_list[-1] if history_list else None

        # Group the visible entries at header boundaries.
        groups: list[list[tuple[str, str, int | None]]] = []
        current_group: list[tuple[str, str, int | None]] = []
        for entry in visible:
            if entry[0] == "header":
                if current_group:
                    groups.append(current_group)
                current_group = [entry]
            else:
                current_group.append(entry)
        if current_group:
            groups.append(current_group)

        # Newest group on top.
        groups.reverse()
        out: list[tuple[tuple[str, str, int | None], bool]] = []
        for group in groups:
            for entry in group:
                is_recent = entry is most_recent_entry
                out.append((entry, is_recent))
        return out

    def scroll_history_up(self, rows: int = 1) -> None:
        self._sync_viewport().scroll_up(rows)

    def scroll_history_down(self, rows: int = 1) -> None:
        self._sync_viewport().scroll_down(rows)

    def scroll_history_to_live_edge(self) -> None:
        self._sync_viewport().scroll_to_live_edge()

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
        # Sumi-e palette (see _BASE_RGB block for the full notes).
        # Borders are dimmed indigo ink so the chrome belongs to the
        # painting instead of being neutral terminal gray. Headers
        # carry persimmon, the [item N/M] marker carries indigo, and
        # verdict topics carry celadon / vermilion / ochre.

        # Advance the coupled-oscillator phase state once per frame.
        # All _apply_shimmer calls in this render pass will read from
        # the same advanced snapshot.
        now = time.monotonic()
        if self._last_phase_update_s is None:
            dt = 0.0
        else:
            dt = now - self._last_phase_update_s
            # Cap dt at 1s to absorb pauses (terminal hidden, debugger
            # break, etc.) without injecting a huge transient that
            # would knock layers off the ideal stack.
            if dt > 1.0:
                dt = 1.0
        self._shimmer_phases.advance(dt)
        self._last_phase_update_s = now

        # Compute the wrap width once — used by both the live panel
        # shimmer and the history panel shimmer.
        wrap_width = self._compute_wrap_width()

        # Header — title + running stats. Muted, single line.
        # The subtitle is in cool steel blue — always-on cool note in
        # the top chrome to balance the warm field below. Same color
        # family as the [item N/M] index markers in the history panel.
        header_text = Text()
        header_text.append(self.title, style="bold bright_white")
        header_text.append("   ", style="dim")
        header_text.append(self.subtitle, style="#5a73b4")
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
            border_style="#3d4458",
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

        # Tail-truncate the live content so it always fits in
        # _LIVE_PANEL_CONTENT_LINES of visual rows. The panel itself
        # has a fixed height so the layout doesn't jitter when bonsai
        # produces a long line that wraps. We track char_offset and
        # pass it to the undulating renderer so the per-character
        # color phase stays stable across truncation.
        live_char_offset = 0
        if displayed_live and wrap_width and wrap_width > 4:
            # Reserve the cursor glyph (2 cols) on the first row only.
            # Total visible chars = first row width + (N-1) * full width
            # Be conservative: use full rows for budget so we don't
            # over-pack and get a 4th row.
            live_max_chars = (
                _LIVE_PANEL_CONTENT_LINES * wrap_width
            ) - 6  # safety margin
            if len(displayed_live) > live_max_chars:
                # Drop chars from the front, mark with leading ellipsis.
                drop = len(displayed_live) - live_max_chars + 1
                live_char_offset = drop
                displayed_live = "…" + displayed_live[drop:]

        if displayed_live:
            live_text = Text(no_wrap=False, overflow="fold")
            cursor_style = "bright_cyan" if is_active else "grey50"
            live_text.append("▌ ", style=cursor_style)
            # Compute freeze age for the renderer's fade. Only meaningful
            # when not actively streaming and we have a recorded freeze
            # timestamp; otherwise the renderer treats sat/val as fully
            # bright.
            freeze_age_s = None
            if not is_active and self._freeze_started_at is not None:
                freeze_age_s = time.monotonic() - self._freeze_started_at
            _render_live_undulating(
                live_text, displayed_live,
                indent_width=2,  # cursor glyph "▌ "
                wrap_width=wrap_width,
                is_active=is_active,
                char_offset=live_char_offset,
                freeze_age_s=freeze_age_s,
            )
        else:
            live_text = Text("▌ ", style="grey39", overflow="fold")
        live_panel = Panel(
            live_text,
            border_style="#3d4458",
            padding=(0, 1),
            title="[grey50]live[/grey50]",
            title_align="left",
            # Fixed height: top border + content + bottom border.
            # Locks the live panel's vertical footprint so the layout
            # doesn't jitter when bonsai produces a long line.
            height=_LIVE_PANEL_CONTENT_LINES + 2,
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
        display_entries = self._viewport_display_entries()
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

            # Coupled phase state is for the default-cycle stack only.
            # The most-recent entry uses the legacy fast-cycle phase
            # (computed inside _apply_shimmer from time.monotonic()).
            phase_override = (
                None
                if is_most_recent
                else self._shimmer_phases.phase(i)
            )

            if kind == "header":
                indent = "─ "
                # The leading `─ ` dash now pulses in vermilion as a
                # "header_dash" kind, sharing the same layer phase and
                # cycle as the rest of the header so the dash + index +
                # title read as a single coordinated stroke. Indent
                # width is 0 here because the dash IS the leading edge.
                _apply_shimmer(
                    history_text, indent, "header_dash",
                    layer_index=i,
                    indent_width=0,
                    wrap_width=wrap_width,
                    cycle_s=entry_cycle,
                )
                # Split the [item N/M] index marker from the rest of
                # the title so the index can be rendered in cool steel
                # blue (always-on cool accent) while the rest stays in
                # warm orange3. Both halves get the same shimmer
                # rhythm — same layer_index, same cycle, same head
                # position via correct indent_width, AND the same
                # coupled-oscillator phase override — so they pulse
                # together but pulse FROM different base colors.
                m = _HEADER_INDEX_RE.match(text)
                if m:
                    index_part = m.group(1)
                    rest_part = m.group(2)
                    _apply_shimmer(
                        history_text, index_part, "header_index",
                        layer_index=i,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                    history_text.append(" ", style="grey39")
                    _apply_shimmer(
                        history_text, rest_part, "header",
                        layer_index=i,
                        indent_width=len(indent) + len(index_part) + 1,
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                else:
                    _apply_shimmer(
                        history_text, text, "header",
                        layer_index=i,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
            elif kind == "topic":
                indent = "  · "
                history_text.append(indent, style="grey50")
                # Pick the topic color variant based on the stored
                # verdict (third tuple slot, named "parity" for line
                # entries but reused as the verdict string for topic
                # entries). Cool sage for matches, warm coral for
                # grader-overshot, warm amber for grader-undershot,
                # plain plum fallback when verdict is unknown.
                topic_kind = {
                    "match": "topic_match",
                    "overshoot": "topic_overshoot",
                    "undershoot": "topic_undershoot",
                }.get(parity, "topic")
                # Pull out the elapsed-time prefix and color it as a
                # punchy warm accent (bold orange3 — same warm as the
                # post-game border and header base) so it pops out
                # from the rest of the line.
                m = _TIME_PREFIX_RE.match(text)
                if m:
                    time_prefix, rest = m.group(1), m.group(2)
                    history_text.append(time_prefix, style="bold #d86324")
                    history_text.append("  ·  ", style="grey50")
                    extra_indent = len(time_prefix) + len("  ·  ")
                    _apply_shimmer(
                        history_text, rest, topic_kind,
                        layer_index=i,
                        indent_width=len(indent) + extra_indent,
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                else:
                    _apply_shimmer(
                        history_text, text, topic_kind,
                        layer_index=i,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
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
                    phase_override=phase_override,
                )

        if not display_entries:
            history_text = Text(
                "(waiting for first summary...)", style="grey39"
            )

        history_panel = Panel(
            history_text,
            border_style="#3d4458",
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
                border_style="#d86324",
                padding=(0, 1),
                title="[bold #d86324]post-game[/bold #d86324]",
                title_align="left",
            )
        elif self.wrap_up_pending:
            # Placeholder while wrap-up generation is in flight.
            # Live elapsed counter so the user sees the script is alive.
            elapsed = time.monotonic() - self.wrap_up_pending_started
            wrap_text = Text(
                f"writing post-game commentary... ({elapsed:.0f}s)",
                style="grey50 italic",
                no_wrap=False,
                overflow="fold",
            )
            wrap_panel = Panel(
                wrap_text,
                border_style="#3d4458",
                padding=(0, 1),
                title="[grey50]post-game · pending[/grey50]",
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
                "  ▌ session ended — k/j scroll, u/d page, 0 live edge, any other key closes ▐",
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
            # Mark the start of the freeze fade — only when there was
            # actual content to freeze. Empty commits don't restart
            # the fade clock.
            self._freeze_started_at = time.monotonic()
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

    def on_wrap_up_pending(self) -> None:
        """Wrap-up generation has started — show placeholder until the
        actual text arrives. Records the timestamp so the placeholder
        can show elapsed time as a liveness signal."""
        self.wrap_up_pending = True
        self.wrap_up_pending_started = time.monotonic()

    def on_wrap_up(self, text: str) -> None:
        """Final post-game commentary from bonsai."""
        self.wrap_up_text = text
        self.wrap_up_pending = False

    def on_topic(self, text: str, verdict: str | None = None) -> None:
        # Topic (after-action) lands in history. Doesn't touch live
        # buffers — frozen_line keeps showing the last bonsai line.
        # The verdict ("match" / "overshoot" / "undershoot" / None)
        # is stored in the third tuple slot so the renderer can pick
        # the right color variant for the topic line.
        self.history.append(("topic", text, verdict))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: narrator_reader.py <fifo-path>", file=sys.stderr)
        return 2

    fifo_path = Path(sys.argv[1])
    if not fifo_path.exists():
        print(f"fifo not found: {fifo_path}", file=sys.stderr)
        return 2

    # Force truecolor mode. The narrator window is spawned by the
    # narrator sink into a fresh WezTerm window, which supports
    # 24-bit color escape codes — required for the sumi-e palette
    # to render with the precision the design assumes (smooth
    # shimmer interpolation across 12+ unique cells per char,
    # distinguishable persimmon vs orange3, indigo vs steel blue,
    # etc.). Without `force_terminal`, Rich would auto-detect
    # `is_terminal=False` on a piped/spawned process and fall back
    # to no-color output.
    console = Console(color_system="truecolor", force_terminal=True)
    display = PaintDryDisplay(console=console)

    # Open the fifo for reading. This blocks until the writer connects.
    fd = os.open(str(fifo_path), os.O_RDONLY)
    fifo = os.fdopen(fd, "r", buffering=1)

    # Install cbreak mode on stdin so the interactive scroll loop can
    # read one byte at a time without waiting for Enter. Best-effort:
    # if stdin is not a TTY (e.g. the reader is smoke-tested from a
    # pipe), skip the terminal plumbing and the scroll thread below.
    # Raw-mode state is restored in the `finally` block so the user's
    # shell is never left in a broken state on exit.
    stdin_fd: int | None = None
    saved_termios = None
    try:
        stdin_fd = sys.stdin.fileno()
        saved_termios = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
    except (termios.error, OSError, ValueError):
        stdin_fd = None
        saved_termios = None

    try:
        # auto_refresh=False — we drive the render manually from a
        # background timer thread. Tried auto_refresh=True with
        # display.__rich__ but rich's diff layer was treating
        # shimmer phase changes as no-op (because the underlying
        # text content was identical between frames, only the
        # per-character styles changed). Manual update + force
        # refresh works reliably.
        animation_stop = threading.Event()
        session_exit = threading.Event()
        scroll_controller = HistoryScrollController(display)

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

            # Interactive scroll input. Runs only when stdin is a
            # real TTY in cbreak mode. Reads one byte at a time and
            # forwards it to the scroll controller. Before session
            # end, unbound keys are swallowed (to avoid accidental
            # exits). After session end, unbound keys trigger the
            # session exit signal — any keystroke closes the reader.
            scroll_stop = threading.Event()

            def _scroll_tick():
                while not scroll_stop.is_set():
                    try:
                        ch = sys.stdin.read(1)
                    except Exception:
                        return
                    if not ch:
                        return
                    handled = scroll_controller.handle_key(ch)
                    if not handled and display.session_ended:
                        session_exit.set()
                        return

            scroll_thread: threading.Thread | None = None
            if stdin_fd is not None:
                scroll_thread = threading.Thread(
                    target=_scroll_tick,
                    name="paint-dry-scroll",
                    daemon=True,
                )
                scroll_thread.start()

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
                        display.on_topic(
                            msg.get("text", ""),
                            verdict=msg.get("verdict"),
                        )
                    elif msg_type == "drop":
                        display.on_drop(
                            msg.get("reason", "unknown"),
                            msg.get("text", ""),
                        )
                    elif msg_type == "wrap_up_pending":
                        display.on_wrap_up_pending()
                    elif msg_type == "wrap_up":
                        display.on_wrap_up(msg.get("text", ""))
                    elif msg_type == "end":
                        # Flag the display so render() shows a
                        # "press any key to close" footer. Keep
                        # the animation thread running so the
                        # shimmer continues to play while the
                        # user reads the final state. The scroll
                        # thread stays alive so the operator can
                        # still scroll the history pane after
                        # session end — any non-scroll key will
                        # fire `session_exit` and close the reader.
                        display.session_ended = True
                        if stdin_fd is None:
                            # No interactive input available
                            # (non-TTY stdin); exit immediately.
                            session_exit.set()
                        session_exit.wait()
                        animation_stop.set()
                        scroll_stop.set()
                        anim_thread.join(timeout=0.5)
                        return 0
    finally:
        animation_stop.set()
        try:
            scroll_stop.set()  # type: ignore[name-defined]
        except NameError:
            pass
        try:
            fifo.close()
        except Exception:
            pass
        if stdin_fd is not None and saved_termios is not None:
            try:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, saved_termios)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
