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

import base64
import json
import math
import os
import re
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

import fitz
from rich.align import Align
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.segment import ControlType, Segment
from rich.style import Style
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
_VISIBLE_HISTORY_ROWS = 30  # visible history budget in WRAPPED visual rows,
                            # not logical entries. Keep the old overall
                            # depth, but count it coherently now that the
                            # scorebug and long wrapped lines exist.
_VISIBLE_DROP_LINES = 4
_FOCUS_PREVIEW_MIN_WIDTH_CHARS = 54
_FOCUS_PREVIEW_MAX_WIDTH_CHARS = 116
_FOCUS_PREVIEW_MIN_HEIGHT_ROWS = 18
_FOCUS_PREVIEW_MAX_HEIGHT_ROWS = 30
_FOCUS_PREVIEW_COMPANION_SCALE = 0.69
_FOCUS_PREVIEW_PENDING_FPS = 8.0
_FOCUS_PREVIEW_BG_RGB = (8, 10, 14)
_FOCUS_PREVIEW_PAPER_RGB = (204, 196, 186)  # used only by the transition
                                             # (pending) glyph overlay; the
                                             # steady-state renderer uses
                                             # the harder colors below
# Legibility-first steady-state palette. High luminance delta against the
# panel background so binary-thresholded cells read as page, not as mush.
# Aesthetics are explicitly deferred — pick whatever reads cleanest first.
_FOCUS_PREVIEW_HARD_INK_RGB = (50, 54, 62)
_FOCUS_PREVIEW_HARD_PAPER_RGB = (238, 232, 220)
_FOCUS_PREVIEW_OVERLAY_CHARS = "0011/."
_FOCUS_PREVIEW_OVERLAY_RGBS = (
    (108, 122, 154),
    (132, 115, 86),
    (94, 116, 106),
)
_HISTORY_TIER_DIM_FLOOR_DEPTH = 9  # the within-item fade should keep
                                   # descending deeper into the stack before
                                   # it settles at the floor.
_HISTORY_TIER_DIM_EASE_POWER = 1.72  # fast initial drop, then a slower tail
                                     # instead of a purely linear ramp.

# Shimmer parameters — slow chyron sweep across the top N history lines.
# Each layer has a fixed phase offset relative to the one above it (so
# they're in stable orbit, not drifting), and intensity decays with
# layer position so older lines pulse dimmer than newer ones.
_SHIMMER_DEFAULT_CYCLE_S = 3.2  # eased back slightly as redraw cadence rises,
                                # so the calmer field stays calm instead of
                                # feeling busier at 24 FPS
_SHIMMER_RECENT_CYCLE_S = 1.35  # still the quickest history motion, but
                                # a touch less twitchy under the smoother redraw
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
_HISTORY_TIER_DIM_MIN = 0.58    # floor for within-item fade.
_HISTORY_GROUP_DIM_STEP = 0.05  # each successive thought line under a header
                                 # should visibly dim, but the fade should
                                 # take longer to settle so deeper within-item
                                 # stacks still read as a gradient instead of
                                 # flattening by line 6.

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
    "header": (186, 82, 52),     # lacquered persimmon red — darker and
                                  # redder than the earlier orange pass,
                                  # so structural titles read like warm
                                  # lacquer instead of pumpkin glow
    "header_index": (90, 115, 180),    # indigo (藍色) — the [item N/M]
                                       # marker carries the cool axis
                                       # of the painting
    "live": (245, 240, 225),     # rice paper — warm off-white for the
                                  # live field, the brightest bone
                                  # surface in the composition
    "status": (96, 64, 38),      # dark coal-ember umber — persistent status
                                  # rail, pushed a step deeper so it feels
                                  # less rosy and more like banked heat
                                  # under ash
    # Topic verdict variants — full-saturation garden colors. The
    # narration rows above use desaturated cousins of these, so the
    # eye reads "muted family below, vivid accent here" and the
    # verdict still encodes meaning at a glance.
    "topic_match": (70, 92, 156),         # deep indigo agreement —
                                          # darker than the header-index
                                          # blue so it harmonizes with
                                          # structure without duplicating it
    "checkpoint": (138, 156, 142),        # anchored moss checkpoint —
                                          # checkpoints should feel like
                                          # compressed descendants of the
                                          # live history rows, not a separate
                                          # steel annotation layer
    "checkpoint_alt": (178, 162, 132),    # anchored bone-earth checkpoint —
                                          # alternating companion to the
                                          # moss checkpoint tone so durable
                                          # history keeps the familiar
                                          # moss/bone cadence
    "checkpoint_mark": (162, 114, 82),    # embered rust notch — structural
                                          # mark for checkpoint rows so the
                                          # checkpoint doesn't begin with a
                                          # dead grey gutter
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
    "checkpoint": 0.92,
    "checkpoint_alt": 0.92,
    "checkpoint_mark": 0.96,
    "status": 1.15,
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
    "header": (232, 136, 102),    # fired lacquer red — brighter crest,
                                   # but still clearly red-led rather than
                                   # tipping back into orange
    "header_index": (185, 210, 240),  # rain-cleared sky blue — indigo
                                       # brightens toward the pale sky
                                       # after a storm wash painting
    "line": (175, 215, 180),      # glazed celadon — sage moss row
                                   # brightens toward kiln-glaze green
    "line_alt": (225, 200, 150),  # fired ochre — dust earth row
                                   # brightens toward kiln-fired earth
    "topic_match": (132, 160, 224),     # rain-lit deep-indigo crest for
                                        # agreement lines
    "checkpoint": (176, 204, 180),      # brighter celadon crest —
                                        # still in the history family, just
                                        # a touch more settled than live rows
    "checkpoint_alt": (222, 198, 150),  # brighter bone-earth crest for the
                                        # alternating checkpoint lane
    "checkpoint_mark": (226, 166, 114), # brighter ember crest for the
                                        # checkpoint mark, tied to the
                                        # header/status warm structure
    "status": (188, 118, 68),           # ember-lit umber crest for the
                                        # sticky status rail — brighter
                                        # orange note without losing the
                                        # darker coal base
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
    "checkpoint",
    "checkpoint_alt",
    "checkpoint_mark",
    "status",
    "topic_overshoot",
    "topic_undershoot",
})

# Live panel reserves a fixed vertical footprint so it doesn't jitter
# the layout when bonsai produces a long line that wraps. The panel
# always shows _LIVE_PANEL_CONTENT_LINES rows of content (plus the
# top + bottom borders). When bonsai's output is longer than will
# fit in that area, we tail-truncate (keep the most recent chars).
_LIVE_PANEL_CONTENT_LINES = 3
_TOP_PANEL_CONTENT_LINES = _LIVE_PANEL_CONTENT_LINES + 1

# Live-line undulation parameters — each character on the live line
# gets a per-position, per-time hue from a warm orange-amber palette.
# Adjacent characters have slightly different hues (per-char phase
# offset) and the whole field undulates over a slow cycle.
# Pulled toward orange (away from yellow) and slightly desaturated
# from the previous values to harmonize with the rest of the sunset
# palette without losing fire feel.
_LIVE_UNDULATION_CYCLE_S = 3.8    # lively enough to read as motion, but
                                   # still slower than token streaming
_LIVE_HUE_CENTER_DEG = 196         # pulled slightly toward mossy aqua so the
                                   # cool lane keeps more green body
_LIVE_HUE_RANGE_DEG = 22           # broader swing so the green note is
                                   # visibly present instead of incidental
_LIVE_PER_CHAR_PHASE_OFFSET = 0.18 # phase shift per character (radians)
_LIVE_PHASE_OFFSET_RAD = 0.0
_LIVE_UNDULATION_DIRECTION = -1.0  # move slowly left, against the main
                                    # shimmer sweep, so the top band feels
                                    # like its own counter-current
_LIVE_BASE_SAT = 0.34              # still soft, but with a more visible wash
_LIVE_BASE_VAL = 0.86              # slightly less white, a little more pigment
_LIVE_WARM_HUE_CENTER_DEG = 22     # yellow-red sibling, friendlier than a hot
                                   # alarm band but more chromatic than before
_LIVE_WARM_HUE_RANGE_DEG = 18
_LIVE_WARM_BASE_SAT = 0.30
_LIVE_WARM_BASE_VAL = 0.87
_LIVE_WARM_LUMINANCE_CORRECTION_STRENGTH = 0.30
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
_LIVE_LUMINANCE_CORRECTION_STRENGTH = 0.45
_STATUS_UNDULATION_CYCLE_S = 6.0   # still slower than live, and eased back a
                                    # bit after the redraw-rate bump
_STATUS_HUE_CENTER_DEG = 22         # shifted away from hot red toward
                                    # dark ember-orange / umber
_STATUS_HUE_RANGE_DEG = 8           # narrower swing so the status rail
                                    # keeps its heavier umber weight
_STATUS_PER_CHAR_PHASE_OFFSET = 0.14
_STATUS_PHASE_OFFSET_RAD = 0.85     # keep status related to live, but out of
                                    # lockstep so they do not breathe as one
_STATUS_UNDULATION_DIRECTION = 1.0
_STATUS_BASE_SAT = 0.66
_STATUS_BASE_VAL = 0.58
_STATUS_LUMINANCE_CORRECTION_STRENGTH = 0.55
_STATUS_COOL_GLINT_RGB = (70, 108, 184)   # restrained deep-indigo glint
                                           # inside the ember rail
_STATUS_BONE_GLINT_RGB = (224, 210, 190)   # pale bone lift so the rail
                                           # can briefly catch ash-light
_STATUS_COOL_GLINT_CYCLE_S = 7.8
_STATUS_BONE_GLINT_CYCLE_S = 9.6
_STATUS_COOL_GLINT_STRENGTH = 0.86
_STATUS_BONE_GLINT_STRENGTH = 0.44
_STATUS_COOL_GLINT_PHASE_OFFSET_RAD = 1.55
_STATUS_BONE_GLINT_PHASE_OFFSET_RAD = 1.10
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
_ACTIVE_ANIMATION_FPS = 24.0  # smoother motion without changing the protocol;
                              # the slower animation families above are eased
                              # back to keep the overall feel restrained
_IDLE_POLL_S = 0.20           # static state still needs to pick up new fifo
                              # messages quickly, but doesn't need redraw spam
_SESSION_END_ANIMATION_LINGER_S = 120.0  # keep the finished painting alive
                                         # and animated for a while before
                                         # letting it settle
_LIVE_PLACEHOLDER_ROTATE_S = 6.0
_LIVE_PLACEHOLDER_OPTIONS = (
    "thinking through the tape...",
    "review booth is checking the work...",
    "grading engine warming up...",
    "calling for the next replay...",
    "waiting on the next chain-of-thought...",
    "running the numbers upstairs...",
)
# Shimmer peak — what each character's color is interpolated toward
# at the shimmer head. Pale moonlit gold (the highlight on a brush
# stroke as the wash dries), so the wave reads as a quiet brightening
# of the ink rather than a fire sweep.
_SHIMMER_PEAK_RGB = (235, 215, 175)
_EMBER_ACCENT_RGB = (232, 136, 102)  # the lighter orange note used where
                                     # we want warm structural emphasis
                                     # without a full verdict signal

_SCOREBUG_BIG_DIGITS = {
    "0": ("╔═╗", "║ ║", "╚═╝"),
    "1": ("╔╗ ", " ║ ", " ║ "),
    "2": ("╔═╗", "╔═╝", "╚═ "),
    "3": ("╔═╗", " ═╣", "╚═╝"),
    "4": ("║ ║", "╚═╣", "  ║"),
    "5": ("╔═ ", "╚═╗", "╚═╝"),
    "6": ("╔═ ", "╠═╗", "╚═╝"),
    "7": ("╔═╗", "╔╝ ", "║  "),
    "8": ("╔═╗", "╠═╣", "╚═╝"),
    "9": ("╔═╗", "╚═╣", "  ╝"),
    ".": ("   ", "   ", " ▪ "),
    "/": ("   ", " ╱ ", "╱  "),
    "-": ("   ", "═══", "   "),
}

_HISTORY_GROUP_SETBACK = 0.036     # lower item headers sit a bit more visibly
                                   # one above them, but not by enough to read
                                   # as separate weather systems
_HISTORY_GROUP_RAKE = 0.022        # gentler within-item rake than the last pass
                                   # so the grouping reads structural, not
                                   # algorithmically terraced
_HISTORY_GROUP_ALT_FIELD = 0.012   # subtle secondary alternating shimmer field
                                   # shared across even/odd item groups
_HISTORY_GROUP_ALT_RATE = 0.55     # slower than the primary history field
_HISTORY_CONTINUATION_ROW_STEP = 0.08  # wrapped continuation rows should step
                                       # down in authority below the first
                                       # visual row of an entry
_HISTORY_CONTINUATION_ROW_MIN = 0.78   # keep deeper wrapped rows visible, but
                                       # clearly subordinate to the first row


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


def _blend_rgb(
    base: tuple[int, int, int],
    target: tuple[int, int, int],
    weight: float,
) -> tuple[int, int, int]:
    """Blend base toward target by weight in [0, 1]."""
    weight = max(0.0, min(1.0, weight))
    return tuple(
        max(
            0,
            min(
                255,
                int(round(channel + (target_channel - channel) * weight)),
            ),
        )
        for channel, target_channel in zip(base, target, strict=True)
    )


def _history_group_phase(
    base_phase: float,
    secondary_phase: float,
    group_index: int,
) -> float:
    """Set back each visible item group and add a subtle parity field.

    The history stack should feel coherent within an item, but item
    boundaries should not all lie on the exact same shimmer plane.
    This helper keeps one local field per item, sets lower headers
    slightly back from the one above them, and layers in a faint
    alternating parity field so neighboring groups do not ride the
    same exact shimmer geometry.
    """
    parity_direction = -1.0 if group_index % 2 else 1.0
    alternating_offset = (
        math.sin(
            2
            * math.pi
            * ((secondary_phase - 0.25) * _HISTORY_GROUP_ALT_RATE)
        )
        * _HISTORY_GROUP_ALT_FIELD
        * parity_direction
    )
    return (
        base_phase
        - (group_index * _HISTORY_GROUP_SETBACK)
        + alternating_offset
    ) % 1.0


def _history_entry_phase(
    base_phase: float,
    secondary_phase: float,
    group_index: int,
    group_depth: int,
) -> float:
    """Phase for one visible history entry.

    Item headers sit slightly behind the one above them, but entries
    within an item still rake back enough to read as a local field.
    A faint alternating parity field reinforces the item boundaries
    without making the geometry feel mechanically terraced.
    """
    group_phase = _history_group_phase(base_phase, secondary_phase, group_index)
    return (group_phase - (group_depth * _HISTORY_GROUP_RAKE)) % 1.0


def _scorebug_big_value_rows(value: str) -> tuple[str, str, str]:
    top_parts: list[str] = []
    middle_parts: list[str] = []
    bottom_parts: list[str] = []
    for ch in value:
        top, middle, bottom = _SCOREBUG_BIG_DIGITS.get(
            ch,
            ("   ", f" {ch} ", "   "),
        )
        top_parts.append(top)
        middle_parts.append(middle)
        bottom_parts.append(bottom)
    return (
        "".join(top_parts),
        "".join(middle_parts),
        "".join(bottom_parts),
    )


def _append_scorebug_value_row(
    row: Text,
    content: str,
    *,
    strong_style: str,
    mid_style: str,
) -> None:
    """Append one scoreboard value row with weighted two-tone strokes."""
    strong_chars = {"╔", "╗", "╚", "╝", "║", "╠", "╣", "╩", "═", "▪"}
    mid_chars = {"╱", " "}
    for ch in content:
        if ch in strong_chars:
            row.append(ch, style=strong_style)
        elif ch in mid_chars:
            row.append(ch, style=mid_style)
        else:
            row.append(ch, style=strong_style)


def _live_placeholder(now_s: float) -> str:
    idx = int(now_s // _LIVE_PLACEHOLDER_ROTATE_S) % len(_LIVE_PLACEHOLDER_OPTIONS)
    return _LIVE_PLACEHOLDER_OPTIONS[idx]


def _scale_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale an RGB triple by factor, preserving channel bounds."""
    factor = max(0.0, factor)
    return tuple(
        max(0, min(255, int(round(channel * factor))))
        for channel in rgb
    )


def _history_tier_dim_factor(layer_index: int) -> float:
    """Return the brightness factor for a line within an item group.

    This is intentionally local to the current header block, not the
    whole viewport. Each step down within an item should be visibly
    dimmer, then clamp at the floor so deep blocks don't disappear.
    """
    if layer_index <= 0:
        return 1.0
    if layer_index >= _HISTORY_TIER_DIM_FLOOR_DEPTH:
        return _HISTORY_TIER_DIM_MIN
    t = layer_index / _HISTORY_TIER_DIM_FLOOR_DEPTH
    eased = (1.0 - t) ** _HISTORY_TIER_DIM_EASE_POWER
    return _HISTORY_TIER_DIM_MIN + ((1.0 - _HISTORY_TIER_DIM_MIN) * eased)


def _render_layer_index(kind: str, group_depth: int) -> int:
    """Return the effective fade layer for a history entry.

    Only narrator thought lines should sink within an item block.
    Structural lines such as headers and resolution/topic lines stay
    at full strength so the eye can keep finding the question/result
    anchors quickly.
    """
    return group_depth if kind == "line" else 0


def _message_requires_immediate_refresh(msg_type: str) -> bool:
    """Return whether a FIFO event should bypass the normal animation cadence.

    Regular stream events should let the animation loop own repaint timing so
    idle and active motion feel consistent. Only boundary moments that would
    feel laggy at 12 FPS get an immediate forced refresh.
    """
    return msg_type in {"session_meta", "focus_preview", "wrap_up", "end"}


def _focus_preview_budget(
    term_width: int | None,
    *,
    source_width_px: int | None = None,
    source_height_px: int | None = None,
) -> tuple[int, int]:
    """Return a terminal-aware preview raster budget.

    The preview should feel like a real companion surface, not a
    postage stamp. Use most of the terminal width while leaving enough
    margin that the panel still breathes, but let high-detail crops
    earn a denser raster than tiny crops on the same terminal.
    """
    if term_width is None or term_width <= 0:
        return _FOCUS_PREVIEW_MIN_WIDTH_CHARS, _FOCUS_PREVIEW_MIN_HEIGHT_ROWS
    available_width = min(
        _FOCUS_PREVIEW_MAX_WIDTH_CHARS,
        int(round((term_width - 8) * _FOCUS_PREVIEW_COMPANION_SCALE)),
    )
    detail_factor = 1.0
    if (
        source_width_px is not None
        and source_height_px is not None
        and source_width_px > 0
        and source_height_px > 0
    ):
        source_area = source_width_px * source_height_px
        detail_factor = _clamp(
            math.sqrt(source_area / float(900 * 500)),
            0.35,
            1.0,
        )
    width_chars = max(
        _FOCUS_PREVIEW_MIN_WIDTH_CHARS,
        int(
            round(
                _FOCUS_PREVIEW_MIN_WIDTH_CHARS
                + ((available_width - _FOCUS_PREVIEW_MIN_WIDTH_CHARS) * detail_factor)
            )
        ),
    )
    if (
        source_width_px is not None
        and source_height_px is not None
        and source_width_px > 0
        and source_height_px > 0
    ):
        source_aspect = source_height_px / source_width_px
        height_target = int(round(width_chars * source_aspect * 0.46))
    else:
        height_target = int(round(width_chars * 0.24))
    height_rows = max(
        _FOCUS_PREVIEW_MIN_HEIGHT_ROWS,
        min(_FOCUS_PREVIEW_MAX_HEIGHT_ROWS, height_target),
    )
    return width_chars, height_rows


# ---------------------------------------------------------------------
# Inline image rendering path (iTerm2 OSC 1337 protocol).
#
# The half-block renderer below is the fallback for terminals that
# can't do image escapes. On WezTerm and iTerm2, we take the much
# simpler and strictly-more-legible path: hand the focus preview PNG
# directly to the terminal via the iTerm2 inline image escape sequence
# and let the terminal rasterize it into screen pixels at the requested
# cell span.
#
# Integration with Rich: the inline image is wrapped in a custom Rich
# Renderable (`FocusPreviewInlineImage`) that yields the escape
# sequence plus enough blank rows to reserve the image's visual
# vertical footprint. Rich thinks it's rendering N rows of padding
# and positions the next panel below where the terminal has drawn
# the image. No Live-region splitting needed.
# ---------------------------------------------------------------------

#: Cell height the inline image path targets. Matches the half-block
#: renderer's typical panel height so the layout doesn't shift when
#: the path switches.
_INLINE_IMAGE_CELL_HEIGHT = 18

#: Max cell width the inline image path will request. Companion-scale
#: similar to the half-block renderer but a bit more generous because
#: real images tolerate larger panels aesthetically.
_INLINE_IMAGE_MAX_CELL_WIDTH = 140

#: Fallback terminal cell aspect ratio (height / width) used when
#: the terminal's real cell dimensions can't be queried via CSI
#: 16t. Most monospace fonts at common sizes fall in [2.0, 2.3].
#: The Kitty place command is aspect-preserving, so if this
#: fallback is wrong the image will letterbox inside the box —
#: tune at the deployment level or (better) ensure the terminal
#: supports CSI 16t query so we get the real value at startup.
_DEFAULT_TERMINAL_CELL_ASPECT = 2.1


def _build_iterm2_inline_image_sequence(
    png_bytes: bytes,
    *,
    cell_width: int,
    cell_height: int,
) -> str:
    """Return an iTerm2 OSC 1337 File= escape sequence that embeds the
    given PNG at the requested cell dimensions.

    Format::

        ESC ] 1337 ; File = inline=1;width=<W>;height=<H>;preserveAspectRatio=1 : <base64> BEL

    WezTerm, iTerm2, and a few other terminals render this as a real
    raster image at the requested footprint. Unsupported terminals
    will either show the sequence as garbled text or (more commonly)
    swallow it silently. Capability detection via
    `_supports_inline_images` gates whether to emit this at all.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    args = (
        f"inline=1;width={cell_width};height={cell_height};preserveAspectRatio=1"
    )
    return f"\x1b]1337;File={args}:{b64}\x07"


def _compute_inline_image_cell_dimensions(
    crop_width_px: int,
    crop_height_px: int,
    *,
    max_cell_height: int = _INLINE_IMAGE_CELL_HEIGHT,
    max_cell_width: int = _INLINE_IMAGE_MAX_CELL_WIDTH,
    terminal_cell_aspect: float = _DEFAULT_TERMINAL_CELL_ASPECT,
) -> tuple[int, int]:
    """Compute the (cell_width, cell_height) footprint for an inline
    image given its source pixel dimensions and the terminal's
    real cell-pixel aspect ratio.

    Strategy: start from max_cell_height, derive cell_width from
    the image aspect and cell aspect. If cell_width exceeds
    max_cell_width, clamp it and shrink cell_height proportionally
    so the image tight-fits the clamped width.

    ``terminal_cell_aspect`` is ``cell_height_px / cell_width_px``
    of the terminal's font as rendered on screen. For typical
    monospace fonts this is around 2.0-2.3. Pass the queried
    value from ``_query_terminal_cell_aspect`` for accuracy, or
    the default constant as a fallback.
    """
    if crop_width_px <= 0 or crop_height_px <= 0:
        return (max(1, max_cell_width), max(1, max_cell_height))
    crop_aspect = crop_width_px / crop_height_px
    cell_height = max(1, max_cell_height)
    # cell_width / cell_height = image_aspect × terminal_cell_aspect
    # derivation: for the box to tight-fit the image in screen pixels,
    #   (cell_width × cell_px_w) / (cell_height × cell_px_h) == image_aspect
    #   (cell_width / cell_height) × (1 / terminal_cell_aspect) == image_aspect
    #   cell_width / cell_height == image_aspect × terminal_cell_aspect
    cell_width = int(round(cell_height * crop_aspect * terminal_cell_aspect))
    if cell_width > max_cell_width:
        shrink = max_cell_width / cell_width
        cell_width = max_cell_width
        cell_height = max(1, int(round(cell_height * shrink)))
    cell_width = max(1, cell_width)
    return (cell_width, cell_height)


# ---------------------------------------------------------------------
# Kitty graphics protocol path (the durable fix for the flicker).
#
# The iTerm2 OSC 1337 path above has a fundamental problem: Rich's
# Live display clears the image region between frames (via CSI 2K),
# so we have to re-emit the full base64 PNG on every frame to keep
# the image visible. WezTerm then re-parses the PNG 24 times per
# second and the operator sees seizure-grade strobing.
#
# The Kitty graphics protocol has a native solution: upload the PNG
# once with a numeric image ID (via APC ESC_G with no action key and
# chunked base64 payload), then reference it on subsequent frames
# with tiny `a=p,i=<id>,c=W,r=H` place commands. Place-by-ID doesn't
# re-parse the PNG — the terminal just blits the cached bitmap at
# the requested cell footprint. ~30 bytes per frame instead of
# ~200 KB, no re-parse, no flicker.
#
# WezTerm supports the Kitty protocol alongside iTerm2's OSC 1337.
# We pick Kitty when available and fall back to OSC 1337 only for
# terminals that don't speak Kitty.
#
# The transmit step is done OUTSIDE Rich's render cycle — we write
# the chunks directly to stdout from PaintDryDisplay.on_focus_preview,
# which runs on the narrator reader's event thread. By the time
# Rich's next frame emits the `a=p` place command via the
# FocusPreviewKittyImage renderable, WezTerm has decoded the PNG in
# the background and the cache is ready. The place command renders
# the image at the correct cursor position inside the renderable's
# own border frame.
# ---------------------------------------------------------------------

#: Numeric Kitty image ID used for the focus preview cache. We use a
#: single fixed ID because we only ever care about the current item's
#: preview — each new focus_preview event re-transmits with the same
#: ID, which overwrites the previous image in WezTerm's cache rather
#: than accumulating. Bounded memory footprint on the terminal side.
_KITTY_IMAGE_ID = 1

#: Base64 chunks must be at most this many characters. Kitty protocol
#: spec: "chunk size of 4096 for the base64-encoded data".
_KITTY_CHUNK_SIZE = 4096


def _build_kitty_transmit_chunks(
    png_bytes: bytes,
    image_id: int,
) -> list[str]:
    """Return a list of Kitty APC escape sequences that transmit the
    given PNG to the terminal under the specified numeric image ID.

    Format per chunk::

        ESC _G <control>;<base64 chunk> ESC \\

    First chunk carries the full control string
    ``f=100,i=<id>,t=d,m=<flag>``. Subsequent chunks carry only
    ``m=<flag>``. All chunks except the last have ``m=1`` (more data
    coming); the last has ``m=0``. All chunks except the last have
    payload size that is a multiple of 4 (base64 alignment).

    No action key (``a=``) is set — this is a transmit-only operation
    that caches the image under ``image_id`` without displaying it.
    Displaying happens later via :func:`_build_kitty_place_sequence`.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    # Chunk size must be a multiple of 4 for all but the last chunk
    # to maintain base64 alignment. _KITTY_CHUNK_SIZE is already a
    # multiple of 4, so naive slicing works.
    raw_chunks = [
        b64[i : i + _KITTY_CHUNK_SIZE] for i in range(0, len(b64), _KITTY_CHUNK_SIZE)
    ]
    if not raw_chunks:
        # Degenerate: zero-byte input. Emit a single empty last chunk
        # so the caller gets a well-formed envelope to send.
        raw_chunks = [""]
    envelopes: list[str] = []
    for idx, chunk in enumerate(raw_chunks):
        is_last = idx == len(raw_chunks) - 1
        m_flag = "0" if is_last else "1"
        if idx == 0:
            control = f"f=100,i={image_id},t=d,m={m_flag}"
        else:
            control = f"m={m_flag}"
        envelopes.append(f"\x1b_G{control};{chunk}\x1b\\")
    return envelopes


def _build_kitty_place_sequence(
    image_id: int,
    *,
    cell_width: int,
    cell_height: int,
) -> str:
    """Return a Kitty APC escape sequence that places a previously-
    transmitted image by ID at the current cursor position, occupying
    ``cell_width × cell_height`` terminal cells.

    Format::

        ESC _G a=p,i=<id>,c=<W>,r=<H>,C=1 ESC \\

    The ``C=1`` parameter is LOAD-BEARING. Without it, Kitty's default
    cursor-movement policy after `a=p` is "move the cursor right by
    ``c`` cells AND down by ``r`` rows." Since Rich's layout engine
    has no idea that our control-marked Segment containing the place
    sequence causes cursor movement in the terminal, the rest of
    Rich's frame output (right border, subsequent border rows, the
    bottom border) lands at completely wrong physical coordinates:
    Rich thinks it's writing "row 2" of the frame, but the cursor is
    actually at physical row ``1 + r`` + column ``1 + c``, so row 2's
    left border appears 18 rows below where we want it and the
    "letterbox" you see below the image is actually Rich's row 2..18
    border segments painted into the empty area below the image.

    With ``C=1``, the place command paints the image but leaves the
    cursor where it was before, so subsequent writes land at the
    expected text-cell coordinates and Rich's frame lines up with
    the image the way we expect.

    No payload — this is a control-only sequence, so the body after
    ``ESC_G`` is just the comma-separated control keys followed
    directly by the terminator.
    """
    return (
        f"\x1b_Ga=p,i={image_id},c={cell_width},r={cell_height},C=1\x1b\\"
    )


def _query_terminal_cell_aspect(
    *,
    timeout_s: float = 0.15,
    stream=None,
) -> float | None:
    """Query the terminal for its cell pixel dimensions via CSI 16t.

    Returns the cell aspect ratio (height / width, in screen pixels)
    if the terminal responds with a well-formed answer. Returns
    ``None`` on any failure — no tty, unsupported sequence, timeout,
    parse error, etc. Callers should fall back to
    ``_DEFAULT_TERMINAL_CELL_ASPECT``.

    Protocol: send ``ESC [ 1 6 t`` to stdout and read the response
    ``ESC [ 6 ; <height_px> ; <width_px> t`` from stdin. Heights and
    widths are integers in pixels.

    Implementation notes:
    - Puts stdin into cbreak mode temporarily so we can read the
      response without waiting for a newline.
    - Uses a short timeout (``timeout_s``) so that non-cooperating
      terminals don't hang startup.
    - Reads via ``select.select`` in a loop, accumulating bytes
      until the response terminator ``t`` arrives or we time out.
    - Restores termios state in a finally block even if the read
      or parse raises.
    - Safe to call with stdout as a non-tty (returns None early).

    The ``stream`` argument is for testing: lets a test swap in a
    fake stream. Normal callers should leave it as None and the
    function will use ``sys.stdout`` for the query and ``sys.stdin``
    for the response.
    """
    import select
    import termios
    import tty

    out = stream if stream is not None else sys.stdout
    inp = sys.stdin
    try:
        if not (out.isatty() and inp.isatty()):
            return None
    except (AttributeError, ValueError):
        return None

    try:
        fd = inp.fileno()
        original = termios.tcgetattr(fd)
    except (termios.error, AttributeError, ValueError, OSError):
        return None

    try:
        tty.setcbreak(fd)
        out.write("\x1b[16t")
        out.flush()

        # Read the response with a timeout. Expected format:
        #   ESC [ 6 ; <h> ; <w> t
        # where h and w are cell pixel dimensions.
        deadline = time.monotonic() + timeout_s
        buffer = b""
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                return None
            if not chunk:
                continue
            buffer += chunk
            if b"t" in buffer:
                break
        else:
            return None

        # Find the response signature: b'\x1b[6;<h>;<w>t'
        start = buffer.find(b"\x1b[6;")
        if start < 0:
            return None
        end = buffer.find(b"t", start)
        if end < 0:
            return None
        payload = buffer[start + len("\x1b[6;") : end].decode(
            "ascii", errors="ignore"
        )
        parts = payload.split(";")
        if len(parts) != 2:
            return None
        cell_height_px = int(parts[0])
        cell_width_px = int(parts[1])
        if cell_width_px <= 0:
            return None
        return cell_height_px / cell_width_px
    except (ValueError, OSError, termios.error):
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
        except (termios.error, OSError):
            pass


def _supports_kitty_graphics(term_program: str | None) -> bool:
    """Return True if the terminal supports the Kitty graphics
    protocol.

    Known-good: WezTerm (since ~2022), kitty itself. iTerm2 supports
    Kitty graphics in recent builds too, but we keep it on the OSC
    1337 path for now since that's the path we have more empirical
    coverage on. Everything else falls through to whatever other
    paths are available.
    """
    if not term_program:
        return False
    return term_program in {"WezTerm", "kitty"}


def _supports_inline_images(term_program: str | None) -> bool:
    """Return True if the terminal identified by ``$TERM_PROGRAM``
    supports the iTerm2 inline image protocol.

    Known-good: WezTerm, iTerm.app (iTerm2). Everything else — xterm,
    Apple_Terminal, tmux, unset — is treated as unsupported and falls
    through to the half-block fallback renderer.
    """
    if not term_program:
        return False
    return term_program in {"WezTerm", "iTerm.app"}


class FocusPreviewInlineImage:
    """Rich Renderable that emits an iTerm2 inline image framed in a
    self-drawn border box.

    This renderable is deliberately NOT wrapped in a ``rich.Panel``
    because Panel's padding logic writes literal spaces to the cells
    to the right of its inner content on every row, filling the panel
    to its declared inner width. Those spaces land on exactly the
    terminal cells that the iTerm2 escape sequence just painted image
    pixels into, overwriting the image the instant it's drawn. The
    failure mode is extremely subtle — Rich's output byte stream does
    carry the full image sequence — but the image is gone before the
    operator sees it because Rich immediately clobbers its cells.

    The fix is to own the border drawing ourselves and use
    cursor-forward escape sequences (``ESC[nC``) marked as zero-width
    control segments to advance the cursor across the image region
    without writing any visible characters to those cells. The
    terminal-painted image pixels are preserved because nothing is
    written over them.

    The renderable yields, in order:
      1. A top border row ``╭─… title …─╮``
      2. A content row ``│`` + space + image escape sequence +
         cursor-forward past the image cells + ``│``
      3. ``cell_height - 1`` more content rows ``│`` + cursor-forward
         past the image cells + ``│``
      4. A bottom border row ``╰─…─╯``

    Rich sees the border segments as visible text with the expected
    cell widths, so its layout accounting treats this renderable as a
    (cell_width + 4) × (cell_height + 2) rectangle in the panels
    stack. No outer Panel needed; the renderable IS the panel.
    """

    def __init__(
        self,
        *,
        png_bytes: bytes,
        cell_width: int,
        cell_height: int,
        title: str = "",
    ) -> None:
        self._png_bytes = png_bytes
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._title = title
        self._sequence = _build_iterm2_inline_image_sequence(
            png_bytes, cell_width=cell_width, cell_height=cell_height
        )
        # NOTE: Rich's Live.LiveRender.position_cursor() emits
        # ERASE_IN_LINE (CSI 2K) on every row of the previous frame
        # before each refresh. That explicitly clears every cell in
        # the region we drew into, including image pixels. So we
        # MUST re-emit the iTerm2 escape sequence on every render
        # call — any "emit once" optimization produces an empty
        # container frame because the image gets cleared between
        # frames and never repainted. The visible cost is a ~24 Hz
        # strobing re-rasterization as WezTerm re-parses the
        # base64 PNG on every tick. The durable fix is the Kitty
        # graphics protocol with placement IDs (upload image once
        # with a=t, reference with a=p on subsequent frames), which
        # avoids re-sending the PNG data. Tracked as a follow-up
        # attractor; until then, we eat the flicker because at
        # least the image is visible.

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        border = Style.parse("#3d4458")
        # Total cell width of the framed output: image + one padding
        # space on each side of the image + border char on each side.
        inner_width = self._cell_width
        total_width = inner_width + 2  # just the borders, no extra pad

        # Top border with embedded title: ╭─ <title> ─…─╮
        title_text = self._title
        max_title = max(0, inner_width - 4)
        if len(title_text) > max_title:
            title_text = title_text[: max(0, max_title - 1)] + "…"
        if title_text:
            prefix = f"╭─ {title_text} "
            filler_len = total_width - len(prefix) - 1
            if filler_len < 0:
                filler_len = 0
            top_border = prefix + ("─" * filler_len) + "╮"
        else:
            top_border = "╭" + ("─" * (total_width - 2)) + "╮"
        yield Segment(top_border, border)
        yield Segment.line()

        # Cursor-forward escape advances the cursor `inner_width`
        # columns without writing any visible characters. Marked as
        # a control segment so Rich's cell_length accounting treats
        # it as zero cells — Rich's line-level logic will not try
        # to line-fit, truncate, or pad around it.
        forward_escape = f"\x1b[{inner_width}C"

        # First interior row carries the iTerm2 image escape sequence.
        # Must re-emit on every render because Rich Live's
        # position_cursor() erases every row of the previous frame
        # before each refresh (see note in __init__).
        yield Segment("│", border)
        yield Segment(self._sequence, None, [(ControlType.BELL,)])
        yield Segment(forward_escape, None, [(ControlType.BELL,)])
        yield Segment("│", border)
        yield Segment.line()

        # Remaining interior rows: border + cursor-forward + border.
        # These rows are visually "blank" from Rich's perspective but
        # the terminal has image pixels painted into the cells from
        # the first row's escape sequence. Cursor-forward preserves
        # those pixels; writing spaces here would overwrite them.
        for _ in range(self._cell_height - 1):
            yield Segment("│", border)
            yield Segment(forward_escape, None, [(ControlType.BELL,)])
            yield Segment("│", border)
            yield Segment.line()

        # Bottom border.
        bottom_border = "╰" + ("─" * (total_width - 2)) + "╯"
        yield Segment(bottom_border, border)
        yield Segment.line()


class FocusPreviewKittyImage:
    """Rich Renderable that places a previously-transmitted Kitty
    graphics image inside a self-drawn border frame.

    Unlike :class:`FocusPreviewInlineImage`, this class does NOT
    carry the PNG data. The transmit (`_build_kitty_transmit_chunks`)
    must be done out-of-band by the caller — typically directly to
    stdout from ``PaintDryDisplay.on_focus_preview`` on the narrator
    event thread, so WezTerm starts decoding in the background
    before Rich's next refresh fires.

    Cell-box sizing is done at RENDER TIME, not at construction.
    The renderable stores the image's pixel dimensions and the
    terminal's cell aspect ratio; on every ``__rich_console__``
    call, it reads the current ``ConsoleOptions.max_width`` and
    recomputes the cell box to fit. This makes the preview
    resize-safe — narrowing or widening the terminal window
    causes the next render to produce a box that fits the new
    width automatically, because Rich passes the updated
    console options through on every refresh.

    The image_id is reused across focus_preview events (see
    ``_KITTY_IMAGE_ID``), so each transmit overwrites the
    previous cached image rather than accumulating — bounded
    memory footprint on the terminal side.
    """

    def __init__(
        self,
        *,
        image_id: int,
        image_pixel_width: int,
        image_pixel_height: int,
        terminal_cell_aspect: float,
        title: str = "",
    ) -> None:
        self._image_id = image_id
        self._image_pixel_width = image_pixel_width
        self._image_pixel_height = image_pixel_height
        self._terminal_cell_aspect = terminal_cell_aspect
        self._title = title

    def _compute_box(self, available_width: int) -> tuple[int, int]:
        """Compute (cell_width, cell_height) for the image at the
        given available width budget. Leaves 2 cells for the
        border so the usable interior is ``available_width - 2``.
        """
        inner_budget = max(1, available_width - 2)
        cw, ch = _compute_inline_image_cell_dimensions(
            self._image_pixel_width,
            self._image_pixel_height,
            max_cell_height=_INLINE_IMAGE_CELL_HEIGHT,
            max_cell_width=min(_INLINE_IMAGE_MAX_CELL_WIDTH, inner_budget),
            terminal_cell_aspect=self._terminal_cell_aspect,
        )
        return cw, ch

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        border = Style.parse("#3d4458")
        cell_width, cell_height = self._compute_box(options.max_width)
        inner_width = cell_width
        total_width = inner_width + 2

        place_sequence = _build_kitty_place_sequence(
            self._image_id,
            cell_width=cell_width,
            cell_height=cell_height,
        )

        # Top border with embedded title.
        title_text = self._title
        max_title = max(0, inner_width - 4)
        if len(title_text) > max_title:
            title_text = title_text[: max(0, max_title - 1)] + "…"
        if title_text:
            prefix = f"╭─ {title_text} "
            filler_len = total_width - len(prefix) - 1
            if filler_len < 0:
                filler_len = 0
            top_border = prefix + ("─" * filler_len) + "╮"
        else:
            top_border = "╭" + ("─" * (total_width - 2)) + "╮"
        yield Segment(top_border, border)
        yield Segment.line()

        forward_escape = f"\x1b[{inner_width}C"

        # First interior row: emit the Kitty place-by-ID command.
        yield Segment("│", border)
        yield Segment(place_sequence, None, [(ControlType.BELL,)])
        yield Segment(forward_escape, None, [(ControlType.BELL,)])
        yield Segment("│", border)
        yield Segment.line()

        # Remaining interior rows: border + cursor-forward + border.
        for _ in range(cell_height - 1):
            yield Segment("│", border)
            yield Segment(forward_escape, None, [(ControlType.BELL,)])
            yield Segment("│", border)
            yield Segment.line()

        # Bottom border.
        bottom_border = "╰" + ("─" * (total_width - 2)) + "╯"
        yield Segment(bottom_border, border)
        yield Segment.line()


def _otsu_threshold(luminances) -> float:
    """Classic Otsu's method: pick the luminance cut that maximizes
    between-class variance across a 256-bin histogram.

    Returns a float threshold in [0, 255]. Degenerate inputs (empty,
    uniform) return a safe midpoint rather than raising — the renderer
    treats such crops as "no ink" and the exact cut doesn't matter.
    """
    histogram = [0] * 256
    total = 0
    for value in luminances:
        bucket = int(value)
        if bucket < 0:
            bucket = 0
        elif bucket > 255:
            bucket = 255
        histogram[bucket] += 1
        total += 1
    if total == 0:
        return 127.5

    sum_total = 0.0
    for i in range(256):
        sum_total += i * histogram[i]

    sum_bg = 0.0
    weight_bg = 0
    best_variance = -1.0
    best_threshold = 127.5
    for i in range(256):
        weight_bg += histogram[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * histogram[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = float(i)
    # Return the upper edge of the chosen histogram bin. The histogram
    # is integer-binned (`int(luma)`), so any float luma that fell into
    # bin `i` is in [i, i+1). Returning `i + 1.0` means strict `<` at
    # the caller correctly classifies every member of that bin as
    # background (the darker / ink class).
    return best_threshold + 1.0


def _build_focus_preview_pixels(
    png_bytes: bytes,
    *,
    max_width_chars: int,
    max_height_rows: int,
) -> list[list[tuple[int, int, int]]]:
    """Sample a source crop into a grid of average-RGB pixels.

    The returned grid has exactly ``target_height`` rows and
    ``target_width`` cols, where those dimensions are ``_scaled_preview_size``
    applied to the source. The caller controls whether this is
    "one row per terminal row" or "two rows per terminal row" (for
    half-blocks) by passing the appropriate ``max_height_rows``.

    No tone mapping. No filmic. No paper/ink lerp. The downstream
    renderer is responsible for turning these raw samples into
    whatever output surface is appropriate.
    """
    pix = fitz.Pixmap(png_bytes)
    target_width, target_height = _scaled_preview_size(
        pix.width,
        pix.height,
        max_width_chars=max_width_chars,
        max_height_rows=max_height_rows,
    )
    pixels: list[list[tuple[int, int, int]]] = []
    for y in range(target_height):
        row: list[tuple[int, int, int]] = []
        for x in range(target_width):
            row.append(_sample_preview_rgb(pix, x, y, target_width, target_height))
        pixels.append(row)
    return pixels


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pixel_luma(rgb: tuple[int, int, int]) -> float:
    return (0.299 * rgb[0]) + (0.587 * rgb[1]) + (0.114 * rgb[2])


def _render_focus_preview_pixels(
    pixels: list[list[tuple[int, int, int]]],
    *,
    now: float | None = None,
    pending: bool = False,
) -> Group:
    """Render a sampled pixel grid as a Rich Group.

    ``pixels`` is expected to be sampled at 2× vertical density relative
    to the terminal row budget — each pair of source rows (2y, 2y+1)
    becomes one terminal row. The steady-state renderer uses that pair
    as the top and bottom halves of a half-block (▀). The pending
    (transition) renderer averages each pair back down to a single
    per-cell RGB and then runs its existing glyph-overlay animation.
    """
    now = time.monotonic() if now is None else now
    if pending:
        return _render_focus_preview_pending(pixels, now=now)
    return _render_focus_preview_steady(pixels)


def _render_focus_preview_steady(
    pixels: list[list[tuple[int, int, int]]],
) -> Group:
    """Legibility-first steady-state renderer.

    Binary Otsu threshold across all sampled luminances, half-block
    cells (one terminal cell = one top half-pixel + one bottom
    half-pixel), hard ink/paper palette. No tone mapping, no lerp,
    no gradient. Ugly is acceptable; illegible is not.
    """
    if not pixels or not pixels[0]:
        return Group()

    source_height = len(pixels)
    source_width = len(pixels[0])

    # Collect all luminances for Otsu. Use ink and paper colors
    # that are actually visually distinct from the panel background,
    # so thresholded cells read as "page" and not as "void".
    luminances = [_pixel_luma(rgb) for row in pixels for rgb in row]
    threshold = _otsu_threshold(luminances)

    # If the crop is effectively single-tone (variance too low for Otsu
    # to find a meaningful cut), fall back to marking everything as paper.
    # This keeps blank regions rendering as page instead of as garbled noise.
    lum_span = max(luminances) - min(luminances) if luminances else 0.0
    degenerate = lum_span < 12.0

    ink_hex = _rgb_to_hex(_FOCUS_PREVIEW_HARD_INK_RGB)
    paper_hex = _rgb_to_hex(_FOCUS_PREVIEW_HARD_PAPER_RGB)

    def _color_for(rgb: tuple[int, int, int]) -> str:
        if degenerate:
            return paper_hex
        return ink_hex if _pixel_luma(rgb) < threshold else paper_hex

    rows: list[Text] = []
    # Walk source rows in pairs. If the source has an odd number of rows,
    # the dangling final row pairs with itself (top == bottom).
    y = 0
    while y < source_height:
        top_row = pixels[y]
        bottom_row = pixels[y + 1] if (y + 1) < source_height else top_row
        text = Text(no_wrap=True, overflow="ignore")
        for col in range(source_width):
            top_color = _color_for(top_row[col])
            bottom_color = (
                _color_for(bottom_row[col])
                if col < len(bottom_row)
                else top_color
            )
            # ▀ = U+2580 UPPER HALF BLOCK. Foreground is the top half,
            # background is the bottom half. Exactly what we want for a
            # binary-threshold image surface.
            text.append("\u2580", style=f"{top_color} on {bottom_color}")
        rows.append(text)
        y += 2
    return Group(*rows)


def _render_focus_preview_pending(
    pixels: list[list[tuple[int, int, int]]],
    *,
    now: float,
) -> Group:
    """Transition-layer renderer (unchanged animation, now fed from a
    2×-vertical pixel grid by averaging row pairs back down to 1×)."""
    if not pixels or not pixels[0]:
        return Group()

    # Average pairs of sampled rows back down to one row per terminal
    # row, so the existing glyph-overlay animation keeps working
    # against the density it was tuned for.
    source_height = len(pixels)
    source_width = len(pixels[0])
    collapsed: list[list[tuple[int, int, int]]] = []
    y = 0
    while y < source_height:
        top_row = pixels[y]
        bottom_row = pixels[y + 1] if (y + 1) < source_height else top_row
        merged: list[tuple[int, int, int]] = []
        for col in range(source_width):
            top_rgb = top_row[col]
            bottom_rgb = bottom_row[col] if col < len(bottom_row) else top_rgb
            merged.append(
                (
                    (top_rgb[0] + bottom_rgb[0]) // 2,
                    (top_rgb[1] + bottom_rgb[1]) // 2,
                    (top_rgb[2] + bottom_rgb[2]) // 2,
                )
            )
        collapsed.append(merged)
        y += 2

    rows: list[Text] = []
    for char_row, pixel_row in enumerate(collapsed):
        row = Text(no_wrap=True, overflow="ignore")
        for col, rgb in enumerate(pixel_row):
            avg_rgb = rgb
            flow = 0.5 + (
                0.5
                * math.sin((col * 0.44) - (char_row * 0.18) - (now * 6.8))
            )
            pulse = 0.5 + (
                0.5
                * math.sin((col * 0.16) + (char_row * 0.31) + (now * 5.3))
            )
            retention = 0.54 + (0.18 * flow)
            toned_rgb = _interp_rgb(_FOCUS_PREVIEW_BG_RGB, avg_rgb, retention)
            avg_luma = (
                (0.299 * toned_rgb[0])
                + (0.587 * toned_rgb[1])
                + (0.114 * toned_rgb[2])
            )
            glyph_gate = 0.38 + (0.20 * pulse)
            glyph_drive = (0.58 * flow) + (0.25 * pulse)
            if glyph_drive > glyph_gate:
                char_phase = (0.62 * flow) + (0.38 * (avg_luma / 255.0))
                char_index = min(
                    len(_FOCUS_PREVIEW_OVERLAY_CHARS) - 1,
                    int(char_phase * len(_FOCUS_PREVIEW_OVERLAY_CHARS)),
                )
                palette_index = min(
                    len(_FOCUS_PREVIEW_OVERLAY_RGBS) - 1,
                    int((avg_luma / 255.0) * len(_FOCUS_PREVIEW_OVERLAY_RGBS)),
                )
                fg_rgb = _interp_rgb(
                    _FOCUS_PREVIEW_OVERLAY_RGBS[palette_index],
                    _FOCUS_PREVIEW_PAPER_RGB,
                    0.16 + (0.18 * flow),
                )
                bg_rgb = _interp_rgb(_FOCUS_PREVIEW_BG_RGB, toned_rgb, 0.34)
                row.append(
                    _FOCUS_PREVIEW_OVERLAY_CHARS[char_index],
                    style=f"{_rgb_to_hex(fg_rgb)} on {_rgb_to_hex(bg_rgb)}",
                )
                continue
            # Non-glyph cells during the transition: render as a faded
            # block color. The transition animation no longer tries to
            # produce a legible steady-state image underneath itself.
            bg_rgb = _interp_rgb(_FOCUS_PREVIEW_BG_RGB, toned_rgb, 0.34)
            row.append(
                " ",
                style=f"{_rgb_to_hex(bg_rgb)} on {_rgb_to_hex(bg_rgb)}",
            )
        rows.append(row)
    return Group(*rows)


def _scaled_preview_size(
    source_width: int,
    source_height: int,
    *,
    max_width_chars: int,
    max_height_rows: int,
) -> tuple[int, int]:
    scale = min(
        max_width_chars / max(1, source_width),
        max_height_rows / max(1, source_height),
        1.0,
    )
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return width, height


def _sample_preview_rgb(
    pix: fitz.Pixmap,
    x: int,
    y: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int]:
    src_x0 = max(0, int((x / target_width) * pix.width))
    src_x1 = min(
        pix.width,
        max(src_x0 + 1, int(math.ceil(((x + 1) / target_width) * pix.width))),
    )
    src_y0 = max(0, int((y / target_height) * pix.height))
    src_y1 = min(
        pix.height,
        max(src_y0 + 1, int(math.ceil(((y + 1) / target_height) * pix.height))),
    )
    samples = pix.samples
    red = 0
    green = 0
    blue = 0
    count = 0
    darkest_luma = float("inf")
    darkest_rgb = _FOCUS_PREVIEW_BG_RGB
    for src_y in range(src_y0, src_y1):
        row_offset = src_y * pix.width * pix.n
        for src_x in range(src_x0, src_x1):
            offset = row_offset + (src_x * pix.n)
            sample_red = samples[offset]
            sample_green = samples[offset + 1]
            sample_blue = samples[offset + 2]
            red += sample_red
            green += sample_green
            blue += sample_blue
            count += 1
            sample_luma = (
                (0.299 * sample_red)
                + (0.587 * sample_green)
                + (0.114 * sample_blue)
            )
            if sample_luma < darkest_luma:
                darkest_luma = sample_luma
                darkest_rgb = (sample_red, sample_green, sample_blue)
    if count == 0:
        return _FOCUS_PREVIEW_BG_RGB
    avg_rgb = (
        int(round(red / count)),
        int(round(green / count)),
        int(round(blue / count)),
    )
    avg_luma = (
        (0.299 * avg_rgb[0])
        + (0.587 * avg_rgb[1])
        + (0.114 * avg_rgb[2])
    )
    ink_weight = _clamp((avg_luma - darkest_luma - 18.0) / 120.0, 0.0, 0.58)
    return _interp_rgb(avg_rgb, darkest_rgb, ink_weight)


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

    tier_dim = _history_tier_dim_factor(layer_index)
    base_rgb = _scale_rgb(_BASE_RGB.get(kind, _BASE_RGB["line"]), tier_dim)
    peak_rgb = _scale_rgb(
        _SHIMMER_KIND_PEAK_RGB.get(kind, _SHIMMER_PEAK_RGB),
        tier_dim,
    )
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
    if kind in {"line", "line_alt"}:
        kind_intensity *= tier_dim
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
            visual_row = absolute_col // wrap_width
            distance = head - visual_col
            row_dim = max(
                _HISTORY_CONTINUATION_ROW_MIN,
                1.0 - (visual_row * _HISTORY_CONTINUATION_ROW_STEP),
            )
            row_base_rgb = _scale_rgb(base_rgb, row_dim)
            row_peak_rgb = _scale_rgb(peak_rgb, row_dim)
            _append_shimmer_char(
                text_obj, ch, distance, row_base_rgb, row_peak_rgb,
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


def _undulation_hue_deg(
    now_s: float,
    global_i: int,
    *,
    cycle_s: float,
    center_deg: float,
    range_deg: float,
    per_char_phase_offset: float,
    phase_offset_rad: float = 0.0,
    direction: float = 1.0,
) -> float:
    """Return the per-character band hue at a given time and position."""
    phase = (
        -direction * now_s * (2 * math.pi / cycle_s)
        + phase_offset_rad
        + global_i * per_char_phase_offset
    )
    return center_deg + range_deg * math.sin(phase)


def _render_warm_undulating(
    text_obj: Text,
    content: str,
    indent_width: int,
    wrap_width: int | None,
    *,
    cycle_s: float,
    center_deg: float,
    range_deg: float,
    per_char_phase_offset: float,
    phase_offset_rad: float,
    direction: float,
    base_sat: float,
    base_val: float,
    luminance_correction_strength: float,
    char_offset: int = 0,
    freeze_age_s: float | None = None,
    frozen_sat_mul: float = 1.0,
    frozen_val_mul: float = 1.0,
    freeze_fade_s: float | None = None,
    shimmer_cycle_s: float | None = None,
    shimmer_width: int = _SHIMMER_WIDTH,
    shimmer_v_boost: float = 0.08,
    shimmer_s_drop: float = 0.40,
    bold_active_head: bool = False,
) -> Text:
    """Render a per-character undulating band with an optional shimmer crest."""
    if not content:
        return text_obj

    now = time.monotonic()
    cycle = shimmer_cycle_s if shimmer_cycle_s is not None else _SHIMMER_RECENT_CYCLE_S
    shimmer_phase = (now % cycle) / cycle
    if wrap_width is not None and wrap_width > shimmer_width:
        shimmer_head = (
            shimmer_phase * (wrap_width + shimmer_width) - shimmer_width
        )
    else:
        shimmer_head = (
            shimmer_phase * (len(content) + shimmer_width) - shimmer_width
        )

    if freeze_age_s is None or freeze_fade_s is None:
        sat_mul = 1.0
        val_mul = 1.0
    else:
        fade = max(0.0, min(1.0, freeze_age_s / freeze_fade_s))
        sat_mul = 1.0 - fade * (1.0 - frozen_sat_mul)
        val_mul = 1.0 - fade * (1.0 - frozen_val_mul)

    s_for_ref = base_sat * sat_mul
    ref_r, ref_g, ref_b = _hsv_to_rgb(center_deg, s_for_ref, 1.0)
    ref_luminance = 0.2126 * ref_r + 0.7152 * ref_g + 0.0722 * ref_b

    for i, ch in enumerate(content):
        global_i = char_offset + i
        h = _undulation_hue_deg(
            now,
            global_i,
            cycle_s=cycle_s,
            center_deg=center_deg,
            range_deg=range_deg,
            per_char_phase_offset=per_char_phase_offset,
            phase_offset_rad=phase_offset_rad,
            direction=direction,
        )
        s = base_sat * sat_mul
        v = base_val * val_mul

        test_r, test_g, test_b = _hsv_to_rgb(h, s, 1.0)
        test_luminance = (
            0.2126 * test_r + 0.7152 * test_g + 0.0722 * test_b
        )
        if test_luminance > 1:
            raw_correction = ref_luminance / test_luminance
            correction = 1.0 + (
                raw_correction - 1.0
            ) * luminance_correction_strength
            v = max(0.0, min(1.0, v * correction))

        if wrap_width is not None and wrap_width > shimmer_width:
            visual_col = (indent_width + i) % wrap_width
            distance = shimmer_head - visual_col
        else:
            distance = shimmer_head - i

        bold_head = False
        if 0 <= distance < shimmer_width:
            shimmer_intensity = 1.0 - (distance / shimmer_width)
            v = min(1.0, v + shimmer_v_boost * shimmer_intensity)
            s = max(0.0, s - shimmer_s_drop * shimmer_intensity)
            if -0.5 <= distance < 1.5:
                bold_head = bold_active_head

        r, g, b = _hsv_to_rgb(h, s, v)
        style = f"#{r:02x}{g:02x}{b:02x}"
        if bold_head:
            style = f"bold {style}"
        text_obj.append(ch, style=style)

    return text_obj


def _render_live_undulating(
    text_obj: Text,
    content: str,
    indent_width: int,
    wrap_width: int | None,
    is_active: bool,
    palette_variant: str = "cool",
    char_offset: int = 0,
    freeze_age_s: float | None = None,
) -> Text:
    """Render the live line with per-character undulating color washes.

    The live lane alternates between a cooler aqua/green/bone wash and
    a softened pastel warm wash on accepted thought lines. Adjacent
    characters still land at slightly different points in the palette
    and the whole field undulates over a slow cycle. The shimmer head
    brightens characters near it and pushes their saturation down
    (toward white) for a heat-flicker feel.

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

    if palette_variant == "warm":
        center_deg = _LIVE_WARM_HUE_CENTER_DEG
        range_deg = _LIVE_WARM_HUE_RANGE_DEG
        base_sat = _LIVE_WARM_BASE_SAT
        base_val = _LIVE_WARM_BASE_VAL
        luminance_correction_strength = _LIVE_WARM_LUMINANCE_CORRECTION_STRENGTH
    else:
        center_deg = _LIVE_HUE_CENTER_DEG
        range_deg = _LIVE_HUE_RANGE_DEG
        base_sat = _LIVE_BASE_SAT
        base_val = _LIVE_BASE_VAL
        luminance_correction_strength = _LIVE_LUMINANCE_CORRECTION_STRENGTH

    return _render_warm_undulating(
        text_obj,
        content,
        indent_width,
        wrap_width,
        cycle_s=_LIVE_UNDULATION_CYCLE_S,
        center_deg=center_deg,
        range_deg=range_deg,
        per_char_phase_offset=_LIVE_PER_CHAR_PHASE_OFFSET,
        phase_offset_rad=_LIVE_PHASE_OFFSET_RAD,
        direction=_LIVE_UNDULATION_DIRECTION,
        base_sat=base_sat,
        base_val=base_val,
        luminance_correction_strength=luminance_correction_strength,
        char_offset=char_offset,
        freeze_age_s=None if is_active else freeze_age_s,
        frozen_sat_mul=_LIVE_FROZEN_SAT_MUL,
        frozen_val_mul=_LIVE_FROZEN_VAL_MUL,
        freeze_fade_s=_LIVE_FREEZE_FADE_S,
        shimmer_cycle_s=_SHIMMER_RECENT_CYCLE_S,
        shimmer_width=_SHIMMER_WIDTH,
        shimmer_v_boost=0.08,
        shimmer_s_drop=0.40,
        bold_active_head=is_active,
    )


def _render_status_undulating(
    text_obj: Text,
    content: str,
    indent_width: int,
    wrap_width: int | None,
) -> Text:
    """Render the sticky status rail as ember heat with cooler ash glints."""
    if not content:
        return text_obj

    now = time.monotonic()
    shimmer_phase = (now % _SHIMMER_DEFAULT_CYCLE_S) / _SHIMMER_DEFAULT_CYCLE_S
    if wrap_width is not None and wrap_width > _SHIMMER_WIDTH:
        shimmer_head = (
            shimmer_phase * (wrap_width + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        )
    else:
        shimmer_head = (
            shimmer_phase * (len(content) + _SHIMMER_WIDTH) - _SHIMMER_WIDTH
        )

    ref_r, ref_g, ref_b = _hsv_to_rgb(_STATUS_HUE_CENTER_DEG, _STATUS_BASE_SAT, 1.0)
    ref_luminance = 0.2126 * ref_r + 0.7152 * ref_g + 0.0722 * ref_b

    for i, ch in enumerate(content):
        h = _undulation_hue_deg(
            now,
            i,
            cycle_s=_STATUS_UNDULATION_CYCLE_S,
            center_deg=_STATUS_HUE_CENTER_DEG,
            range_deg=_STATUS_HUE_RANGE_DEG,
            per_char_phase_offset=_STATUS_PER_CHAR_PHASE_OFFSET,
            phase_offset_rad=_STATUS_PHASE_OFFSET_RAD,
            direction=_STATUS_UNDULATION_DIRECTION,
        )
        s = _STATUS_BASE_SAT
        v = _STATUS_BASE_VAL

        test_r, test_g, test_b = _hsv_to_rgb(h, s, 1.0)
        test_luminance = 0.2126 * test_r + 0.7152 * test_g + 0.0722 * test_b
        if test_luminance > 1:
            raw_correction = ref_luminance / test_luminance
            correction = 1.0 + (
                raw_correction - 1.0
            ) * _STATUS_LUMINANCE_CORRECTION_STRENGTH
            v = max(0.0, min(1.0, v * correction))

        if wrap_width is not None and wrap_width > _SHIMMER_WIDTH:
            visual_col = (indent_width + i) % wrap_width
            distance = shimmer_head - visual_col
        else:
            distance = shimmer_head - i

        if 0 <= distance < _SHIMMER_WIDTH:
            shimmer_intensity = 1.0 - (distance / _SHIMMER_WIDTH)
            v = min(1.0, v + 0.09 * shimmer_intensity)
            s = max(0.0, s - 0.18 * shimmer_intensity)

        rgb = _hsv_to_rgb(h, s, v)
        cool_glint = max(
            0.0,
            math.sin(
                -now * (2 * math.pi / _STATUS_COOL_GLINT_CYCLE_S)
                + _STATUS_COOL_GLINT_PHASE_OFFSET_RAD
                + i * (_STATUS_PER_CHAR_PHASE_OFFSET * 0.72)
            ),
        )
        bone_glint = max(
            0.0,
            math.sin(
                now * (2 * math.pi / _STATUS_BONE_GLINT_CYCLE_S)
                + _STATUS_BONE_GLINT_PHASE_OFFSET_RAD
                + i * (_STATUS_PER_CHAR_PHASE_OFFSET * 0.46)
            ),
        )
        cool_weight = (cool_glint ** 1.35) * _STATUS_COOL_GLINT_STRENGTH
        bone_weight = (bone_glint ** 2.6) * _STATUS_BONE_GLINT_STRENGTH

        if 0 <= distance < _SHIMMER_WIDTH:
            shimmer_intensity = 1.0 - (distance / _SHIMMER_WIDTH)
            cool_weight *= 1.0 - (0.35 * shimmer_intensity)
            bone_weight += 0.10 * shimmer_intensity

        rgb = _blend_rgb(rgb, _STATUS_COOL_GLINT_RGB, cool_weight)
        rgb = _blend_rgb(
            rgb,
            _STATUS_BONE_GLINT_RGB,
            bone_weight * (1.0 - 0.45 * cool_weight),
        )
        text_obj.append(ch, style=_rgb_to_hex(rgb))

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
        self.title = "PROJECT PAINT DRY · sumi-e"
        self.subtitle = "bonsai narrator · live"
        self.current_model: str = ""
        self.current_set_label: str = ""
        self.current_subset_count: int | None = None
        self.current_item_bug: str = ""
        self.score_on_target_points = 0.0
        self.score_points_possible = 0.0
        self.score_left_on_table_points = 0.0
        self.score_left_on_table_potential = 0.0
        self.score_bad_call_points = 0.0
        self.score_bad_call_potential = 0.0

        # Sticky live: the thought lane has a streaming buffer plus a
        # frozen committed line. The status rail has its own separate
        # streaming buffer so status updates typewriter in-place without
        # stealing the live thought lane during a status refresh.
        self.status_line: str = ""
        self.status_streaming_line: str = ""
        self.streaming_line: str = ""
        self.frozen_line: str = ""
        self._frozen_line_parity: int = 0
        # Timestamp at which the most recent dispatch finished streaming.
        # Used to drive the slow fade from "just-arrived bright" to
        # "settled past tense" colors over _LIVE_FREEZE_FADE_S seconds
        # so the punch of completing a dispatch is visible. None when
        # nothing has been frozen yet.
        self._freeze_started_at: float | None = None

        # History entries are 3-tuples (kind, text, parity):
        #   kind in {"line", "header", "topic", "checkpoint"}
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
            num_layers=_VISIBLE_HISTORY_ROWS,
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
        self.focus_preview_png: bytes | None = None
        self.focus_preview_pixels: list[list[tuple[int, int, int]]] | None = None
        self.focus_preview_renderable: Group | None = None
        self.focus_preview_label: str = ""
        self.focus_preview_source: str = ""
        self.focus_preview_pending: bool = False
        self.focus_preview_pending_started: float | None = None
        self._focus_preview_pending_bucket: int | None = None
        self._focus_preview_pending_renderable: Group | None = None
        # Inline-image renderer state. When the terminal supports
        # iTerm2 inline images (WezTerm, iTerm2), we skip the half-
        # block renderer and let the terminal rasterize the crop PNG
        # directly — strictly better legibility at no grain cost.
        # Capability detection runs once at construction from
        # $TERM_PROGRAM. The cached renderable carries the escape
        # sequence + padding so Rich's layout reserves the right
        # vertical space.
        self.focus_preview_inline_renderable: FocusPreviewInlineImage | None = None
        self._inline_images_supported: bool = _supports_inline_images(
            os.environ.get("TERM_PROGRAM")
        )
        # Kitty graphics protocol state. Preferred over the iTerm2
        # OSC 1337 path when available because it supports
        # upload-once + reference-by-ID, which eliminates the
        # per-frame PNG re-parse that causes flicker on the iTerm2
        # path. The transmit happens outside Rich's render cycle
        # (directly to stdout from on_focus_preview on the narrator
        # event thread), then the renderable below yields only the
        # tiny place command on each frame.
        self.focus_preview_kitty_renderable: FocusPreviewKittyImage | None = None
        self._kitty_graphics_supported: bool = _supports_kitty_graphics(
            os.environ.get("TERM_PROGRAM")
        )
        # Query the terminal for its real cell pixel dimensions once
        # at init. Used by the Kitty renderer to compute correct
        # cell boxes for images — hardcoding this is brittle across
        # different fonts, terminal window sizes, and HiDPI settings,
        # and a bad value produces visible letterbox. If the query
        # fails (not a tty, terminal doesn't support CSI 16t,
        # timeout), fall back to a sane default constant.
        queried_aspect = _query_terminal_cell_aspect()
        self._terminal_cell_aspect: float = (
            queried_aspect
            if queried_aspect is not None
            else _DEFAULT_TERMINAL_CELL_ASPECT
        )
        # When True, render() shows a "press Enter to close" footer and
        # the final frame stays static while waiting for Enter.
        self.session_ended: bool = False
        self._session_ended_at: float | None = None
        self._session_started_at: float | None = None
        self._turn_started_at: float | None = None

    def __rich__(self) -> Group:
        return self.render()

    @staticmethod
    def _format_scorebug_points(value: float) -> str:
        return f"{value:.1f}"

    @staticmethod
    def _append_scorebug_cell(
        row: Text,
        label: str,
        value: str,
        *,
        label_style: str,
        value_style: str,
        label_pad: int = 1,
        value_pad: int = 1,
    ) -> None:
        if row.plain:
            row.append("  ", style="dim")
        row.append(f"{' ' * label_pad}{label}{' ' * label_pad}", style=label_style)
        row.append(f"{' ' * value_pad}{value}{' ' * value_pad}", style=value_style)

    @staticmethod
    def _append_scorebug_big_value_cell(
        label_row: Text,
        value_top_row: Text,
        value_middle_row: Text,
        value_bottom_row: Text,
        label: str,
        value: str,
        *,
        label_style: str,
        value_style: str,
        value_mid_style: str,
        label_pad: int = 3,
        value_pad: int = 1,
    ) -> None:
        if label_row.plain:
            label_row.append("  ", style="dim")
            value_top_row.append("  ", style="dim")
            value_middle_row.append("  ", style="dim")
            value_bottom_row.append("  ", style="dim")
        top, middle, bottom = _scorebug_big_value_rows(value)
        cell_width = len(f"{' ' * value_pad}{top}{' ' * value_pad}")
        label_lead = ""
        label_trail_width = max(0, cell_width - len(label_lead) - len(label))
        label_row.append(
            f"{label_lead}{label}{' ' * label_trail_width}",
            style=label_style,
        )
        _append_scorebug_value_row(
            value_top_row,
            f"{' ' * value_pad}{top}{' ' * value_pad}",
            strong_style=value_style,
            mid_style=value_mid_style,
        )
        _append_scorebug_value_row(
            value_middle_row,
            f"{' ' * value_pad}{middle}{' ' * value_pad}",
            strong_style=value_style,
            mid_style=value_mid_style,
        )
        _append_scorebug_value_row(
            value_bottom_row,
            f"{' ' * value_pad}{bottom}{' ' * value_pad}",
            strong_style=value_style,
            mid_style=value_mid_style,
        )

    def should_animate(self, now: float | None = None) -> bool:
        """Return whether the UI should keep driving the shimmer loop.

        Project Paint Dry is not a static log viewer. Even when no new
        tokens are arriving, the active session still has ongoing shimmer
        across the status rail, live field, and history stack. The only
        time we intentionally stop repainting is after the session has
        ended and the final frame is meant to stay still while waiting
        for Enter.
        """
        if not self.session_ended:
            return True
        if self._session_ended_at is None:
            return False
        now = time.monotonic() if now is None else now
        return now < (self._session_ended_at + _SESSION_END_ANIMATION_LINGER_S)

    @staticmethod
    def _entry_visual_rows(entry: tuple, wrap_width: int | None) -> int:
        """Estimate how many visual rows an entry will consume when wrapped."""
        if wrap_width is None or wrap_width <= 0:
            return 1

        kind = entry[0]
        text = entry[1]
        if kind == "header":
            prefix_width = len("─ ")
        elif kind == "topic":
            prefix_width = len("  · ")
        else:
            prefix_width = len("    ")

        visual_cols = max(1, prefix_width + len(text))
        return max(1, math.ceil(visual_cols / wrap_width))

    def _build_display_entries(
        self,
        wrap_width: int | None = None,
    ) -> list[tuple[tuple, bool, int]]:
        """Group history into items, reverse so newest item is first,
        and within each group keep entries in chronological order so
        the header sits ABOVE its narrator lines and topic.

        Then fill the visible budget by priority:
          1. All headers and topics (ESSENTIAL — structural anchors).
             These never get dropped while the deque has them.
          2. Narrator lines, newest first (OPTIONAL — disposable middle).
             Filled into whatever budget is left after essentials.

        This means a long-thinking item with 30+ narrator lines doesn't
        push older items' headers and topics off the display — only
        narrator lines drop. The user can always see "this is the item,
        here's the verdict" for every visible item; the play-by-play
        between them is the part that compresses.

        Returns a list of (entry, is_most_recent, group_depth) tuples
        in display order. group_depth is the per-item depth that resets
        at each header, so visual fading can restart from every item
        heading instead of running as one global downhill wash.
        is_most_recent is True for exactly the entry at the back of the
        deque (the most-recently-committed thing).
        """
        history_list = list(self.history)
        if not history_list:
            return []
        most_recent_idx = len(history_list) - 1

        # Forward-iterate, grouping at header boundaries, tracking
        # original deque indices.
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

        # Newest item on top. Within each group, keep the header first,
        # move the verdict/topic line directly underneath it for quick
        # scanning, then flip narrator lines so the freshest thought
        # sits closest to the decision and older thoughts descend.
        groups.reverse()

        # Flat list of (entry, deque_idx) in display order (top-down)
        flat: list[tuple[tuple, int]] = []
        for group in groups:
            header = [pair for pair in group if pair[0][0] == "header"]
            lines = [pair for pair in group if pair[0][0] == "line"]
            rest = [pair for pair in group if pair[0][0] not in ("header", "line")]
            rest.sort(
                key=lambda pair: {
                    "topic": 0,
                    "checkpoint": 1,
                }.get(pair[0][0], 2)
            )
            flat.extend(header)
            flat.extend(rest)
            flat.extend(reversed(lines))

        # Two-pass priority fill:
        #   1. Essentials (headers + topics) — keep newest-first up to budget
        #   2. Narrator lines — keep newest-first to fill what's left
        budget = _VISIBLE_HISTORY_ROWS
        keep_positions: set[int] = set()
        used_rows = 0

        # Pass 1: essentials in display order (newest items first since
        # we already reversed groups). If we'd overflow, oldest items'
        # essentials drop first — but the deque cap should make this
        # rare in practice.
        for pos, (entry, _idx) in enumerate(flat):
            if entry[0] in ("header", "topic", "checkpoint"):
                row_cost = self._entry_visual_rows(entry, wrap_width)
                if used_rows >= budget:
                    break
                if used_rows > 0 and used_rows + row_cost > budget:
                    break
                keep_positions.add(pos)
                used_rows += row_cost

        # Pass 2: narrator lines, sorted by RECENCY (highest deque idx
        # first), to fill the remaining budget. This drops oldest
        # narrator lines first when an item produces more lines than
        # the budget can hold.
        optionals = [
            (pos, entry, idx)
            for pos, (entry, idx) in enumerate(flat)
            if entry[0] not in ("header", "topic", "checkpoint")
        ]
        optionals.sort(key=lambda t: -t[2])  # newest first

        for pos, _entry, _idx in optionals:
            row_cost = self._entry_visual_rows(_entry, wrap_width)
            if used_rows >= budget:
                break
            if used_rows > 0 and used_rows + row_cost > budget:
                continue
            keep_positions.add(pos)
            used_rows += row_cost

        # Build the final list in original (top-to-bottom) display order
        display: list[tuple[tuple, bool, int]] = []
        group_depth = -1
        for pos, (entry, idx) in enumerate(flat):
            if pos in keep_positions:
                if entry[0] == "header":
                    group_depth = 0
                else:
                    group_depth = max(0, group_depth + 1)
                display.append((entry, idx == most_recent_idx, group_depth))
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
        total_elapsed_s = (
            int(max(0.0, now - self._session_started_at))
            if self._session_started_at is not None
            else 0
        )
        turn_elapsed_s = (
            int(max(0.0, now - self._turn_started_at))
            if self._turn_started_at is not None
            else None
        )
        header_text.append(
            f"total={total_elapsed_s}s",
            style="#7f95cf" if self._session_started_at is not None else "grey50",
        )
        header_text.append("  ", style="dim")
        header_text.append(
            f"turn={turn_elapsed_s}s" if turn_elapsed_s is not None else "turn=--",
            style=_rgb_to_hex(_EMBER_ACCENT_RGB) if turn_elapsed_s is not None else "grey50",
        )
        header_text.append("  ", style="dim")
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

        scorebug_panel = None
        if self.current_model or self.current_item_bug or self.current_set_label:
            scorebug_top = Text()
            self._append_scorebug_cell(
                scorebug_top,
                "CURRENT MODEL",
                self.current_model or "—",
                label_style="bold #eaf2ff on #405a93",
                value_style="bold #d8e5ff on #27344f",
            )
            if self.current_set_label:
                set_value = self.current_set_label
                self._append_scorebug_cell(
                    scorebug_top,
                    "SET",
                    set_value,
                    label_style="bold #eef6ff on #32506e",
                    value_style="bold #d7e8ff on #1d3147",
                )
            if self.current_item_bug:
                self._append_scorebug_cell(
                    scorebug_top,
                    "ITEM",
                    self.current_item_bug,
                    label_style="bold #fff1e6 on #7d4a2e",
                    value_style=f"bold #fff6ef on {_rgb_to_hex(_EMBER_ACCENT_RGB)}",
                )

            scorebug_gap = Text(" ", style="grey35")
            scorebug_rows: list[Text] = [scorebug_top, scorebug_gap]
            if self.score_points_possible > 0:
                scorebug_labels = Text()
                scorebug_values_top = Text()
                scorebug_values_middle = Text()
                scorebug_values_bottom = Text()
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "ON TARGET",
                    (
                        f"{self._format_scorebug_points(self.score_on_target_points)}"
                        f"/{self._format_scorebug_points(self.score_points_possible)}"
                    ),
                    label_style="bold #eef3ff on #32578e",
                    value_style="bold #dce9ff on #1c2d47",
                    value_mid_style="bold #8ea5cb on #1c2d47",
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "LEFT ON TABLE",
                    (
                        f"{self._format_scorebug_points(self.score_left_on_table_points)}"
                        f"/{self._format_scorebug_points(self.score_left_on_table_potential)}"
                    ),
                    label_style="bold #fff1d6 on #6b5028",
                    value_style="bold #ffefcf on #3e2f1b",
                    value_mid_style="bold #a18f66 on #3e2f1b",
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "BAD CALLS",
                    (
                        f"{self._format_scorebug_points(self.score_bad_call_points)}"
                        f"/{self._format_scorebug_points(self.score_bad_call_potential)}"
                    ),
                    label_style="bold #ffe5dd on #7a392f",
                    value_style="bold #ffe3d8 on #47211d",
                    value_mid_style="bold #a57e76 on #47211d",
                )
                scorebug_rows.extend(
                    [
                        scorebug_labels,
                        scorebug_values_top,
                        scorebug_values_middle,
                        scorebug_values_bottom,
                    ]
                )
            else:
                scorebug_labels = Text()
                scorebug_values_top = Text()
                scorebug_values_middle = Text()
                scorebug_values_bottom = Text()
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "ON TARGET",
                    "0.0/0.0",
                    label_style="bold #eef3ff on #32578e",
                    value_style="bold #dce9ff on #1c2d47",
                    value_mid_style="bold #8ea5cb on #1c2d47",
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "LEFT ON TABLE",
                    "0.0/0.0",
                    label_style="bold #fff1d6 on #6b5028",
                    value_style="bold #ffefcf on #3e2f1b",
                    value_mid_style="bold #a18f66 on #3e2f1b",
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "BAD CALLS",
                    "0.0/0.0",
                    label_style="bold #ffe5dd on #7a392f",
                    value_style="bold #ffe3d8 on #47211d",
                    value_mid_style="bold #a57e76 on #47211d",
                )
                scorebug_rows.extend(
                    [
                        scorebug_labels,
                        scorebug_values_top,
                        scorebug_values_middle,
                        scorebug_values_bottom,
                    ]
                )

            scorebug_panel = Panel(
                Align.left(Group(*scorebug_rows)),
                border_style="#3d4458",
                padding=(0, 1),
            )

        # Top panel — cool sticky status rail above the warmer live line.
        # Status is persistent and structural, so it gets the calmer
        # indigo-steel shimmer. The live first-person line below it is
        # the more volatile surface, so it carries the warmer active
        # treatment without overwriting the sticky status.
        displayed_live = self.streaming_line or self.frozen_line
        is_active = bool(self.streaming_line)
        live_palette_variant = "warm" if (
            self._line_parity if is_active else self._frozen_line_parity
        ) else "cool"

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
            cursor_style = _rgb_to_hex(_EMBER_ACCENT_RGB) if is_active else "grey50"
            live_text.append("▌ ", style=cursor_style)
            freeze_age_s = None
            if not is_active and self._freeze_started_at is not None:
                freeze_age_s = time.monotonic() - self._freeze_started_at
            _render_live_undulating(
                live_text,
                displayed_live,
                indent_width=2,
                wrap_width=wrap_width,
                is_active=is_active,
                palette_variant=live_palette_variant,
                char_offset=live_char_offset,
                freeze_age_s=freeze_age_s,
            )
        else:
            live_text = Text(no_wrap=False, overflow="fold")
            live_text.append("▌ ", style="grey39")
            _render_live_undulating(
                live_text,
                _live_placeholder(now),
                indent_width=2,
                wrap_width=wrap_width,
                is_active=False,
                palette_variant=live_palette_variant,
                char_offset=0,
                freeze_age_s=None,
            )

        status_text = Text(no_wrap=False, overflow="fold")
        displayed_status = self.status_streaming_line or self.status_line
        if displayed_status:
            displayed_status = displayed_status.upper()
            status_gutter_rgb = _interp_rgb(
                _BASE_RGB["status"],
                _SHIMMER_KIND_PEAK_RGB["status"],
                0.28,
            )
            status_text.append("▌ ", style=_rgb_to_hex(status_gutter_rgb))
            _render_status_undulating(
                status_text,
                displayed_status,
                indent_width=2,
                wrap_width=wrap_width,
            )
        else:
            status_text.append("▌ ", style="grey39")
            status_text.append("AWAITING STATUS", style="grey50")

        live_panel = Panel(
            Group(status_text, live_text),
            border_style="#3d4458",
            padding=(0, 1),
            title="[grey50]status + live[/grey50]",
            title_align="left",
            # Fixed height: top border + content + bottom border.
            # Locks the live panel's vertical footprint so the layout
            # doesn't jitter when bonsai produces a long line.
            height=_TOP_PANEL_CONTENT_LINES + 2,
        )

        focus_preview_panel = None
        have_kitty = self.focus_preview_kitty_renderable is not None
        have_inline = self.focus_preview_inline_renderable is not None
        have_fallback = self.focus_preview_renderable is not None
        if have_kitty and not self.focus_preview_pending:
            # Kitty graphics, steady state: use the place-by-ID
            # renderable DIRECTLY (not wrapped in a Panel, same
            # reasoning as FocusPreviewInlineImage). The PNG was
            # transmitted to the terminal on the event thread
            # inside on_focus_preview, so by the time this render
            # fires the cache is typically ready. The place
            # command is tiny and flicker-free.
            focus_preview_panel = self.focus_preview_kitty_renderable
        elif have_inline and not self.focus_preview_pending:
            # Inline image, steady state: use the custom renderable
            # DIRECTLY, not wrapped in a Panel. Panel's Padding layer
            # writes literal spaces to the cells on either side of
            # the content, which lands on exactly the cells WezTerm
            # just painted image pixels into, overwriting the image.
            # The renderable draws its own border + title and uses
            # cursor-forward escapes (not spaces) to advance past
            # the image cells without touching them.
            focus_preview_panel = self.focus_preview_inline_renderable
        elif have_kitty or have_inline or have_fallback:
            preview_title = "[grey50]focus preview"
            if self.focus_preview_pending:
                preview_title += " · pending"
            if self.focus_preview_label:
                preview_title += f" · {self.focus_preview_label}"
            preview_title += "[/grey50]"
            # Pending or fallback path: use the half-block renderer
            # wrapped in a Panel. During the pending transition the
            # inline image path falls through to the half-block
            # animation because the inline path is static; if no
            # fallback pixels are available (because the inline path
            # was active and replaced them), we show an empty panel
            # with the title, which is still informative.
            if self.focus_preview_pending and have_fallback and self.focus_preview_pixels is not None:
                pending_bucket = int(now * _FOCUS_PREVIEW_PENDING_FPS)
                if (
                    self._focus_preview_pending_renderable is None
                    or self._focus_preview_pending_bucket != pending_bucket
                ):
                    self._focus_preview_pending_renderable = _render_focus_preview_pixels(
                        self.focus_preview_pixels,
                        now=now,
                        pending=True,
                    )
                    self._focus_preview_pending_bucket = pending_bucket
                preview_renderable = self._focus_preview_pending_renderable
            elif have_fallback:
                preview_renderable = self.focus_preview_renderable
            else:
                # Have inline pending but no fallback — just show a
                # blank placeholder, the next focus_preview event
                # will swap it for the real inline image.
                preview_renderable = Text(
                    "(preview loading…)", style="grey50 italic"
                )
            focus_preview_panel = Panel(
                Align.center(preview_renderable),
                border_style="#3d4458",
                padding=(0, 1),
                title=preview_title,
                title_align="left",
            )

        # History panel — items grouped by header. Each item is a
        # group: header at the top, decision/topic directly below it,
        # then narrator lines newest-first beneath that. Groups are
        # rendered newest-first, so the current item sits at the top
        # of the panel and older items sink below as new items start.
        #
        # Layer index for shimmer is visual position (0 = topmost),
        # so the current item's header gets the brightest shimmer and
        # fades downward through the visible layers.
        #
        # Wrap-aware shimmer: when a long line wraps inside the panel
        # to multiple visual rows, the shimmer is computed by VISUAL
        # COLUMN (modulo wrap_width) so the wave stays in phase across
        # the wrap.
        display_entries = self._build_display_entries(wrap_width=wrap_width)
        history_text = Text(no_wrap=False, overflow="fold")
        global_history_phase = self._shimmer_phases.phase(0)
        current_group_index = -1
        current_group_base_phase = 0.0
        for i, (entry, is_most_recent, group_depth) in enumerate(display_entries):
            kind = entry[0]
            text = entry[1]
            parity = entry[2] if len(entry) > 2 else None
            if kind == "header":
                current_group_index += 1
                current_group_base_phase = self._shimmer_phases.phase(current_group_index)
            render_layer = _render_layer_index(kind, group_depth)
            if i > 0:
                history_text.append("\n")

            # Most-recently-committed entry pulses on the FAST cycle;
            # everything else uses the slowed default. Strong contrast.
            entry_cycle = (
                _SHIMMER_RECENT_CYCLE_S
                if is_most_recent
                else _SHIMMER_DEFAULT_CYCLE_S
            )

            # Keep a coherent local shimmer field within an item, but
            # terrace headers forward a little and rake reasoning back
            # more aggressively inside that item.
            phase_override = _history_entry_phase(
                current_group_base_phase,
                global_history_phase,
                current_group_index,
                group_depth,
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
                    layer_index=render_layer,
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
                        layer_index=render_layer,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                    history_text.append(" ", style="grey39")
                    _apply_shimmer(
                        history_text, rest_part, "header",
                        layer_index=render_layer,
                        indent_width=len(indent) + len(index_part) + 1,
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                else:
                    _apply_shimmer(
                        history_text, text, "header",
                        layer_index=render_layer,
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
                # entries). Deep indigo for matches, warm vermilion
                # for grader-overshot, warm ochre for grader-undershot,
                # bone fallback when verdict is unknown.
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
                    history_text.append(
                        time_prefix,
                        style=f"bold {_rgb_to_hex(_EMBER_ACCENT_RGB)}",
                    )
                    history_text.append("  ·  ", style="grey50")
                    extra_indent = len(time_prefix) + len("  ·  ")
                    _apply_shimmer(
                        history_text, rest, topic_kind,
                        layer_index=render_layer,
                        indent_width=len(indent) + extra_indent,
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
                else:
                    _apply_shimmer(
                        history_text, text, topic_kind,
                        layer_index=render_layer,
                        indent_width=len(indent),
                        wrap_width=wrap_width,
                        cycle_s=entry_cycle,
                        phase_override=phase_override,
                    )
            elif kind == "checkpoint":
                indent = "  ≈ "
                _apply_shimmer(
                    history_text, indent, "checkpoint_mark",
                    layer_index=render_layer,
                    indent_width=0,
                    wrap_width=wrap_width,
                    cycle_s=entry_cycle,
                    phase_override=phase_override,
                )
                checkpoint_kind = "checkpoint_alt" if parity == 1 else "checkpoint"
                _apply_shimmer(
                    history_text, text, checkpoint_kind,
                    layer_index=render_layer,
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
                    layer_index=render_layer,
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
            visible_drops = list(self.drops)[-_VISIBLE_DROP_LINES:]
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

        # Order: scorebug, header, live, history, post-game, drops, [footer]
        # The big tally cells are the most glanceable session-state
        # surface, so they lead the stack. The project header drops
        # below them as show identity rather than primary telemetry.
        panels = []
        if scorebug_panel is not None:
            panels.append(scorebug_panel)
        panels.append(header)
        panels.append(live_panel)
        if focus_preview_panel is not None:
            panels.append(focus_preview_panel)
        panels.append(history_panel)
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
        header_now = time.monotonic()
        if self._session_started_at is None:
            self._session_started_at = header_now
        self._turn_started_at = header_now
        self.status_line = ""
        self.status_streaming_line = ""
        self.streaming_line = ""
        self.frozen_line = ""
        self._frozen_line_parity = self._line_parity
        self._freeze_started_at = None
        if (
            self.focus_preview_renderable is not None
            or self.focus_preview_inline_renderable is not None
            or self.focus_preview_kitty_renderable is not None
        ):
            self.focus_preview_pending = True
            self.focus_preview_pending_started = header_now
            self._focus_preview_pending_bucket = None
            self._focus_preview_pending_renderable = None
        else:
            self.focus_preview_png = None
            self.focus_preview_pixels = None
            self.focus_preview_renderable = None
            self.focus_preview_inline_renderable = None
            self.focus_preview_kitty_renderable = None
            self.focus_preview_label = ""
            self.focus_preview_source = ""
            self.focus_preview_pending = False
            self.focus_preview_pending_started = None
            self._focus_preview_pending_bucket = None
            self._focus_preview_pending_renderable = None
        m = _HEADER_INDEX_RE.match(text)
        if m:
            self.current_item_bug = m.group(1).removeprefix("[item ").removesuffix("]").upper()
        self.history.append(("header", text, None))

    def on_session_meta(
        self,
        *,
        model: str | None = None,
        set_label: str | None = None,
        subset_count: int | None = None,
    ) -> None:
        if model:
            self.current_model = model
        if set_label:
            self.current_set_label = set_label
        if subset_count is not None:
            self.current_subset_count = subset_count

    def on_delta(self, text: str, mode: str = "thought") -> None:
        if mode == "status":
            self.status_streaming_line += text
        else:
            self.streaming_line += text

    def on_commit(self, mode: str = "thought") -> None:
        # Thought commits now stay in the sticky live lane only.
        # Durable history is carried by headers, topic lines, and
        # checkpoints. Status commits update only the sticky status rail.
        if self.status_streaming_line and mode == "status":
            self.status_line = self.status_streaming_line
            self.status_streaming_line = ""
        elif self.streaming_line:
            committed_parity = self._line_parity
            self._frozen_line_parity = committed_parity
            self._line_parity = 1 - self._line_parity
            self.stat_emitted += 1
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
        if reason.startswith("dedup"):
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
        self.status_streaming_line = ""

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

    def on_focus_preview(
        self,
        png_bytes: bytes,
        *,
        label: str = "",
        source: str = "",
    ) -> None:
        term_width = None
        if self._console is not None:
            try:
                term_width = self._console.size.width
            except Exception:
                term_width = None
        self.focus_preview_png = png_bytes
        self.focus_preview_label = label
        self.focus_preview_source = source
        self.focus_preview_pending = False
        self.focus_preview_pending_started = None
        self._focus_preview_pending_bucket = None
        self._focus_preview_pending_renderable = None

        # Kitty graphics protocol path: preferred on WezTerm/kitty.
        # Upload the PNG chunks directly to stdout NOW, on the
        # narrator event thread, so WezTerm starts decoding the
        # image in the background before Rich's next refresh fires.
        # Then build a FocusPreviewKittyImage renderable that
        # computes its cell box at RENDER time from the current
        # console dimensions — so the image resizes correctly
        # across terminal window resizes without requiring a
        # fresh focus_preview event.
        if self._kitty_graphics_supported:
            try:
                pix = fitz.Pixmap(png_bytes)
                title = f"focus preview · {label}" if label else "focus preview"
                # Transmit the PNG chunks directly to the console
                # output stream. Bypasses Rich entirely and runs
                # on the event thread, so the decode has a head
                # start on the next Rich refresh.
                transmit_stream = (
                    self._console.file if self._console is not None else sys.stdout
                )
                chunks = _build_kitty_transmit_chunks(png_bytes, _KITTY_IMAGE_ID)
                for chunk in chunks:
                    transmit_stream.write(chunk)
                transmit_stream.flush()
                self.focus_preview_kitty_renderable = FocusPreviewKittyImage(
                    image_id=_KITTY_IMAGE_ID,
                    image_pixel_width=pix.width,
                    image_pixel_height=pix.height,
                    terminal_cell_aspect=self._terminal_cell_aspect,
                    title=title,
                )
                # No need to build the iTerm2 or half-block paths —
                # the Kitty path owns the panel when it's available.
                self.focus_preview_inline_renderable = None
                self.focus_preview_pixels = None
                self.focus_preview_renderable = None
                return
            except Exception:
                # If anything in the Kitty path fails (fitz quirk,
                # stdout write error, etc.), fall through to the
                # iTerm2 OSC 1337 path so the operator still sees
                # something.
                self.focus_preview_kitty_renderable = None

        # iTerm2 OSC 1337 path: fallback for terminals that don't
        # support Kitty graphics, or when the Kitty path failed to
        # build. Known flicker issue (see FocusPreviewInlineImage
        # docstring) but better than nothing.
        if self._inline_images_supported:
            try:
                pix = fitz.Pixmap(png_bytes)
                cell_width, cell_height = _compute_inline_image_cell_dimensions(
                    pix.width,
                    pix.height,
                    max_cell_height=_INLINE_IMAGE_CELL_HEIGHT,
                    max_cell_width=_INLINE_IMAGE_MAX_CELL_WIDTH,
                    terminal_cell_aspect=self._terminal_cell_aspect,
                )
                title = f"focus preview · {label}" if label else "focus preview"
                self.focus_preview_inline_renderable = FocusPreviewInlineImage(
                    png_bytes=png_bytes,
                    cell_width=cell_width,
                    cell_height=cell_height,
                    title=title,
                )
                self.focus_preview_kitty_renderable = None
                self.focus_preview_pixels = None
                self.focus_preview_renderable = None
                return
            except Exception:
                self.focus_preview_inline_renderable = None

        # Half-block fallback path: for terminals that don't support
        # inline images at all. Sample at 2× vertical density so the
        # steady-state renderer can pack a top/bottom half-pixel
        # pair into each terminal row via half-block (▀) cells.
        budget_width, budget_height = _focus_preview_budget(term_width)
        self.focus_preview_pixels = _build_focus_preview_pixels(
            png_bytes,
            max_width_chars=budget_width,
            max_height_rows=budget_height * 2,
        )
        self.focus_preview_renderable = _render_focus_preview_pixels(
            self.focus_preview_pixels
        )
        self.focus_preview_inline_renderable = None
        self.focus_preview_kitty_renderable = None

    def on_topic(
        self,
        text: str,
        verdict: str | None = None,
        *,
        grader_score: float | None = None,
        truth_score: float | None = None,
        max_points: float | None = None,
    ) -> None:
        # Topic (after-action) lands in history. Doesn't touch live
        # buffers — frozen_line keeps showing the last bonsai line.
        # The verdict ("match" / "overshoot" / "undershoot" / None)
        # is stored in the third tuple slot so the renderer can pick
        # the right color variant for the topic line.
        self.history.append(("topic", text, verdict))
        if (
            grader_score is None
            or truth_score is None
            or max_points is None
        ):
            return
        self.score_points_possible += max_points
        if abs(grader_score - truth_score) < 1e-9:
            self.score_on_target_points += truth_score
            return
        if grader_score < truth_score:
            self.score_left_on_table_points += truth_score - grader_score
            self.score_left_on_table_potential += truth_score
            return
        if grader_score > truth_score:
            self.score_bad_call_points += grader_score - truth_score
            self.score_bad_call_potential += max(0.0, max_points - truth_score)

    def on_checkpoint(self, text: str) -> None:
        checkpoint_parity = next(
            (
                parity
                for kind, _text, parity in reversed(self.history)
                if kind == "line" and parity is not None
            ),
            self._line_parity,
        )
        self.history.append(("checkpoint", text, checkpoint_parity))


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
            refresh_per_second=int(_ACTIVE_ANIMATION_FPS),
            screen=False,
            auto_refresh=False,
        ) as live:
            def _wait_for_manual_close() -> int:
                while True:
                    if not display.should_animate():
                        try:
                            live.update(display.render(), refresh=True)
                        except Exception:
                            pass
                    try:
                        ready, _, _ = select.select([sys.stdin], [], [], _IDLE_POLL_S)
                    except Exception:
                        time.sleep(_IDLE_POLL_S)
                        continue
                    if not ready:
                        continue
                    try:
                        line = sys.stdin.readline()
                    except Exception:
                        line = ""
                    if line:
                        animation_stop.set()
                        anim_thread.join(timeout=0.5)
                        return 0

            def _animation_tick():
                while not animation_stop.is_set():
                    if display.should_animate():
                        try:
                            live.update(display.render(), refresh=True)
                        except Exception:
                            # Transient race with the message loop mutating
                            # display state — next tick will recover.
                            pass
                        time.sleep(1.0 / _ACTIVE_ANIMATION_FPS)
                    else:
                        time.sleep(_IDLE_POLL_S)

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
                    elif msg_type == "session_meta":
                        display.on_session_meta(
                            model=msg.get("model"),
                            set_label=msg.get("set_label"),
                            subset_count=msg.get("subset_count"),
                        )
                    elif msg_type == "focus_preview":
                        try:
                            png_bytes = base64.b64decode(
                                msg.get("png_base64", ""),
                            )
                        except Exception:
                            png_bytes = b""
                        if png_bytes:
                            display.on_focus_preview(
                                png_bytes,
                                label=msg.get("label", ""),
                                source=msg.get("source", ""),
                            )
                    elif msg_type == "delta":
                        display.on_delta(
                            msg.get("text", ""),
                            mode=msg.get("mode", "thought"),
                        )
                    elif msg_type == "commit":
                        display.on_commit(msg.get("mode", "thought"))
                    elif msg_type == "rollback_live":
                        display.on_rollback_live()
                    elif msg_type == "topic":
                        display.on_topic(
                            msg.get("text", ""),
                            verdict=msg.get("verdict"),
                            grader_score=msg.get("grader_score"),
                            truth_score=msg.get("truth_score"),
                            max_points=msg.get("max_points"),
                        )
                    elif msg_type == "checkpoint":
                        display.on_checkpoint(msg.get("text", ""))
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
                        display.session_ended = True
                        display._session_ended_at = time.monotonic()
                        live.update(display.render(), refresh=True)
                        return _wait_for_manual_close()

                    if _message_requires_immediate_refresh(msg_type):
                        try:
                            live.update(display.render(), refresh=True)
                        except Exception:
                            pass
    finally:
        animation_stop.set()
        try:
            fifo.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
