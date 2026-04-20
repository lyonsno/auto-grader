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
import random
import select
import signal
import sys
import termios
import threading
import time
import tty
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

# Keep the full structured-row family recorded locally near the reader
# surface, even while implementation is still partial, so scrollback
# archiving and in-pane rendering share one stable vocabulary.
_LEGIBILITY_STRUCTURED_ROW_LABELS = {
    "basis": "Basis",
    "ambiguity": "Ambiguity",
    "credit_preserved": "Credit preserved for",
    "deduction": "Deduction",
    "review_marker": "Review needed",
    "professor_mismatch": "Professor mismatch",
}
_LEGIBILITY_STRUCTURED_ROW_ORDER = {
    "basis": 1,
    "ambiguity": 2,
    "credit_preserved": 3,
    "deduction": 4,
    "review_marker": 5,
    "professor_mismatch": 6,
}
_LEGIBILITY_STRUCTURED_ROW_KINDS = frozenset(_LEGIBILITY_STRUCTURED_ROW_LABELS)


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
    "line_alt": (198, 186, 168), # warm bone — the alternating row should
                                  # read clearly against the moss row,
                                  # not collapse into muddy ochre
    "topic": (220, 205, 180),    # warm bone — fallback when verdict is
                                  # unknown / no prediction data. Bone's
                                  # structural home outside the live field
    "header": (156, 52, 62),     # lacquered burgundy — red-led enough to
                                  # read warmer at a glance, but still dark
                                  # enough that the plum undertone shows up
                                  # as a secondary accent rather than the
                                  # whole header drifting purple
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
    "checkpoint_alt": (202, 190, 170),    # anchored warm bone checkpoint —
                                          # brighter alternating companion
                                          # so the stack keeps the visible
                                          # moss/bone cadence the operator
                                          # expects under the header
    "checkpoint_mark": (162, 114, 82),    # embered rust notch — structural
                                          # mark for checkpoint rows so the
                                          # checkpoint doesn't begin with a
                                          # dead grey gutter
    "topic_overshoot": (210, 90, 65),     # vermilion (朱色) — too generous
    "topic_undershoot": (188, 154, 98),   # wheat ale — too strict,
                                          # softened away from signal-gold
                                          # toward a browner paper-earth
    # Header dash — vermilion stroke at the start of every item header.
    # Gives vermilion a STRUCTURAL home (was the only verdict color
    # appearing purely as a verdict indicator) and pulses in sync with
    # the rest of the header so the painting reads as one stroke per
    # item: vermilion dash → indigo index → persimmon title.
    "header_dash": (210, 118, 78),
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
    "header": (214, 104, 122),    # fired burgundy crest — visibly redder
                                   # than the base, but still carrying plum
                                   # on the high end so the header gets a
                                   # subtle internal wine/plum undulation
    "header_index": (185, 210, 240),  # rain-cleared sky blue — indigo
                                       # brightens toward the pale sky
                                       # after a storm wash painting
    "line": (175, 215, 180),      # glazed celadon — sage moss row
                                   # brightens toward kiln-glaze green
    "line_alt": (232, 220, 198),  # lit bone crest — the warm alternating
                                   # row should brighten within the bone
                                   # family instead of flashing ochre
    "topic_match": (132, 160, 224),     # rain-lit deep-indigo crest for
                                        # agreement lines
    "checkpoint": (176, 204, 180),      # brighter celadon crest —
                                        # still in the history family, just
                                        # a touch more settled than live rows
    "checkpoint_alt": (236, 222, 198),  # brighter bone crest for the
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
    "topic_undershoot": (228, 192, 136), # wheat-lit earth — warm,
                                         # but no longer a bright gold flare
    "header_dash": (234, 152, 108),      # fired apricot-vermilion — the dash
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
_LIVE_BASE_SAT = 0.28              # softened back down so the cool lane reads
                                   # like washed mineral color, not electric light
_LIVE_BASE_VAL = 0.80              # dimmer for legibility and to sit inside the
                                   # paper/ink world instead of above it
_LIVE_WARM_HUE_CENTER_DEG = 22     # yellow-red sibling, friendlier than a hot
                                   # alarm band but more chromatic than before
_LIVE_WARM_HUE_RANGE_DEG = 18
_LIVE_WARM_BASE_SAT = 0.34         # a touch more pigment so the warm lane reads
                                   # as color instead of almost-white
_LIVE_WARM_BASE_VAL = 0.82         # still pastel, but no longer teasing the eye
                                   # from the edge of white
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
_PREVIEW_ANIMATION_FPS = 24.0  # now that the preview band is a precomposed
                               # Kitty image (~30 bytes/frame instead of
                               # ~12KB of text segments), there is no reason
                               # to throttle below the active animation rate.
                               # The original 10fps throttle made the shimmer
                               # look sluggish because it was tuned for 24fps.
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
    "0": ("╔═╗", "╠ ╣", "╚═╝"),
    "1": ("╔╗ ", " ║ ", " ╹ "),
    "2": ("╔═╗", "╔═╝", "╚═ "),
    "3": ("╔═╗", " ═╣", "╚═╝"),
    "4": ("╔ ╗", "╚═╣", "  ╹"),
    "5": ("╔═ ", "╚═╗", "╚═╝"),
    "6": ("╔═ ", "╠═╗", "╚═╝"),
    "7": ("╔═╗", "╔╝ ", "║  "),
    "8": ("╔═╗", "╠═╣", "╚═╝"),
    "9": ("╔═╗", "╚═╣", "  ╝"),
    ".": ("   ", "   ", " ▪ "),
    "/": ("   ", " ╱ ", "   "),
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


def _live_frame_requires_full_clear(
    last_paint_size: tuple[int, int] | None,
    current_size: tuple[int, int],
) -> bool:
    """Return whether the next live paint needs a global alt-screen clear.

    The expensive full-screen clear is only necessary when there is no
    previous frame yet or when terminal geometry changed. Stable same-size
    repaints can cursor-home and let Rich redraw in place, which keeps the
    lively surface moving without hauling the whole alternate screen through
    ESC[2J on every tick.
    """
    return last_paint_size is None or current_size != last_paint_size


def suppress_live_erase(live: "Live") -> None:
    """Neuter Rich Live's per-row CSI 2K erase so Kitty pixels survive.

    Rich's ``LiveRender.position_cursor()`` emits ``CSI 2K`` (erase
    entire line) on every row of the previous frame before each refresh.
    That destroys Kitty image compositor pixels between frames, causing
    a visible strobe at 24 fps.

    The animation loop already manages alt-screen entry and cursor-home /
    full-clear positioning itself (``\\033[H`` or ``\\033[2J\\033[H``
    before each ``live.update``). Rich's erase is therefore redundant
    *and* destructive. This function replaces ``position_cursor`` with
    a no-op so the only per-frame positioning comes from our animation
    loop.
    """
    from rich.control import Control

    render = getattr(live, "_live_render", None)
    if render is not None:
        render.position_cursor = lambda: Control()  # type: ignore[attr-defined]


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


def _append_header_title(text: Text, title: str, phase: float) -> None:
    """Keep the scene-setter title flat white so the color lives in the field below."""
    del phase
    text.append(title, style="bold bright_white")


def _append_scorebug_value_row(
    row: Text,
    content: str,
    *,
    strong_style: str,
    mid_style: str,
    texture_style: str,
    texture_seed: int,
) -> None:
    """Append one scoreboard value row with weighted strokes and sparse field texture."""
    strong_chars = {"╔", "╗", "╚", "╝", "║", "╠", "╣", "╩", "═", "▪"}
    mid_chars = {"╱"}
    for idx, ch in enumerate(content):
        if ch in strong_chars:
            row.append(ch, style=strong_style)
        elif ch in mid_chars:
            row.append(ch, style=mid_style)
        elif ch == " ":
            texture_char = _scorebug_texture_char(idx, texture_seed)
            if texture_char == " ":
                row.append(" ", style=texture_style)
            else:
                row.append(texture_char, style=texture_style)
        else:
            row.append(ch, style=strong_style)


def _scorebug_texture_char(slot_index: int, seed: int) -> str:
    """Return a sparse deterministic texture character for the scorebug field.

    The field should read like low-frequency terminal texture, not like a
    repeating wallpaper. Keep density low and avoid strong vertical glyphs.
    """
    group = slot_index // 3
    within = slot_index % 3
    mixed = (group * 17) + (seed * 13) + ((group // 4) * 7)
    bucket = mixed % 19
    if bucket in {0, 1, 2} and within in {1, 2}:
        return "░"
    if seed == 1 and bucket in {3, 4} and within in {0, 2}:
        return "·"
    if bucket in {5, 6} and within in {0, 1}:
        return "▒"
    if bucket in {9, 12, 15} and within == 1:
        return "·"
    if bucket == 17 and within in {0, 1, 2}:
        return "┈"
    return " "


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

    Headers stay full-strength so the eye can keep finding the item
    anchor quickly. Everything below the header participates in the
    local depth fade; otherwise the structured rows beneath a header
    flatten into one loud block instead of settling as they descend.
    """
    return 0 if kind == "header" else group_depth


def _message_requires_immediate_refresh(msg_type: str) -> bool:
    """Return whether a FIFO event should bypass the normal animation cadence.

    Regular stream events should let the animation loop own repaint timing so
    idle and active motion feel consistent. Only boundary moments that would
    feel laggy at 12 FPS get an immediate forced refresh.
    """
    return msg_type in {
        "session_meta",
        "focus_preview",
        "wrap_up",
        "basis",
        "review_marker",
        "end",
    }


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

#: Starting Kitty image ID used for focus-preview uploads. The live
#: display hands out fresh IDs from this seed so successive previews
#: don't rely on in-place overwrite of one cached image. That overwrite
#: path turned out to be a bad fit once the composite started using real
#: transparency: later previews could read as though old opaque pixels
#: were still hanging around. Fresh IDs make each preview a clean
#: compositor surface.
_KITTY_IMAGE_ID = 1

#: Base64 chunks must be at most this many characters. Kitty protocol
#: spec: "chunk size of 4096 for the base64-encoded data".
_KITTY_CHUNK_SIZE = 4096

#: Extra rows above and below the image inside the band. These rows
#: run full terminal width with texture but no image content. Two
#: means one row above the image and one row below.
_BAND_EXTRA_ROWS = 2

#: Number of solid-block columns hugging the image edge.
#: Column 0 = █ (bright, matches the extra rows above/below),
#: column 1 = █ (same glyph, color has started fading slightly),
#: column 2 = ▓, then braille starts. Three columns gives the
#: sides enough visual weight to read as a continuous frame with
#: the top/bottom extra rows.
_SOLID_COLUMNS = 3

#: Faint floor for both braille density and color intensity at the
#: terminal edges. 0.12 means the outermost cells carry ~12% of
#: peak density/color — faintly visible rather than invisible.
_TEXTURE_EDGE_FLOOR = 0.12

#: Texture accent color near the image edge — warm bone from the
#: narrator's moss/bone palette, replacing the earlier sepia tone.
_TEXTURE_ACCENT_RGB = (220, 205, 180)

#: Terminal background color the texture fades toward. Matches the
#: panel background in focus_preview.py and the dark narrator UI.
_TEXTURE_BG_RGB = (8, 10, 14)


#: Braille base codepoint — U+2800 is the empty braille pattern.
#: Add a bitmask in [0, 255] to get a braille char with that dot
#: combination lit. The bitmask follows the Unicode braille ordering:
#:   bit 0 = dot 1 (top-left), bit 1 = dot 2 (middle-left),
#:   bit 2 = dot 3 (bottom-left of main 3), bit 3 = dot 4 (top-right),
#:   bit 4 = dot 5 (middle-right), bit 5 = dot 6 (bottom-right),
#:   bit 6 = dot 7 (bottom-left extra), bit 7 = dot 8 (bottom-right extra).
_BRAILLE_BASE = 0x2800


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


#: Braille dot indices for left column (bits 0,1,2,6) and right
#: column (bits 3,4,5,7). Near the image edge, preferring vertical
#: stripe patterns creates a directional grain that echoes the
#: solid-block boundary.
_BRAILLE_LEFT_COL = (0, 1, 2, 6)
_BRAILLE_RIGHT_COL = (3, 4, 5, 7)


def _texture_cell(
    *,
    distance_from_image: int,
    max_distance: int,
    seed_key: tuple,
) -> tuple[str, tuple[int, int, int]]:
    """Pick a (glyph, rgb) pair for one texture cell."""
    d = max(0, distance_from_image)
    span = max(1, max_distance)
    t = min(1.0, d / span)
    falloff = (1.0 - t) ** 1.8
    intensity = _TEXTURE_EDGE_FLOOR + (1.0 - _TEXTURE_EDGE_FLOOR) * falloff

    rgb = _lerp_rgb(_TEXTURE_BG_RGB, _TEXTURE_ACCENT_RGB, intensity)

    if d < _SOLID_COLUMNS:
        glyph = "▓" if d == _SOLID_COLUMNS - 1 else "█"
        return glyph, rgb

    density = intensity
    rand = random.Random(hash(seed_key))

    if density <= _TEXTURE_EDGE_FLOOR + 0.01:
        if rand.random() < _TEXTURE_EDGE_FLOOR:
            glyph = chr(_BRAILLE_BASE + (1 << rand.randint(0, 7)))
            return glyph, rgb
        return " ", _TEXTURE_BG_RGB

    bias_zone = max(1, int(span * 0.2))
    braille_d = d - _SOLID_COLUMNS
    vertical_bias = max(0.0, 1.0 - braille_d / bias_zone)

    n_dots = max(1, min(8, int(round(density * 8))))
    bits = 0

    if rand.random() < vertical_bias:
        col = _BRAILLE_LEFT_COL if rand.random() < 0.5 else _BRAILLE_RIGHT_COL
        other = _BRAILLE_RIGHT_COL if col is _BRAILLE_LEFT_COL else _BRAILLE_LEFT_COL
        primary = list(col)
        secondary = list(other)
        rand.shuffle(primary)
        rand.shuffle(secondary)
        pool = primary + secondary
        for bit_idx in pool[:n_dots]:
            bits |= 1 << bit_idx
    else:
        dot_order = list(range(8))
        rand.shuffle(dot_order)
        for bit_idx in dot_order[:n_dots]:
            bits |= 1 << bit_idx

    glyph = chr(_BRAILLE_BASE + bits)
    return glyph, rgb


def _lerp_rgb(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linear interpolation between two RGB tuples."""
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
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


def _read_tty_key(fd: int) -> str | None:
    """Read one raw key byte from a TTY file descriptor.

    The history controls are single-byte bindings. Read from the file
    descriptor directly rather than through ``sys.stdin``'s buffered text
    wrapper so cbreak-mode input remains reliable under spawned-terminal
    stdin plumbing.
    """
    try:
        chunk = os.read(fd, 1)
    except OSError:
        return None
    if not chunk:
        return None
    return chunk.decode("latin-1", errors="ignore")


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


def _build_composite_band_png(
    crop_png_bytes: bytes,
    *,
    term_width: int,
    image_cell_width: int,
    image_cell_height: int,
    image_id: int,
    title: str,
    cell_px_w: int = 8,
    cell_px_h: int = 16,
) -> bytes:
    """Build a single PNG that composites the exam crop with the ornate
    textured band surround.

    The resulting image covers the full band at ``term_width`` cells
    wide and ``(image_cell_height + _BAND_EXTRA_ROWS + 2)`` cells tall
    (image rows + 1 extra row above + 1 extra row below + 2 border
    rows). The texture, borders, and exam crop are all baked into one
    image so the terminal receives a single Kitty placement instead of
    hundreds of styled text segments per frame.

    ``cell_px_w`` and ``cell_px_h`` set the pixel resolution per cell
    in the composite — the terminal scales the placed image to fit
    the cell footprint regardless of source resolution.
    """
    crop_png_bytes = _trim_edge_crop_matte(crop_png_bytes)

    image_left = max(0, (term_width - image_cell_width) // 2)
    image_right = image_left + image_cell_width
    band_cell_rows = image_cell_height + _BAND_EXTRA_ROWS + 2

    px_w = term_width * cell_px_w
    px_h = band_cell_rows * cell_px_h

    # Start with a transparent RGBA canvas. The textured band and border
    # rows are painted opaque; the image box interior is left transparent
    # unless occupied by the scaled crop. That lets the terminal's real
    # background show through in any letterboxed negative space instead of
    # baking in a near-match dark matte that never quite lines up.
    comp = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, px_w, px_h), 1)
    comp.clear_with()

    # --- Paint border rows ---
    border_rgba = (135, 160, 145, 255)  # moss, matches _emit_band_border_row
    _paint_border_row(comp, 0, term_width, cell_px_w, cell_px_h, border_rgba, title)
    _paint_border_row(
        comp, band_cell_rows - 1, term_width, cell_px_w, cell_px_h, border_rgba, "",
    )

    # --- Paint texture with glyph patterns ---
    # Braille dot layout: 2 columns × 4 rows within each cell.
    # Bit positions map to (col, row) in the dot grid:
    #   bit 0 → (0,0), bit 1 → (0,1), bit 2 → (0,2), bit 6 → (0,3)
    #   bit 3 → (1,0), bit 4 → (1,1), bit 5 → (1,2), bit 7 → (1,3)
    _BRAILLE_BIT_POSITIONS = [
        (0, 0), (0, 1), (0, 2),  # bits 0-2: left column rows 0-2
        (1, 0), (1, 1), (1, 2),  # bits 3-5: right column rows 0-2
        (0, 3),                   # bit 6: left column row 3
        (1, 3),                   # bit 7: right column row 3
    ]
    dot_w = max(1, cell_px_w // 2)
    dot_h = max(1, cell_px_h // 4)
    # Inset dots slightly for rounder appearance.
    dot_inset_x = max(0, (dot_w - max(1, dot_w * 2 // 3)) // 2)
    dot_inset_y = max(0, (dot_h - max(1, dot_h * 2 // 3)) // 2)

    for cell_row in range(1, band_cell_rows - 1):
        if cell_row == 1:
            row_seed_id = 0
        elif cell_row == band_cell_rows - 2:
            row_seed_id = 1 + image_cell_height
        else:
            row_seed_id = cell_row - 1

        for col in range(term_width):
            in_image_span = image_left <= col < image_right
            is_image_row = 2 <= cell_row <= 1 + image_cell_height
            if in_image_span and is_image_row:
                continue

            if col < image_left:
                distance = image_left - col
                max_dist = image_left
            elif col >= image_right:
                distance = col - image_right + 1
                max_dist = max(1, term_width - image_right)
            else:
                distance = 0
                max_dist = 1

            glyph, rgb = _texture_cell(
                distance_from_image=distance,
                max_distance=max_dist,
                seed_key=(image_id, row_seed_id, col),
            )

            x0 = col * cell_px_w
            y0 = cell_row * cell_px_h
            cell_rect = fitz.IRect(x0, y0, x0 + cell_px_w, y0 + cell_px_h)
            if in_image_span:
                comp.set_rect(cell_rect, (*_TEXTURE_BG_RGB, 255))
                continue
            # The textured surround should remain an opaque dark field.
            # Only the image-box negative space gets to be transparent.
            comp.set_rect(cell_rect, (*_TEXTURE_BG_RGB, 255))

            if glyph == " ":
                # Empty texture cell — leave the opaque dark background.
                continue
            elif glyph in ("█", "▓"):
                # Full or dense block — fill the entire cell.
                if glyph == "▓":
                    # ▓ is ~75% fill — paint a slightly inset rect.
                    inset = max(1, cell_px_w // 8)
                    comp.set_rect(
                        fitz.IRect(
                            x0 + inset, y0 + inset,
                            x0 + cell_px_w - inset, y0 + cell_px_h - inset,
                        ),
                        (*rgb, 255),
                    )
                else:
                    comp.set_rect(
                        cell_rect,
                        (*rgb, 255),
                    )
            elif ord(glyph) >= _BRAILLE_BASE:
                # Braille character — paint individual dots.
                bits = ord(glyph) - _BRAILLE_BASE
                for bit_idx in range(8):
                    if bits & (1 << bit_idx):
                        dcol, drow = _BRAILLE_BIT_POSITIONS[bit_idx]
                        dx = x0 + dcol * dot_w + dot_inset_x
                        dy = y0 + drow * dot_h + dot_inset_y
                        dw = dot_w - 2 * dot_inset_x
                        dh = dot_h - 2 * dot_inset_y
                        if dw > 0 and dh > 0:
                            comp.set_rect(
                                fitz.IRect(dx, dy, dx + dw, dy + dh),
                                (*rgb, 255),
                            )
            else:
                # Fallback: any other glyph — fill the cell.
                comp.set_rect(
                    cell_rect,
                    (*rgb, 255),
                )

    # --- Paste the exam crop centered in the image region ---
    # Build the final composite by rendering a PDF page that has the
    # texture background as a base image and the crop overlaid on top.
    # This avoids slow pixel-by-pixel Python loops — fitz handles the
    # scaling and compositing natively.
    crop_pix = fitz.Pixmap(crop_png_bytes)
    crop_x0 = image_left * cell_px_w
    crop_y0 = 2 * cell_px_h  # after top border + extra row
    crop_target_w = image_cell_width * cell_px_w
    crop_target_h = image_cell_height * cell_px_h

    # Scale the crop to COVER the image region, centered and clipped.
    # The earlier contain behavior left a subtle internal top/bottom
    # matte on wide crops that kept reading as letterboxing in smoke.
    src_w, src_h = crop_pix.width, crop_pix.height
    scale = max(crop_target_w / max(1, src_w), crop_target_h / max(1, src_h))
    scaled_w = max(1, int(src_w * scale))
    scaled_h = max(1, int(src_h * scale))

    # Use a PDF page to composite: background (texture) + foreground (crop).
    final_doc = fitz.open()
    final_page = final_doc.new_page(width=px_w, height=px_h)
    # Insert the texture background as the base layer.
    bg_png = comp.tobytes("png")
    final_page.insert_image(fitz.Rect(0, 0, px_w, px_h), stream=bg_png)

    # Cover-fill the preview box on a temporary page whose bounds are the
    # target image region itself. Any overflow from the scaled crop gets
    # clipped at the page edge, which is exactly the "crop to fill" we want.
    cover_doc = fitz.open()
    cover_page = cover_doc.new_page(width=crop_target_w, height=crop_target_h)
    cover_x0 = (crop_target_w - scaled_w) / 2.0
    cover_y0 = (crop_target_h - scaled_h) / 2.0
    cover_page.insert_image(
        fitz.Rect(cover_x0, cover_y0, cover_x0 + scaled_w, cover_y0 + scaled_h),
        stream=crop_png_bytes,
    )
    cover_png = cover_page.get_pixmap(alpha=True).tobytes("png")
    cover_doc.close()

    # Insert the clipped exam crop on top at the computed position.
    final_page.insert_image(
        fitz.Rect(crop_x0, crop_y0, crop_x0 + crop_target_w, crop_y0 + crop_target_h),
        stream=cover_png,
    )
    # Render the composited page to PNG.
    final_pix = final_page.get_pixmap(alpha=True)
    result = final_pix.tobytes("png")
    final_doc.close()

    return result


def _trim_near_black_crop_margins(
    crop_png_bytes: bytes,
    *,
    threshold: int = 20,
) -> bytes:
    """Trim contiguous near-black edge margins from a preview crop.

    Some focus-preview crops arrive with black scan matte or letterbox bars
    already baked in. If we scale those blindly, the preview reads like a
    warped black frame inside our own band. Trim only contiguous edges that
    are overwhelmingly near-black; interior dark content is left untouched.
    """
    pix = fitz.Pixmap(crop_png_bytes)
    width = pix.width
    height = pix.height
    channels = pix.n
    samples = pix.samples

    def _pixel_is_near_black(x: int, y: int) -> bool:
        off = (y * width + x) * channels
        r = samples[off]
        g = samples[off + 1]
        b = samples[off + 2]
        a = samples[off + 3] if channels >= 4 else 255
        return a == 0 or (r <= threshold and g <= threshold and b <= threshold)

    def _row_is_near_black(y: int) -> bool:
        return all(_pixel_is_near_black(x, y) for x in range(width))

    def _col_is_near_black(x: int) -> bool:
        return all(_pixel_is_near_black(x, y) for y in range(height))

    top = 0
    while top < height - 1 and _row_is_near_black(top):
        top += 1

    bottom = height - 1
    while bottom > top and _row_is_near_black(bottom):
        bottom -= 1

    left = 0
    while left < width - 1 and _col_is_near_black(left):
        left += 1

    right = width - 1
    while right > left and _col_is_near_black(right):
        right -= 1

    if top == 0 and left == 0 and right == width - 1 and bottom == height - 1:
        return crop_png_bytes

    new_w = right - left + 1
    new_h = bottom - top + 1
    trimmed = bytearray(new_w * new_h * channels)
    for row in range(new_h):
        src_start = ((top + row) * width + left) * channels
        src_end = src_start + new_w * channels
        dst_start = row * new_w * channels
        trimmed[dst_start : dst_start + new_w * channels] = samples[src_start:src_end]

    trimmed_pix = fitz.Pixmap(
        fitz.csRGB,
        new_w,
        new_h,
        bytes(trimmed),
        channels >= 4,
    )
    return trimmed_pix.tobytes("png")


def _trim_uniform_edge_margins(
    crop_png_bytes: bytes,
    *,
    color_tolerance: int = 28,
    coverage: float = 0.92,
) -> bytes:
    """Trim contiguous edge rows / columns matching the corner matte color.

    After black bars are removed, many exam crops still carry a uniform paper-
    colored scan border. Trim only the edge matte that matches the corner
    sample closely enough; interior paper remains untouched once text or
    drawings start breaking the edge rows/columns.
    """
    pix = fitz.Pixmap(crop_png_bytes)
    width = pix.width
    height = pix.height
    channels = pix.n
    samples = pix.samples

    def _rgba(x: int, y: int) -> tuple[int, int, int, int]:
        off = (y * width + x) * channels
        r = samples[off]
        g = samples[off + 1]
        b = samples[off + 2]
        a = samples[off + 3] if channels >= 4 else 255
        return (r, g, b, a)

    def _close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return (
            abs(a[0] - b[0]) <= color_tolerance
            and abs(a[1] - b[1]) <= color_tolerance
            and abs(a[2] - b[2]) <= color_tolerance
            and abs(a[3] - b[3]) <= color_tolerance
        )

    top_left = _rgba(0, 0)
    top_right = _rgba(width - 1, 0)
    bottom_left = _rgba(0, height - 1)
    bottom_right = _rgba(width - 1, height - 1)
    center = _rgba(width // 2, height // 2)

    # If the body of the crop already matches the corner sample, there
    # is no distinct paper-colored frame to remove. This guards against
    # collapsing a uniform crop down to a 1x1 swatch.
    if (
        _close(top_left, center)
        and _close(top_right, center)
        and _close(bottom_left, center)
        and _close(bottom_right, center)
    ):
        return crop_png_bytes

    def _row_matches(y: int, sample: tuple[int, int, int, int]) -> bool:
        hits = sum(1 for x in range(width) if _close(_rgba(x, y), sample))
        return hits / max(1, width) >= coverage

    def _col_matches(x: int, sample: tuple[int, int, int, int]) -> bool:
        hits = sum(1 for y in range(height) if _close(_rgba(x, y), sample))
        return hits / max(1, height) >= coverage

    top = 0
    while top < height - 1 and _row_matches(top, top_left):
        top += 1

    bottom = height - 1
    while bottom > top and _row_matches(bottom, bottom_left):
        bottom -= 1

    left = 0
    while left < width - 1 and _col_matches(left, top_left):
        left += 1

    right = width - 1
    while right > left and _col_matches(right, top_right):
        right -= 1

    if top == 0 and left == 0 and right == width - 1 and bottom == height - 1:
        return crop_png_bytes

    new_w = right - left + 1
    new_h = bottom - top + 1
    trimmed = bytearray(new_w * new_h * channels)
    for row in range(new_h):
        src_start = ((top + row) * width + left) * channels
        src_end = src_start + new_w * channels
        dst_start = row * new_w * channels
        trimmed[dst_start : dst_start + new_w * channels] = samples[src_start:src_end]

    trimmed_pix = fitz.Pixmap(
        fitz.csRGB,
        new_w,
        new_h,
        bytes(trimmed),
        channels >= 4,
    )
    return trimmed_pix.tobytes("png")


def _trim_edge_crop_matte(crop_png_bytes: bytes) -> bytes:
    """Remove both dark scan matte and uniform paper-colored edge bands."""
    trimmed = _trim_near_black_crop_margins(crop_png_bytes)
    return _trim_uniform_edge_margins(trimmed)


def _paint_border_row(
    pix: "fitz.Pixmap",
    cell_row: int,
    term_width: int,
    cell_px_w: int,
    cell_px_h: int,
    rgb: tuple[int, int, int],
    title: str,
) -> None:
    """Paint a border row: thin horizontal line with optional title.

    The original text-based ``_emit_band_border_row`` renders ``─``
    characters — a thin horizontal rule at the cell's vertical
    midline, with the cell background being the terminal's dark
    background (not a solid colored strip). We replicate that: a
    2px line at the vertical center of the cell row in the moss
    color, on the dark ``_TEXTURE_BG_RGB`` background. The title
    text is rendered in the moss color on the same dark background.
    """
    y0 = cell_row * cell_px_h
    y1 = y0 + cell_px_h
    row_w = min(term_width * cell_px_w, pix.width)
    if y1 <= y0 or row_w <= 0:
        return

    # Border/title rows should stay opaque dark even when the rest of the
    # composite canvas is RGBA-transparent. If we leave these rows
    # transparent, stale text from the previous frame ghosts through the
    # top strip when Rich repaints in place.
    bg_rgba = (*_TEXTURE_BG_RGB, 255)
    pix.set_rect(fitz.IRect(0, y0, row_w, y1), bg_rgba)

    # Paint a thin line on top of the dark strip.
    line_h = 2
    line_y = y0 + cell_px_h // 2 - line_h // 2
    pix.set_rect(fitz.IRect(0, line_y, row_w, line_y + line_h), rgb)

    if not title:
        return

    # Build the title string matching _emit_band_border_row format.
    prefix_text = f"\u2500 {title} "
    remaining = term_width - len(prefix_text)
    if remaining < 0:
        prefix_text = prefix_text[:term_width]
        remaining = 0
    border_text = prefix_text + ("\u2500" * remaining)

    # Render the title text on the dark background in moss color.
    bg_rgb_f = (
        _TEXTURE_BG_RGB[0] / 255,
        _TEXTURE_BG_RGB[1] / 255,
        _TEXTURE_BG_RGB[2] / 255,
    )
    text_rgb_f = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    try:
        txt_doc = fitz.open()
        txt_page = txt_doc.new_page(width=row_w, height=cell_px_h)
        txt_page.draw_rect(
            fitz.Rect(0, 0, row_w, cell_px_h),
            fill=bg_rgb_f,
        )
        fontsize = max(10, cell_px_h * 1.05)
        txt_page.insert_text(
            fitz.Point(0, cell_px_h * 0.82),
            border_text,
            fontsize=fontsize,
            color=text_rgb_f,
            fontname="courier-bold",
        )
        txt_pix = txt_page.get_pixmap(alpha=False)
        txt_doc.close()

        blit_w = min(txt_pix.width, row_w)
        blit_h = min(txt_pix.height, cell_px_h)
        for row in range(blit_h):
            for col in range(blit_w):
                pixel = txt_pix.pixel(col, row)
                pix.set_pixel(col, y0 + row, (pixel[0], pixel[1], pixel[2], 255))
    except Exception:
        pass  # Thin line is already there as fallback.


class FocusPreviewKittyImage:
    """Rich Renderable that places a precomposed Kitty image covering
    the full ornate preview band.

    The composite PNG — which includes the exam crop, the textured
    surround, and the border lines all baked into one image — is
    transmitted once via Kitty ``a=t`` on the event thread. This
    renderable's only job is to emit the tiny ``a=p`` place command
    (~30 bytes) and cursor-forward escapes on every frame so the
    terminal paints the cached composite at the right position.

    On terminal resize, the renderable detects that
    ``options.max_width`` no longer matches the width the composite
    was built for, rebuilds the composite at the new geometry, and
    retransmits it via Kitty before placing. This keeps the preview
    resize-safe without requiring a fresh ``on_focus_preview`` event.
    """

    def __init__(
        self,
        *,
        image_id: int,
        band_cell_width: int,
        band_cell_height: int,
        title: str = "",
        crop_png_bytes: bytes = b"",
        image_pixel_width: int = 0,
        image_pixel_height: int = 0,
        terminal_cell_aspect: float = _DEFAULT_TERMINAL_CELL_ASPECT,
    ) -> None:
        self._image_id = image_id
        self._band_cell_width = band_cell_width
        self._band_cell_height = band_cell_height
        self._title = title
        # Stored for resize rebuild.
        self._crop_png_bytes = crop_png_bytes
        self._image_pixel_width = image_pixel_width
        self._image_pixel_height = image_pixel_height
        self._terminal_cell_aspect = terminal_cell_aspect

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Emit the Kitty place command and pre-clear the preview rows.

        No styled text segments — the entire band is a single placed
        image. Rich sees cursor-forward control segments and newlines,
        which cost ~30 bytes total per frame instead of ~12KB.
        Always place at the dimensions the composite was built for.
        On resize, ``retransmit_kitty_image`` rebuilds the composite
        at the new geometry and updates ``_band_cell_width`` and
        ``_band_cell_height`` before the next frame renders. Between
        the resize and the rebuild there may be one frame where the
        composite overflows or underflows the terminal width — that
        is acceptable. Trying to rescale the placement to the current
        terminal width here causes aspect-ratio warping because the
        composite pixels don't match the recomputed cell dimensions.
        The ``a=p`` placement fires on every frame. Rich's Live
        erases each line (CSI 2K) between frames, which wipes the
        terminal cells the image was painted into. Without a fresh
        placement the image disappears. The placement command is
        ~30 bytes and tells the terminal to re-place the already-
        cached image — no PNG retransmission, just a cursor-position
        reference. The earlier rapid flicker was caused by transmit
        interleaving (now fixed via drain_pending_kitty_transmit),
        not by the placement itself.
        """
        # Emit full-width blank rows FIRST so the cells under any
        # transparent pixels in the composite are reset to the true
        # terminal background before placement. If we only walk the
        # cursor forward, stale history text survives under the
        # transparent letterbox or title-strip gaps once lower panels
        # get tall enough to reach those rows.
        #
        # Then save cursor, return to the start of the band, place the
        # image (which paints over the cleared cells in the compositor
        # layer), and restore cursor so Rich continues below the band.
        save_cursor = "\x1b[s"
        restore_cursor = "\x1b[u"
        # Move up to the start of the band after walking past it.
        move_up = f"\x1b[{self._band_cell_height}A"
        carriage_return = "\r"
        blank_row = " " * self._band_cell_width
        for row in range(self._band_cell_height):
            yield Segment(blank_row)
            yield Segment.line()
        place_sequence = _build_kitty_place_sequence(
            self._image_id,
            cell_width=self._band_cell_width,
            cell_height=self._band_cell_height,
        )
        # Save cursor (at bottom of band), move back to top-left of
        # band, place image, restore cursor to bottom.
        yield Segment(
            save_cursor + move_up + carriage_return + place_sequence + restore_cursor,
            None,
            [(ControlType.BELL,)],
        )


class FocusPreviewLoadingBand:
    """Placeholder renderable shown before any focus_preview event
    fires, or between items while a new preview is loading. Same
    band structure as :class:`FocusPreviewKittyImage` but with a
    short ``(preview loading…)`` text string where the image
    would be, so the layout slot looks continuous across the
    transition into the first real preview.
    """

    def __init__(self, *, title: str = "focus preview") -> None:
        self._title = title

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        term_width = max(1, options.max_width)

        # Scale with the terminal like the real Kitty renderer does.
        # Use a ~4:3 exam crop aspect ratio as the stand-in so the
        # placeholder box matches what the first real preview will
        # look like. The real renderer leaves 2 cells for borders
        # and caps at _INLINE_IMAGE_MAX_CELL_WIDTH.
        inner_budget = max(1, term_width - 2)
        cell_width = min(_INLINE_IMAGE_MAX_CELL_WIDTH, inner_budget)
        cell_height = _INLINE_IMAGE_CELL_HEIGHT
        image_left = max(0, (term_width - cell_width) // 2)
        image_right = image_left + cell_width

        placeholder_text = "(preview loading…)"
        # Loading band has no image id — use a stable sentinel so the
        # seeded per-cell texture noise is deterministic across frames
        # without colliding with real image ids.
        image_id = 0

        # Top border
        yield from _emit_band_border_row(term_width, title=self._title)

        # Extra texture row above.
        yield from _emit_band_texture_only_row(
            term_width,
            image_left=image_left,
            image_right=image_right,
            row_seed_id=0,
            image_id=image_id,
        )

        # Image rows — instead of a Kitty place, emit the
        # placeholder text centered in the middle of the image
        # region on the middle image row, empty on the others.
        middle_row = cell_height // 2
        text_col = image_left + max(
            0, (cell_width - len(placeholder_text)) // 2
        )
        dim_style = Style.parse(_rgb_to_hex(_TEXTURE_ACCENT_RGB) + " dim")
        bg_style = Style.parse("on " + _rgb_to_hex(_TEXTURE_BG_RGB))

        for image_row in range(cell_height):
            # Left texture
            yield from _emit_band_texture_span(
                col_start=0,
                col_end=image_left,
                image_left=image_left,
                image_right=image_right,
                term_width=term_width,
                row_seed_id=1 + image_row,
                image_id=image_id,
            )
            # Middle: placeholder text on middle row, empty on others.
            if image_row == middle_row:
                pad_left = max(0, text_col - image_left)
                pad_right = max(
                    0, cell_width - pad_left - len(placeholder_text)
                )
                yield Segment(" " * pad_left, bg_style)
                yield Segment(placeholder_text, dim_style)
                yield Segment(" " * pad_right, bg_style)
            else:
                yield Segment(" " * cell_width, bg_style)
            # Right texture
            yield from _emit_band_texture_span(
                col_start=image_right,
                col_end=term_width,
                image_left=image_left,
                image_right=image_right,
                term_width=term_width,
                row_seed_id=1 + image_row,
                image_id=image_id,
            )
            yield Segment.line()

        # Extra texture row below.
        yield from _emit_band_texture_only_row(
            term_width,
            image_left=image_left,
            image_right=image_right,
            row_seed_id=1 + cell_height,
            image_id=image_id,
        )
        # Bottom border rule.
        yield from _emit_band_border_row(term_width, title="")


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
        #   kind in {"line", "header", "topic", "basis", "review_marker", "checkpoint"}
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
        # Pending composite PNG to transmit on the animation thread.
        # Set by on_focus_preview (event thread), drained by
        # _live_update (animation thread) before the next frame.
        self._pending_kitty_transmit: bytes | None = None
        self._pending_kitty_image_id: int | None = None
        self._next_kitty_image_id: int = _KITTY_IMAGE_ID
        # Do not probe stdin for terminal cell aspect during startup.
        # The live history controls share that input path; a startup-time
        # terminal query is the wrong tradeoff if it can wedge or poison
        # interactive scrolling. Use the stable default here and leave any
        # future terminal query work to a dedicated, better-tested path.
        self._terminal_cell_aspect: float = _DEFAULT_TERMINAL_CELL_ASPECT
        # When True, render() shows the post-session footer and the
        # final frame stays static while waiting for Enter. This keeps
        # the non-Crispy end-of-session close affordance intact on the
        # current union surface.
        self.session_ended: bool = False
        self._session_ended_at: float | None = None
        self._session_started_at: float | None = None
        self._turn_started_at: float | None = None

        # In-pane history scroll viewport. Persistent across frames so
        # scroll state survives redraws and new commits while the
        # operator is scrolled up.
        self._viewport: HistoryViewport | None = None
        self._viewport_wrap_width: int | None = None
        self._viewport_synced_len: int = 0
        # Optional explicit wrap-width override for tests and for the
        # public viewport accessor when no console is attached.
        self._wrap_width_override: int | None = None

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
        separator_style: str = "dim",
        label_pad: int = 1,
        value_pad: int = 1,
    ) -> None:
        if row.plain:
            row.append("  ", style=separator_style)
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
        value_row_styles: tuple[str, str, str],
        value_mid_row_styles: tuple[str, str, str],
        value_texture_styles: tuple[str, str, str],
        separator_styles: tuple[str, str, str, str] = ("dim", "dim", "dim", "dim"),
        label_pad: int = 3,
        value_pad: int = 1,
    ) -> None:
        if label_row.plain:
            label_row.append("· ", style=separator_styles[0])
            value_top_row.append("· ", style=separator_styles[1])
            value_middle_row.append("· ", style=separator_styles[2])
            value_bottom_row.append("· ", style=separator_styles[3])
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
            strong_style=value_row_styles[0],
            mid_style=value_mid_row_styles[0],
            texture_style=value_texture_styles[0],
            texture_seed=0,
        )
        _append_scorebug_value_row(
            value_middle_row,
            f"{' ' * value_pad}{middle}{' ' * value_pad}",
            strong_style=value_row_styles[1],
            mid_style=value_mid_row_styles[1],
            texture_style=value_texture_styles[1],
            texture_seed=9,
        )
        _append_scorebug_value_row(
            value_bottom_row,
            f"{' ' * value_pad}{bottom}{' ' * value_pad}",
            strong_style=value_row_styles[2],
            mid_style=value_mid_row_styles[2],
            texture_style=value_texture_styles[2],
            texture_seed=2,
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

    def target_animation_fps(self) -> float:
        if self._has_steady_image_preview():
            return _PREVIEW_ANIMATION_FPS
        return _ACTIVE_ANIMATION_FPS

    def _has_steady_image_preview(self) -> bool:
        return (
            not self.focus_preview_pending
            and (
                self.focus_preview_inline_renderable is not None
                or self.focus_preview_kitty_renderable is not None
            )
        )

    def should_refresh_on_event(self, msg_type: str) -> bool:
        if not self._has_steady_image_preview():
            return _message_requires_immediate_refresh(msg_type)
        return msg_type in {
            "session_meta",
            "header",
            "focus_preview",
            "commit",
            "rollback_live",
            "topic",
            "basis",
            "review_marker",
            "checkpoint",
            "drop",
            "wrap_up_pending",
            "wrap_up",
            "end",
        }

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
        elif kind == "basis":
            prefix_width = len("  ≡ Basis: ")
        elif kind == "review_marker":
            prefix_width = len("  ! Review needed: ")
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
        groups = self._history_groups_with_indices(history_list)

        # Newest item on top. Within each group, keep the header first,
        # move the verdict/topic line directly underneath it for quick
        # scanning, then flip narrator lines so the freshest thought
        # sits closest to the decision and older thoughts descend.
        groups.reverse()

        # Flat list of (entry, deque_idx) in display order (top-down)
        flat: list[tuple[tuple, int]] = []
        for group in groups:
            flat.extend(self._ordered_group_pairs(group))

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
            if entry[0] in ("header", "topic", "basis", "review_marker", "checkpoint"):
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
            if entry[0] not in ("header", "topic", "basis", "review_marker", "checkpoint")
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

    @staticmethod
    def _history_groups_with_indices(
        history_list: list[tuple[str, str, int | None]],
    ) -> list[list[tuple[tuple[str, str, int | None], int]]]:
        groups: list[list[tuple[tuple[str, str, int | None], int]]] = []
        current_group: list[tuple[tuple[str, str, int | None], int]] = []
        for idx, entry in enumerate(history_list):
            if entry[0] == "header":
                if current_group:
                    groups.append(current_group)
                current_group = [(entry, idx)]
            else:
                current_group.append((entry, idx))
        if current_group:
            groups.append(current_group)
        return groups

    @staticmethod
    def _ordered_group_pairs(
        group: list[tuple[tuple[str, str, int | None], int]],
    ) -> list[tuple[tuple[str, str, int | None], int]]:
        header = [pair for pair in group if pair[0][0] == "header"]
        lines = [pair for pair in group if pair[0][0] == "line"]
        rest = [pair for pair in group if pair[0][0] not in ("header", "line")]
        rest.sort(
            key=lambda pair: {
                "topic": 0,
                **_LEGIBILITY_STRUCTURED_ROW_ORDER,
                "checkpoint": 7,
            }.get(pair[0][0], 2)
        )
        return [*header, *rest, *reversed(lines)]

    # -- History viewport (Crispy Drips) --------------------------------

    def _flat_display_entries(
        self,
        *,
        wrap_width: int | None = None,
    ) -> list[tuple[tuple[str, str, int | None], bool]]:
        """Return the live-edge history selection in oldest->newest group order.

        The viewport should inherit the richer essentials-first and
        structured-row-aware display semantics we already use on the smoke
        surface, rather than downgrading back to a simpler header/topic-only
        filter just to gain scrolling.
        """
        display_entries = self._build_display_entries(wrap_width=wrap_width)
        if not display_entries:
            return []

        groups: list[list[tuple[tuple[str, str, int | None], bool]]] = []
        current_group: list[tuple[tuple[str, str, int | None], bool]] = []
        for entry, is_most_recent, _group_depth in display_entries:
            if entry[0] == "header":
                if current_group:
                    groups.append(current_group)
                current_group = [(entry, is_most_recent)]
            else:
                current_group.append((entry, is_most_recent))
        if current_group:
            groups.append(current_group)

        out: list[tuple[tuple[str, str, int | None], bool]] = []
        for group in reversed(groups):
            out.extend(group)
        return out

    def _resolve_wrap_width(self) -> int:
        wrap = self._compute_wrap_width()
        if wrap is None:
            wrap = self._wrap_width_override or 80
        return wrap

    def _sync_viewport(self) -> HistoryViewport:
        """Ensure the viewport reflects current history and wrap width."""
        wrap_width = self._resolve_wrap_width()
        flat = self._flat_display_entries(wrap_width=wrap_width)
        flat_entries = [entry for entry, _ in flat]

        rebuild = (
            self._viewport is None
            or self._viewport_wrap_width != wrap_width
        )

        if not rebuild and self._viewport is not None:
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
                visible_rows=_VISIBLE_HISTORY_ROWS,
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
            new_entries = flat_entries[self._viewport_synced_len :]
            for entry in new_entries:
                self._viewport.append(entry)
            self._viewport_synced_len = len(flat_entries)
        return self._viewport

    def history_viewport(self) -> HistoryViewport:
        """Public accessor for the synced history viewport."""
        return self._sync_viewport()

    def _viewport_display_entries(
        self,
    ) -> list[tuple[tuple[str, str, int | None], bool, int]]:
        """Return the viewport's visible slice in render order."""
        vp = self._sync_viewport()
        visible = vp.visible_entries()
        if not visible:
            return []

        history_list = list(self.history)
        most_recent_entry = history_list[-1] if history_list else None

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

        groups.reverse()
        out: list[tuple[tuple[str, str, int | None], bool, int]] = []
        for group in groups:
            group_depth = -1
            for entry in group:
                if entry[0] == "header":
                    group_depth = 0
                else:
                    group_depth = max(0, group_depth + 1)
                out.append((entry, entry == most_recent_entry, group_depth))
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
        _append_header_title(header_text, self.title, self._shimmer_phases.phase(0))
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

        # Scoreboard-dial treatment for the three event-count counters.
        # EMITTED / DEDUP / EMPTY each render through
        # `_append_scorebug_cell` so they reuse the same capsule-and-
        # value language the Liquid Varnish Squadron scorebug panel
        # below uses for CURRENT MODEL / ITEM, instead of sitting in
        # the top chrome as a flat telemetry tail. The event-count
        # dials light up on green / amber / red capsules only when
        # nonzero — at zero they fall back to a muted grey capsule so
        # a quiet run doesn't scream color.
        #
        # The two timers (TOTAL and TURN) are NOT rendered here. They
        # are promoted below into full tall-digit scorebug plates
        # alongside ON TARGET / LEFT ON TABLE / BAD CALLS so the
        # timer promotion is actually legible as dial-shape scoreboard
        # instrumentation rather than being lost in the top header
        # chrome.
        _emitted_idle_label = "bold #e7f8db on #306525"
        _emitted_idle_value = "bold #d9eecd on #1a3915"
        _emitted_label = "bold #f4ffea on #4a9838"
        _emitted_value = "bold #eaffdf on #23561a"
        _dedup_idle_label = "bold #f2f8cc on #627121"
        _dedup_idle_value = "bold #e2eeb7 on #3b4514"
        _dedup_label = "bold #fbffd7 on #809326"
        _dedup_value = "bold #f1f7bf on #4c5718"
        _empty_idle_label = "bold #f6d9d0 on #753025"
        _empty_idle_value = "bold #e8c1b8 on #491a15"
        _empty_label = "bold #ffe4dc on #a13f2f"
        _empty_value = "bold #ffd1c7 on #5f2019"
        self._append_scorebug_cell(
            header_text,
            "EMITTED",
            f"{self.stat_emitted}",
            label_style=(
                _emitted_label if self.stat_emitted > 0 else _emitted_idle_label
            ),
            value_style=(
                _emitted_value if self.stat_emitted > 0 else _emitted_idle_value
            ),
        )
        self._append_scorebug_cell(
            header_text,
            "DEDUP",
            f"{self.stat_dropped_dedup}",
            label_style=(
                _dedup_label
                if self.stat_dropped_dedup > 0
                else _dedup_idle_label
            ),
            value_style=(
                _dedup_value
                if self.stat_dropped_dedup > 0
                else _dedup_idle_value
            ),
        )
        self._append_scorebug_cell(
            header_text,
            "EMPTY",
            f"{self.stat_dropped_empty}",
            label_style=(
                _empty_label
                if self.stat_dropped_empty > 0
                else _empty_idle_label
            ),
            value_style=(
                _empty_value
                if self.stat_dropped_empty > 0
                else _empty_idle_value
            ),
        )
        header = Panel(
            Align.left(header_text),
            border_style="#3d4458",
            padding=(0, 1),
        )

        scorebug_panel = None
        if self.current_model or self.current_item_bug or self.current_set_label:
            meta_bg = "#3a362d"
            meta_separator_style = "bold #7a725f on #3a362d"
            tally_label_bg = "#3b382e"
            tally_label_separator_style = "bold #877d62"
            tally_top_separator_style = "bold #7f765d"
            tally_mid_separator_style = "bold #766d56"
            tally_bottom_separator_style = "bold #6e654f"
            tally_top_bg = "#363328"
            tally_mid_bg = "#322f25"
            tally_bottom_bg = "#2f2b22"
            on_target_value_strong_styles = (
                "bold #e0ebf2",
                "bold #d4e1ea",
                "bold #c8d6e0",
            )
            on_target_value_mid_styles = (
                "bold #a1b4c2",
                "bold #96a9b8",
                "bold #8c9fac",
            )
            on_target_value_texture_styles = (
                "#718697",
                "#697d8d",
                "#617485",
            )
            left_table_value_strong_styles = (
                "bold #efe2c4",
                "bold #dbc08d",
                "bold #d0b07d",
            )
            left_table_value_mid_styles = (
                "bold #b08f62",
                "bold #a48259",
                "bold #977550",
            )
            left_table_value_texture_styles = (
                "#846c4d",
                "#796247",
                "#6d5841",
            )
            bad_calls_value_strong_styles = (
                "bold #ead6d0",
                "bold #d1ada2",
                "bold #bc8f83",
            )
            bad_calls_value_mid_styles = (
                "bold #aa857a",
                "bold #9d796f",
                "bold #906d64",
            )
            bad_calls_value_texture_styles = (
                "#7f645c",
                "#745a52",
                "#694f49",
            )
            scorebug_top = Text()
            self._append_scorebug_cell(
                scorebug_top,
                "CURRENT MODEL",
                self.current_model or "—",
                label_style=f"bold #c9c1b6 on {meta_bg}",
                value_style=f"bold #ece7de on {meta_bg}",
                separator_style=meta_separator_style,
            )
            if self.current_set_label:
                set_value = self.current_set_label
                self._append_scorebug_cell(
                    scorebug_top,
                    "SET",
                    set_value,
                    label_style=f"bold #c9c1b6 on {meta_bg}",
                    value_style=f"bold #e7decb on {meta_bg}",
                    separator_style=meta_separator_style,
                )
            if self.current_item_bug:
                self._append_scorebug_cell(
                    scorebug_top,
                    "ITEM",
                    self.current_item_bug,
                    label_style=f"bold #c9c1b6 on {meta_bg}",
                    value_style=f"bold #efe0cf on {meta_bg}",
                    separator_style=meta_separator_style,
                )

            scorebug_gap = Text(" ", style="grey35")
            scorebug_rows: list[Text] = [scorebug_top, scorebug_gap]

            # Timer plate styles (Gauge Saints II). These two plates
            # sit leftmost in the big-value strip so the run chrome
            # reads before the grading tally. TOTAL gets a quiet
            # slate/graphite family so it anchors the strip without
            # competing with ON TARGET's crisper blue. TURN gets a
            # warm sand/umber family that reads distinct from both
            # LEFT ON TABLE's yellow-bronze and BAD CALLS' red-brown,
            # and carries the "currently on the clock" weight that
            # matches the ember ITEM tag above.
            _total_label_style = f"bold #c0d4d8 on {tally_label_bg}"
            _total_value_row_styles = (
                "bold #dde9e6",
                "bold #d1e0dd",
                "bold #c6d6d3",
            )
            _total_value_mid_row_styles = (
                "bold #97aaa6",
                "bold #8ea19c",
                "bold #849691",
            )
            _total_value_texture_styles = (
                "#748984",
                "#6a7d79",
                "#60726e",
            )
            _turn_label_style = f"bold #dfb57d on {tally_label_bg}"
            _turn_value_row_styles = (
                "bold #f1d5a2",
                "bold #e5c792",
                "bold #d9b983",
            )
            _turn_value_mid_row_styles = (
                "bold #b19168",
                "bold #a5845f",
                "bold #987751",
            )
            _turn_value_texture_styles = (
                "#846e50",
                "#796449",
                "#6d5a42",
            )
            _total_value_str = f"{total_elapsed_s}"
            _turn_value_str = (
                f"{turn_elapsed_s}" if turn_elapsed_s is not None else "--"
            )

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
                    "TOTAL",
                    _total_value_str,
                    label_style=_total_label_style,
                    value_row_styles=_total_value_row_styles,
                    value_mid_row_styles=_total_value_mid_row_styles,
                    value_texture_styles=_total_value_texture_styles,
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "TURN",
                    _turn_value_str,
                    label_style=_turn_label_style,
                    value_row_styles=_turn_value_row_styles,
                    value_mid_row_styles=_turn_value_mid_row_styles,
                    value_texture_styles=_turn_value_texture_styles,
                )
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
                    label_style=f"bold #a8b8bb on {tally_label_bg}",
                    value_row_styles=on_target_value_strong_styles,
                    value_mid_row_styles=on_target_value_mid_styles,
                    value_texture_styles=on_target_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
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
                    label_style=f"bold #c0aa83 on {tally_label_bg}",
                    value_row_styles=left_table_value_strong_styles,
                    value_mid_row_styles=left_table_value_mid_styles,
                    value_texture_styles=left_table_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
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
                    label_style=f"bold #bc9589 on {tally_label_bg}",
                    value_row_styles=bad_calls_value_strong_styles,
                    value_mid_row_styles=bad_calls_value_mid_styles,
                    value_texture_styles=bad_calls_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
                )
                scorebug_rows.extend(
                    [
                        scorebug_labels,
                        scorebug_values_top,
                        scorebug_values_middle,
                        scorebug_values_bottom,
                        Text(" ", style="grey35"),
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
                    "TOTAL",
                    _total_value_str,
                    label_style=_total_label_style,
                    value_row_styles=_total_value_row_styles,
                    value_mid_row_styles=_total_value_mid_row_styles,
                    value_texture_styles=_total_value_texture_styles,
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "TURN",
                    _turn_value_str,
                    label_style=_turn_label_style,
                    value_row_styles=_turn_value_row_styles,
                    value_mid_row_styles=_turn_value_mid_row_styles,
                    value_texture_styles=_turn_value_texture_styles,
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "ON TARGET",
                    "0.0/0.0",
                    label_style=f"bold #a8b8bb on {tally_label_bg}",
                    value_row_styles=on_target_value_strong_styles,
                    value_mid_row_styles=on_target_value_mid_styles,
                    value_texture_styles=on_target_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "LEFT ON TABLE",
                    "0.0/0.0",
                    label_style=f"bold #c0aa83 on {tally_label_bg}",
                    value_row_styles=left_table_value_strong_styles,
                    value_mid_row_styles=left_table_value_mid_styles,
                    value_texture_styles=left_table_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
                )
                self._append_scorebug_big_value_cell(
                    scorebug_labels,
                    scorebug_values_top,
                    scorebug_values_middle,
                    scorebug_values_bottom,
                    "BAD CALLS",
                    "0.0/0.0",
                    label_style=f"bold #bc9589 on {tally_label_bg}",
                    value_row_styles=bad_calls_value_strong_styles,
                    value_mid_row_styles=bad_calls_value_mid_styles,
                    value_texture_styles=bad_calls_value_texture_styles,
                    separator_styles=(
                        tally_label_separator_style,
                        tally_top_separator_style,
                        tally_mid_separator_style,
                        tally_bottom_separator_style,
                    ),
                )
                scorebug_rows.extend(
                    [
                        scorebug_labels,
                        scorebug_values_top,
                        scorebug_values_middle,
                        scorebug_values_bottom,
                        Text(" ", style="grey35"),
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
        display_entries = self._viewport_display_entries()
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
            elif kind == "basis":
                indent = "  ≡ "
                history_text.append(indent, style="grey50")
                history_text.append(
                    "Basis: ",
                    style=f"bold {_rgb_to_hex(_EMBER_ACCENT_RGB)}",
                )
                _apply_shimmer(
                    history_text, text, "checkpoint_alt",
                    layer_index=render_layer,
                    indent_width=len(indent) + len("Basis: "),
                    wrap_width=wrap_width,
                    cycle_s=entry_cycle,
                    phase_override=phase_override,
                )
            elif kind == "review_marker":
                indent = "  ! "
                history_text.append(indent, style="grey50")
                history_text.append(
                    "Review needed: ",
                    style=f"bold {_rgb_to_hex(_EMBER_ACCENT_RGB)}",
                )
                _apply_shimmer(
                    history_text, text, "checkpoint",
                    layer_index=render_layer,
                    indent_width=len(indent) + len("Review needed: "),
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
                title="[grey42]rejected[/grey42]",
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

        # Order: header, scorebug, live, history, post-game, drops, [footer]
        # The PROJECT PAINT DRY band is the primary scene-setter again;
        # the scorebug stays immediately below it as the denser
        # instrumentation slab.
        panels = []
        panels.append(header)
        if scorebug_panel is not None:
            panels.append(scorebug_panel)
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

    def _allocate_kitty_image_id(self) -> int:
        image_id = self._next_kitty_image_id
        self._next_kitty_image_id += 1
        return image_id

    def drain_pending_kitty_transmit(self) -> None:
        """Transmit a pending composite PNG on the animation thread.

        Called from ``_live_update`` before the next frame so Kitty
        escape sequences don't interleave with Rich's output.
        """
        png = self._pending_kitty_transmit
        if png is None:
            return
        image_id = self._pending_kitty_image_id
        self._pending_kitty_transmit = None
        self._pending_kitty_image_id = None
        if image_id is None:
            return
        stream = self._console.file if self._console is not None else sys.stdout
        try:
            for chunk in _build_kitty_transmit_chunks(png, image_id):
                stream.write(chunk)
            stream.flush()
        except Exception:
            pass

    def retransmit_kitty_image(self) -> None:
        """Rebuild and re-upload the Kitty composite at current geometry.

        Called on terminal resize to flush the terminal's image
        compositing layer and rebuild the composite at the new
        terminal width. The renderable stores the original crop PNG
        and image dimensions so the composite can be rebuilt without
        a fresh ``on_focus_preview`` event.
        """
        if not self._kitty_graphics_supported:
            return
        if self.focus_preview_kitty_renderable is None:
            return
        rend = self.focus_preview_kitty_renderable
        if not rend._crop_png_bytes:
            return
        stream = self._console.file if self._console is not None else sys.stdout
        try:
            console_width = self._console.size.width if self._console else 120
            inner_budget = max(1, console_width - 2)
            image_cw, image_ch = _compute_inline_image_cell_dimensions(
                rend._image_pixel_width,
                rend._image_pixel_height,
                max_cell_height=_INLINE_IMAGE_CELL_HEIGHT,
                max_cell_width=min(_INLINE_IMAGE_MAX_CELL_WIDTH, inner_budget),
                terminal_cell_aspect=rend._terminal_cell_aspect,
            )
            band_cell_rows = image_ch + _BAND_EXTRA_ROWS + 2
            composite_png = _build_composite_band_png(
                rend._crop_png_bytes,
                term_width=console_width,
                image_cell_width=image_cw,
                image_cell_height=image_ch,
                image_id=rend._image_id,
                title=rend._title,
            )
            image_id = self._allocate_kitty_image_id()
            for chunk in _build_kitty_transmit_chunks(composite_png, image_id):
                stream.write(chunk)
            stream.flush()
            # Don't overwrite focus_preview_png — it holds the raw
            # crop bytes (pipeline contract). The composite is ephemeral.
            rend._image_id = image_id
            rend._band_cell_width = console_width
            rend._band_cell_height = band_cell_rows
        except Exception:
            # Leaving the stale Kitty renderable in place produces a
            # visibly stretched preview with no local recovery until a
            # future focus_preview event arrives. Drop back to the
            # non-Kitty path immediately so the operator keeps a
            # truthful preview surface instead of a corrupted one.
            self._pending_kitty_transmit = None
            self.focus_preview_kitty_renderable = None
            self._kitty_graphics_supported = False
            self.on_focus_preview(
                rend._crop_png_bytes,
                label=self.focus_preview_label,
                source=self.focus_preview_source,
            )
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
        # Keep pending=True during composite build + transmit so
        # render() doesn't try to place a mid-upload image. The
        # Kitty path clears pending after the renderable is fully
        # ready; non-Kitty paths clear it below.
        self.focus_preview_pending = True
        self.focus_preview_pending_started = time.monotonic()
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
                # Compute image cell box at the current console width.
                console_width = (
                    self._console.size.width
                    if self._console is not None
                    else 120
                )
                inner_budget = max(1, console_width - 2)
                image_cw, image_ch = _compute_inline_image_cell_dimensions(
                    pix.width,
                    pix.height,
                    max_cell_height=_INLINE_IMAGE_CELL_HEIGHT,
                    max_cell_width=min(
                        _INLINE_IMAGE_MAX_CELL_WIDTH, inner_budget,
                    ),
                    terminal_cell_aspect=self._terminal_cell_aspect,
                )
                band_cell_rows = image_ch + _BAND_EXTRA_ROWS + 2
                # Build the composite PNG: exam crop + ornate texture
                # + borders all baked into one image.
                composite_png = _build_composite_band_png(
                    png_bytes,
                    term_width=console_width,
                    image_cell_width=image_cw,
                    image_cell_height=image_ch,
                    image_id=self._next_kitty_image_id,
                    title=title,
                )
                image_id = self._allocate_kitty_image_id()
                # Don't transmit from the event thread — that interleaves
                # Kitty escape sequences with Rich's output on the
                # animation thread, producing visible garbage. Instead,
                # store the composite for the animation thread to
                # transmit before the next frame via _live_update.
                self._pending_kitty_transmit = composite_png
                self._pending_kitty_image_id = image_id
                # focus_preview_png keeps the raw incoming PNG bytes
                # (pipeline contract). The composite is stored on the
                # renderable and rebuilt by retransmit_kitty_image.
                self.focus_preview_kitty_renderable = FocusPreviewKittyImage(
                    image_id=image_id,
                    band_cell_width=console_width,
                    band_cell_height=band_cell_rows,
                    title=title,
                    crop_png_bytes=png_bytes,
                    image_pixel_width=pix.width,
                    image_pixel_height=pix.height,
                    terminal_cell_aspect=self._terminal_cell_aspect,
                )
                # No need to build the iTerm2 or half-block paths —
                # the Kitty path owns the panel when it's available.
                self.focus_preview_inline_renderable = None
                self.focus_preview_pixels = None
                self.focus_preview_renderable = None
                # Composite is fully transmitted and renderable is
                # assigned — clear pending so render() places it.
                self.focus_preview_pending = False
                return
            except Exception:
                # If anything in the Kitty path fails (fitz quirk,
                # stdout write error, etc.), fall through to the
                # iTerm2 OSC 1337 path so the operator still sees
                # something.
                self._pending_kitty_image_id = None
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
                self.focus_preview_pending = False
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
        self.focus_preview_pending = False

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

    def on_basis(self, text: str) -> None:
        self.history.append(("basis", text, None))

    def on_review_marker(self, text: str) -> None:
        self.history.append(("review_marker", text, None))


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
        # screen=True — use the terminal's alternate screen buffer.
        # Alt-screen gives us clean resize handling: Rich redraws the
        # full screen on every frame instead of trying to diff against
        # reflowed main-buffer text. Previously reverted because the
        # rendered height could exceed the terminal's row count, but
        # render() now caps the history panel via a height budget so
        # the total output never exceeds console.size.height.
        # Only clear the alt-screen when geometry changed or the very
        # first frame still needs a clean slate. Rich redraws from the
        # top on each paint, so stable same-size frames can just cursor-
        # home and repaint in place. That preserves the resize safety we
        # want without paying the global ESC[2J tax on every animation
        # tick while a steady preview is already settled.
        # Resize coordination.  SIGWINCH fires on the main thread
        # (signal delivery) while the animation thread is writing to
        # stdout via live.update().  Writing \033[2J from the signal
        # handler interleaves with Rich's output and corrupts the
        # frame.  Instead, the handler only sets a flag and wakes the
        # animation thread; all stdout writes happen on one thread.
        _resize_pending = threading.Event()
        # Wakeup event: the animation thread sleeps on this instead
        # of time.sleep() so SIGWINCH can cut the sleep short.
        _animation_wake = threading.Event()
        _paint_lock = threading.Lock()

        _prev_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _on_sigwinch(signum, frame):
            _resize_pending.set()
            _animation_wake.set()  # wake the animation thread NOW
            if callable(_prev_sigwinch):
                _prev_sigwinch(signum, frame)

        signal.signal(signal.SIGWINCH, _on_sigwinch)

        _last_paint_size: tuple[int, int] | None = None

        def _live_update():
            """Render, clear, and paint one frame.

            Render and paint are size-checked: the terminal size is
            snapshotted before render and verified again after.  If
            the size changed during render (resize arrived mid-
            layout), re-render at the new size.  Up to 3 retries to
            converge during rapid resize; after that, paint whatever
            we have (a briefly stale frame is better than dropping
            a frame entirely).

            On resize, Kitty graphics images are deleted and
            re-transmitted so the terminal's image compositing layer
            doesn't retain stale pixels from the previous geometry.

            All stdout writes happen here on the animation thread,
            never from the signal handler, so there is no interleaving.
            """
            nonlocal _last_paint_size
            try:
                with _paint_lock:
                    _resize_pending.clear()

                    # Drain any pending Kitty transmit from the event thread
                    # before painting so escape sequences don't interleave
                    # with Rich's output.
                    display.drain_pending_kitty_transmit()

                    cur_size = (console.size.width, console.size.height)
                    if (
                        _live_frame_requires_full_clear(
                            _last_paint_size, cur_size
                        )
                        and _last_paint_size is not None
                    ):
                        display.retransmit_kitty_image()
                    # Also retransmit if the composite was built at a different
                    # width than the current terminal — happens when
                    # on_focus_preview fires during a resize or the terminal
                    # was resized after the event thread read console.size.
                    else:
                        _krend = display.focus_preview_kitty_renderable
                        if (
                            _krend is not None
                            and _krend._band_cell_width != cur_size[0]
                        ):
                            display.retransmit_kitty_image()

                    renderable = None
                    for _attempt in range(3):
                        size_before = (console.size.width, console.size.height)
                        renderable = display.render()
                        size_after = (console.size.width, console.size.height)
                        if size_before == size_after:
                            break  # geometry stable — safe to paint

                    paint_size = (console.size.width, console.size.height)
                    # Buffer the entire frame into a single write so the
                    # terminal receives padding spaces and the deferred Kitty
                    # a=p placement atomically. Without buffering, Rich writes
                    # the padding in small chunks, the terminal clears image
                    # cells, and the a=p arrives in a later write — producing
                    # a visible flicker between the clear and the re-place.
                    _real_file = console.file
                    _real_write = _real_file.write
                    _frame_parts: list[str] = []
                    if _live_frame_requires_full_clear(
                        _last_paint_size, paint_size
                    ):
                        _frame_parts.append("\033[2J\033[H")
                    else:
                        _frame_parts.append("\033[H")
                    _real_file.write = _frame_parts.append  # type: ignore[assignment]
                    try:
                        live.update(renderable, refresh=True)
                    finally:
                        _real_file.write = _real_write  # type: ignore[assignment]
                    _real_write("".join(_frame_parts))
                    _real_file.flush()
                    _last_paint_size = paint_size
            finally:
                _animation_wake.clear()

        # Enter alt-screen manually.  We use screen=False on Live so
        # Rich doesn't wrap our renderable in Screen (which pads to a
        # height that can be stale during resize).  Our _live_update
        # does its own clear + cursor-home on every frame.
        console.file.write("\033[?1049h")
        console.file.flush()
        with Live(
            display.render(),
            console=console,
            refresh_per_second=int(_ACTIVE_ANIMATION_FPS),
            screen=False,
            auto_refresh=False,
        ) as live:
            # Suppress Rich's per-row CSI 2K erase.  We manage alt-screen
            # and cursor-home ourselves; Rich's erase is redundant and
            # destroys Kitty image compositor pixels between frames.
            suppress_live_erase(live)

            def _wait_for_manual_close() -> int:
                while True:
                    if not display.should_animate():
                        try:
                            _live_update()
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
                    if display.should_animate() or _resize_pending.is_set():
                        try:
                            _live_update()
                        except Exception:
                            # Transient race with the message loop mutating
                            # display state — next tick will recover.
                            pass
                        _animation_wake.clear()
                        _animation_wake.wait(
                            timeout=1.0 / display.target_animation_fps()
                        )
                    else:
                        _animation_wake.clear()
                        _animation_wake.wait(timeout=_IDLE_POLL_S)

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
                    if stdin_fd is None:
                        return
                    ch = _read_tty_key(stdin_fd)
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
                    elif msg_type == "basis":
                        display.on_basis(msg.get("text", ""))
                    elif msg_type == "review_marker":
                        display.on_review_marker(msg.get("text", ""))
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
                        _live_update()
                        if stdin_fd is None:
                            session_exit.set()
                        session_exit.wait()
                        animation_stop.set()
                        scroll_stop.set()
                        anim_thread.join(timeout=0.5)
                        return 0

                    if _message_requires_immediate_refresh(msg_type):
                        try:
                            _live_update()
                        except Exception:
                            pass
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
