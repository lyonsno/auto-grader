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
import select
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
    "1": (" ╗ ", " ║ ", " ╩ "),
    "2": ("╔═╗", "╔═╝", "╚═ "),
    "3": ("╔═╗", " ═╣", "╚═╝"),
    "4": ("║ ║", "╚═╣", "  ╩"),
    "5": ("╔═╗", "╠═ ", "╚═╝"),
    "6": ("╔═ ", "╠═╗", "╚═╝"),
    "7": ("╔═╗", "  ║", "  ╵"),
    "8": ("╔═╗", "╠═╣", "╚═╝"),
    "9": ("╔═╗", "╚═╣", "  ╝"),
    ".": ("   ", "   ", " ▪ "),
    "/": ("   ", " ╱ ", "╱  "),
    "-": ("   ", "═══", "   "),
}

_HISTORY_GROUP_SETBACK = 0.03      # lower item headers sit slightly behind the
                                   # one above them, but not by enough to read
                                   # as separate weather systems
_HISTORY_GROUP_RAKE = 0.028        # gentler within-item rake than the last pass
                                   # so the grouping reads structural, not
                                   # algorithmically terraced
_HISTORY_GROUP_ALT_FIELD = 0.012   # subtle secondary alternating shimmer field
                                   # shared across even/odd item groups
_HISTORY_GROUP_ALT_RATE = 0.55     # slower than the primary history field


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
        " ".join(top_parts),
        " ".join(middle_parts),
        " ".join(bottom_parts),
    )


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
    return msg_type in {"session_meta", "wrap_up", "end"}


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
        label_pad: int = 3,
        value_pad: int = 1,
    ) -> None:
        if label_row.plain:
            label_row.append("  ", style="dim")
            value_top_row.append("  ", style="dim")
            value_middle_row.append("  ", style="dim")
            value_bottom_row.append("  ", style="dim")
        label_row.append(
            f"{' ' * label_pad}{label}{' ' * label_pad}",
            style=label_style,
        )
        top, middle, bottom = _scorebug_big_value_rows(value)
        value_top_row.append(
            f"{' ' * value_pad}{top}{' ' * value_pad}",
            style=value_style,
        )
        value_middle_row.append(
            f"{' ' * value_pad}{middle}{' ' * value_pad}",
            style=value_style,
        )
        value_bottom_row.append(
            f"{' ' * value_pad}{bottom}{' ' * value_pad}",
            style=value_style,
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
            if entry[0] in ("header", "topic"):
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
            if entry[0] not in ("header", "topic")
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
        panels.extend([live_panel, history_panel])
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
        # Thought commits land in the history and become the sticky
        # frozen line. Status commits update only the sticky status rail.
        if self.status_streaming_line and mode == "status":
            self.status_line = self.status_streaming_line
            self.status_streaming_line = ""
        elif self.streaming_line:
            committed_parity = self._line_parity
            self.history.append(
                ("line", self.streaming_line, committed_parity)
            )
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
