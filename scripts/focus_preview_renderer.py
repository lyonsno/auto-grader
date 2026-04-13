"""Focus preview renderer for the Project Paint Dry narrator.

Extracted from narrator_reader.py to decouple the focus preview
rendering pipeline (Kitty graphics, textured band, image processing)
from the narrator's history/shimmer/scorebug code.
"""
from __future__ import annotations

import base64
import math
import os
import random
import select
import sys
import time

import fitz
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import ControlType, Segment
from rich.style import Style
from rich.text import Text


# ── Shared trivial utilities for the extracted preview renderer ─────
# narrator_reader re-exports a small subset of these helpers so the
# preview surface can keep a stable module seam without re-owning the
# implementations locally.


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interp_rgb(
    base: tuple[int, int, int],
    peak: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Backward-compatible alias for the module's rounded RGB interpolation."""
    return _lerp_rgb(base, peak, t)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pixel_luma(rgb: tuple[int, int, int]) -> float:
    return (0.299 * rgb[0]) + (0.587 * rgb[1]) + (0.114 * rgb[2])


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


# ── Focus preview constants ────────────────────────────────────────

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

# ── Inline image constants ──────────────────────────────────────────

#: Cell height the inline image path targets.
_INLINE_IMAGE_CELL_HEIGHT = 18

#: Max cell width the inline image path will request.
_INLINE_IMAGE_MAX_CELL_WIDTH = 140

#: Fallback terminal cell aspect ratio (height / width).
_DEFAULT_TERMINAL_CELL_ASPECT = 2.1

# ── Kitty graphics protocol constants ───────────────────────────────

#: Numeric Kitty image ID used for the focus preview cache.
_KITTY_IMAGE_ID = 1

#: Base64 chunk size for Kitty protocol.
_KITTY_CHUNK_SIZE = 4096

# ── Band texture parameters ─────────────────────────────────────────

#: Extra rows above and below the image inside the band.
_BAND_EXTRA_ROWS = 2

#: Number of solid-block columns hugging the image edge.
_SOLID_COLUMNS = 3

#: Faint floor for both braille density and color intensity at the
#: terminal edges.
_TEXTURE_EDGE_FLOOR = 0.12

#: Texture accent color near the image edge.
_TEXTURE_ACCENT_RGB = (220, 205, 180)

#: Terminal background color the texture fades toward.
_TEXTURE_BG_RGB = (8, 10, 14)

#: Braille base codepoint — U+2800 is the empty braille pattern.
_BRAILLE_BASE = 0x2800

#: Braille dot indices for left column (bits 0,1,2,6) and right
#: column (bits 3,4,5,7).
_BRAILLE_LEFT_COL = (0, 1, 2, 6)
_BRAILLE_RIGHT_COL = (3, 4, 5, 7)


# ── Functions ───────────────────────────────────────────────────────


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
    cell_width = int(round(cell_height * crop_aspect * terminal_cell_aspect))
    if cell_width > max_cell_width:
        shrink = max_cell_width / cell_width
        cell_width = max_cell_width
        cell_height = max(1, int(round(cell_height * shrink)))
    cell_width = max(1, cell_width)
    return (cell_width, cell_height)


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
    ``c`` cells AND down by ``r`` rows." With ``C=1``, the place
    command paints the image but leaves the cursor where it was before.

    No payload — this is a control-only sequence.
    """
    return (
        f"\x1b_Ga=p,i={image_id},c={cell_width},r={cell_height},C=1\x1b\\"
    )


def _texture_cell(
    *,
    distance_from_image: int,
    max_distance: int,
    seed_key: tuple,
) -> tuple[str, tuple[int, int, int]]:
    """Pick a (glyph, rgb) pair for one texture cell.

    Two zones:

    1. **Solid columns** (d < _SOLID_COLUMNS): deterministic solid
       blocks — █ at d=0,1; ▓ at d=2. No randomness, no braille.
    2. **Braille field** (d >= _SOLID_COLUMNS): braille dots with
       density and color that fall off gently over the full distance
       to the terminal edge.

    ``max_distance`` is the distance from image edge to the terminal
    edge on this side.
    """
    d = max(0, distance_from_image)
    span = max(1, max_distance)

    # Normalized position [0, 1] from image edge to terminal edge.
    t = min(1.0, d / span)

    # Gentle power curve: (1 - t)^1.8 holds value through the
    # middle of the range better than smoothstep.
    falloff = (1.0 - t) ** 1.8
    intensity = _TEXTURE_EDGE_FLOOR + (1.0 - _TEXTURE_EDGE_FLOOR) * falloff

    rgb = _lerp_rgb(_TEXTURE_BG_RGB, _TEXTURE_ACCENT_RGB, intensity)

    # Zone 1: solid blocks — deterministic, no randomness needed.
    if d < _SOLID_COLUMNS:
        glyph = "▓" if d == _SOLID_COLUMNS - 1 else "█"
        return glyph, rgb

    # Zone 2: braille — density tracks the same power curve.
    density = intensity

    # Deterministic randomness from the seed key.
    rand = random.Random(hash(seed_key))

    # At the floor, most cells are empty with rare single dots.
    if density <= _TEXTURE_EDGE_FLOOR + 0.01:
        if rand.random() < _TEXTURE_EDGE_FLOOR:
            glyph = chr(_BRAILLE_BASE + (1 << rand.randint(0, 7)))
            return glyph, rgb
        return " ", _TEXTURE_BG_RGB

    # Directional bias: near the image, prefer vertical column
    # patterns. Bias fades linearly over the first 20% of the span.
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


def _emit_band_border_row(term_width: int, *, title: str):
    """Yield one full-width `─` border row, optionally with an
    inlined left-aligned title.
    """
    border_style = Style.parse(_rgb_to_hex((135, 160, 145)))
    if title:
        prefix_text = f"─ {title} "
        remaining = term_width - len(prefix_text)
        if remaining < 0:
            prefix_text = prefix_text[:term_width]
            remaining = 0
        row = prefix_text + ("─" * remaining)
    else:
        row = "─" * term_width
    yield Segment(row, border_style)
    yield Segment.line()


def _emit_band_texture_span(
    *,
    col_start: int,
    col_end: int,
    image_left: int,
    image_right: int,
    term_width: int,
    row_seed_id: int,
    image_id: int,
):
    """Yield one Segment per cell in [col_start, col_end), each picked
    by :func:`_texture_cell` based on distance from the nearest image
    edge.
    """
    for col in range(col_start, col_end):
        if col < image_left:
            distance = image_left - col
            max_dist = image_left  # left edge to image
        elif col >= image_right:
            distance = col - image_right + 1
            max_dist = max(1, term_width - image_right)  # image to right edge
        else:
            distance = 0
            max_dist = 1
        glyph, rgb = _texture_cell(
            distance_from_image=distance,
            max_distance=max_dist,
            seed_key=(image_id, row_seed_id, col),
        )
        yield Segment(glyph, Style.parse(_rgb_to_hex(rgb)))


def _emit_band_texture_only_row(
    term_width: int,
    *,
    image_left: int,
    image_right: int,
    row_seed_id: int,
    image_id: int,
):
    """Yield a full-width row of texture cells, with no image
    placement.
    """
    yield from _emit_band_texture_span(
        col_start=0,
        col_end=term_width,
        image_left=image_left,
        image_right=image_right,
        term_width=term_width,
        row_seed_id=row_seed_id,
        image_id=image_id,
    )
    yield Segment.line()


def _query_terminal_cell_aspect(
    *,
    timeout_s: float = 0.15,
    stream=None,
) -> float | None:
    """Query the terminal for its cell pixel dimensions via CSI 16t.

    Returns the cell aspect ratio (height / width, in screen pixels)
    if the terminal responds with a well-formed answer. Returns
    ``None`` on any failure.
    """
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
    """
    if not term_program:
        return False
    return term_program in {"WezTerm", "kitty"}


def _supports_inline_images(term_program: str | None) -> bool:
    """Return True if the terminal identified by ``$TERM_PROGRAM``
    supports the iTerm2 inline image protocol.
    """
    if not term_program:
        return False
    return term_program in {"WezTerm", "iTerm.app"}


def _otsu_threshold(luminances) -> float:
    """Classic Otsu's method: pick the luminance cut that maximizes
    between-class variance across a 256-bin histogram.

    Returns a float threshold in [0, 255]. Degenerate inputs (empty,
    uniform) return a safe midpoint rather than raising.
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
    return best_threshold + 1.0


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


def _render_focus_preview_pixels(
    pixels: list[list[tuple[int, int, int]]],
    *,
    now: float | None = None,
    pending: bool = False,
) -> "Group":
    """Render a sampled pixel grid as a Rich Group.

    ``pixels`` is expected to be sampled at 2× vertical density relative
    to the terminal row budget — each pair of source rows (2y, 2y+1)
    becomes one terminal row. The steady-state renderer uses that pair
    as the top and bottom halves of a half-block (▀). The pending
    (transition) renderer averages each pair back down to a single
    per-cell RGB and then runs its existing glyph-overlay animation.
    """
    from rich.console import Group

    now = time.monotonic() if now is None else now
    if pending:
        return _render_focus_preview_pending(pixels, now=now)
    return _render_focus_preview_steady(pixels)


def _render_focus_preview_steady(
    pixels: list[list[tuple[int, int, int]]],
) -> "Group":
    """Legibility-first steady-state renderer.

    Binary Otsu threshold across all sampled luminances, half-block
    cells (one terminal cell = one top half-pixel + one bottom
    half-pixel), hard ink/paper palette.
    """
    from rich.console import Group

    if not pixels or not pixels[0]:
        return Group()

    source_height = len(pixels)
    source_width = len(pixels[0])

    luminances = [_pixel_luma(rgb) for row in pixels for rgb in row]
    threshold = _otsu_threshold(luminances)

    lum_span = max(luminances) - min(luminances) if luminances else 0.0
    degenerate = lum_span < 12.0

    ink_hex = _rgb_to_hex(_FOCUS_PREVIEW_HARD_INK_RGB)
    paper_hex = _rgb_to_hex(_FOCUS_PREVIEW_HARD_PAPER_RGB)

    def _color_for(rgb: tuple[int, int, int]) -> str:
        if degenerate:
            return paper_hex
        return ink_hex if _pixel_luma(rgb) < threshold else paper_hex

    rows: list[Text] = []
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
            text.append("\u2580", style=f"{top_color} on {bottom_color}")
        rows.append(text)
        y += 2
    return Group(*rows)


def _render_focus_preview_pending(
    pixels: list[list[tuple[int, int, int]]],
    *,
    now: float,
) -> "Group":
    """Transition-layer renderer (unchanged animation, now fed from a
    2×-vertical pixel grid by averaging row pairs back down to 1×)."""
    from rich.console import Group

    if not pixels or not pixels[0]:
        return Group()

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
            bg_rgb = _interp_rgb(_FOCUS_PREVIEW_BG_RGB, toned_rgb, 0.34)
            row.append(
                " ",
                style=f"{_rgb_to_hex(bg_rgb)} on {_rgb_to_hex(bg_rgb)}",
            )
        rows.append(row)
    return Group(*rows)


# ── Classes ─────────────────────────────────────────────────────────


class FocusPreviewInlineImage:
    """Rich Renderable that emits an iTerm2 inline image framed in a
    self-drawn border box.

    This renderable is deliberately NOT wrapped in a ``rich.Panel``
    because Panel's padding logic writes literal spaces to the cells
    to the right of its inner content on every row, filling the panel
    to its declared inner width. Those spaces land on exactly the
    terminal cells that the iTerm2 escape sequence just painted image
    pixels into, overwriting the image the instant it's drawn.

    The fix is to own the border drawing ourselves and use
    cursor-forward escape sequences (``ESC[nC``) marked as zero-width
    control segments to advance the cursor across the image region
    without writing any visible characters to those cells.
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

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        border = Style.parse("#3d4458")
        inner_width = self._cell_width
        total_width = inner_width + 2

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

        yield Segment("│", border)
        yield Segment(self._sequence, None, [(ControlType.BELL,)])
        yield Segment(forward_escape, None, [(ControlType.BELL,)])
        yield Segment("│", border)
        yield Segment.line()

        for _ in range(self._cell_height - 1):
            yield Segment("│", border)
            yield Segment(forward_escape, None, [(ControlType.BELL,)])
            yield Segment("│", border)
            yield Segment.line()

        bottom_border = "╰" + ("─" * (total_width - 2)) + "╯"
        yield Segment(bottom_border, border)
        yield Segment.line()


class FocusPreviewKittyImage:
    """Rich Renderable that places a previously-transmitted Kitty
    graphics image inside a self-drawn border frame.

    Unlike :class:`FocusPreviewInlineImage`, this class does NOT
    carry the PNG data. The transmit (`_build_kitty_transmit_chunks`)
    must be done out-of-band by the caller.

    Cell-box sizing is done at RENDER TIME, not at construction.
    """

    def __init__(
        self,
        *,
        image_id: int,
        texture_seed: int,
        image_pixel_width: int,
        image_pixel_height: int,
        terminal_cell_aspect: float,
        title: str = "",
    ) -> None:
        self._image_id = image_id
        self._texture_seed = texture_seed
        self._image_pixel_width = image_pixel_width
        self._image_pixel_height = image_pixel_height
        self._terminal_cell_aspect = terminal_cell_aspect
        self._title = title

    def _compute_box(self, available_width: int) -> tuple[int, int]:
        """Compute (cell_width, cell_height) for the image."""
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
        """Render the focus preview as a full-terminal-width band."""
        term_width = max(1, options.max_width)
        cell_width, cell_height = self._compute_box(term_width)

        image_left = max(0, (term_width - cell_width) // 2)
        image_right = image_left + cell_width

        place_sequence = _build_kitty_place_sequence(
            self._image_id,
            cell_width=cell_width,
            cell_height=cell_height,
        )

        # Top border rule with inlined title
        yield from _emit_band_border_row(term_width, title=self._title)

        # Extra texture row above the image
        yield from _emit_band_texture_only_row(
            term_width,
            image_left=image_left,
            image_right=image_right,
            row_seed_id=0,
            image_id=self._texture_seed,
        )

        # Image rows
        forward_escape = f"\x1b[{cell_width}C"
        for image_row in range(cell_height):
            yield from _emit_band_texture_span(
                col_start=0,
                col_end=image_left,
                image_left=image_left,
                image_right=image_right,
                term_width=term_width,
                row_seed_id=1 + image_row,
                image_id=self._texture_seed,
            )
            if image_row == 0:
                yield Segment(
                    place_sequence, None, [(ControlType.BELL,)]
                )
            yield Segment(forward_escape, None, [(ControlType.BELL,)])
            yield from _emit_band_texture_span(
                col_start=image_right,
                col_end=term_width,
                image_left=image_left,
                image_right=image_right,
                term_width=term_width,
                row_seed_id=1 + image_row,
                image_id=self._texture_seed,
            )
            yield Segment.line()

        # Extra texture row below the image
        yield from _emit_band_texture_only_row(
            term_width,
            image_left=image_left,
            image_right=image_right,
            row_seed_id=1 + cell_height,
            image_id=self._texture_seed,
        )

        # Bottom border rule
        yield from _emit_band_border_row(term_width, title="")


class FocusPreviewLoadingBand:
    """Placeholder renderable shown before any focus_preview event
    fires, or between items while a new preview is loading.
    """

    def __init__(self, *, title: str = "focus preview") -> None:
        self._title = title

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        term_width = max(1, options.max_width)

        inner_budget = max(1, term_width - 2)
        cell_width = min(_INLINE_IMAGE_MAX_CELL_WIDTH, inner_budget)
        cell_height = _INLINE_IMAGE_CELL_HEIGHT
        image_left = max(0, (term_width - cell_width) // 2)
        image_right = image_left + cell_width

        placeholder_text = "(preview loading…)"
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

        # Bottom border.
        yield from _emit_band_border_row(term_width, title="")
