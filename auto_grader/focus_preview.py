from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


#: Sepia tone-map target color. Source pixels are interpolated
#: toward this warm parchment by _SEPIA_MIX, which softens the
#: harsh paper-white contrast against the dark terminal background.
_SEPIA_TARGET_RGB = (215, 180, 130)

#: How strongly the sepia tone-map pulls source pixels toward the
#: warm target. 0.0 = no tint (raw scan), 1.0 = fully tinted.
#: ~0.30 is visible on white paper without distorting ink contrast.
_SEPIA_MIX = 0.30


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    Applies a sepia warm-tone to every source pixel. No vignette,
    no corner rounding, no outline ring — edge treatment is handled
    entirely by the textured-band renderer in narrator_reader.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    crop_rgb, crop_width, crop_height = _trim_edge_matte(
        crop_rgb,
        crop_width,
        crop_height,
    )
    crop_rgb, crop_width, crop_height = _tighten_to_content_bounds(
        crop_rgb,
        crop_width,
        crop_height,
    )
    bg_rgb = _average_corner_rgb(crop_rgb, crop_width, crop_height)
    max_bg_diff = _max_background_diff(crop_rgb, crop_width, crop_height, bg_rgb)
    use_transparent_background = max_bg_diff >= 72
    out = bytearray(crop_width * crop_height * 4)

    for y in range(crop_height):
        for x in range(crop_width):
            offset = (y * crop_width + x) * 3
            out_offset = (y * crop_width + x) * 4
            src_rgb = tuple(crop_rgb[offset : offset + 3])
            for channel in range(3):
                src_value = src_rgb[channel]
                value = (
                    src_value * (1.0 - _SEPIA_MIX)
                    + _SEPIA_TARGET_RGB[channel] * _SEPIA_MIX
                )
                out[out_offset + channel] = int(round(value))
            if use_transparent_background:
                diff = max(
                    abs(src_rgb[0] - bg_rgb[0]),
                    abs(src_rgb[1] - bg_rgb[1]),
                    abs(src_rgb[2] - bg_rgb[2]),
                )
                if diff < 48:
                    alpha = 0
                else:
                    alpha = 255
            else:
                alpha = 255
            out[out_offset + 3] = alpha

    pix = fitz.Pixmap(fitz.csRGB, crop_width, crop_height, bytes(out), True)
    return pix.tobytes("png")


def _crop_focus_region(
    page_png: bytes,
    focus_region: FocusRegion,
) -> tuple[bytes, int, int]:
    pix = _png_to_pixmap(page_png)
    page_width = pix.width
    page_height = pix.height

    x0 = max(0, min(page_width - 1, int(math.floor(focus_region.x * page_width))))
    y0 = max(0, min(page_height - 1, int(math.floor(focus_region.y * page_height))))
    x1 = max(
        x0 + 1,
        min(page_width, int(math.ceil((focus_region.x + focus_region.width) * page_width))),
    )
    y1 = max(
        y0 + 1,
        min(page_height, int(math.ceil((focus_region.y + focus_region.height) * page_height))),
    )

    crop_width = x1 - x0
    crop_height = y1 - y0
    crop = bytearray(crop_width * crop_height * 3)
    samples = pix.samples

    for y in range(crop_height):
        for x in range(crop_width):
            src_offset = ((y0 + y) * page_width + (x0 + x)) * pix.n
            dest_offset = (y * crop_width + x) * 3
            crop[dest_offset : dest_offset + 3] = samples[src_offset : src_offset + 3]

    return bytes(crop), crop_width, crop_height


def _trim_edge_matte(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
) -> tuple[bytes, int, int]:
    crop_rgb, crop_width, crop_height = _trim_near_black_matte(
        crop_rgb,
        crop_width,
        crop_height,
    )
    return _trim_uniform_paper_matte(crop_rgb, crop_width, crop_height)


def _tighten_to_content_bounds(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
    *,
    content_threshold: int = 48,
    pad_px: int = 4,
) -> tuple[bytes, int, int]:
    """Tighten a crop to the meaningful content inside it.

    Edge-matte trimming handles obvious scan borders. This second pass handles
    the harder case: a crop that is mostly paper margin with a smaller content
    island inside it. We detect pixels that differ materially from the corner
    paper tone, compute a bounding box, and keep only that box plus a tiny
    stable pad so the preview doesn't feel claustrophobic.
    """

    def _rgb(x: int, y: int) -> tuple[int, int, int]:
        off = (y * crop_width + x) * 3
        return tuple(crop_rgb[off : off + 3])

    bg = tuple(
        round(
            (
                a + b + c + d
            )
            / 4
        )
        for a, b, c, d in zip(
            _rgb(0, 0),
            _rgb(crop_width - 1, 0),
            _rgb(0, crop_height - 1),
            _rgb(crop_width - 1, crop_height - 1),
        )
    )

    left = crop_width
    right = -1
    top = crop_height
    bottom = -1

    for y in range(crop_height):
        row_base = y * crop_width * 3
        for x in range(crop_width):
            off = row_base + x * 3
            r = crop_rgb[off]
            g = crop_rgb[off + 1]
            b = crop_rgb[off + 2]
            if max(abs(r - bg[0]), abs(g - bg[1]), abs(b - bg[2])) >= content_threshold:
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

    if right < left or bottom < top:
        return crop_rgb, crop_width, crop_height

    left = max(0, left - pad_px)
    top = max(0, top - pad_px)
    right = min(crop_width - 1, right + pad_px)
    bottom = min(crop_height - 1, bottom + pad_px)

    return _slice_crop_rgb(crop_rgb, crop_width, crop_height, left, top, right, bottom)


def _average_corner_rgb(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int]:
    def _rgb(x: int, y: int) -> tuple[int, int, int]:
        off = (y * crop_width + x) * 3
        return tuple(crop_rgb[off : off + 3])

    return tuple(
        round((a + b + c + d) / 4)
        for a, b, c, d in zip(
            _rgb(0, 0),
            _rgb(crop_width - 1, 0),
            _rgb(0, crop_height - 1),
            _rgb(crop_width - 1, crop_height - 1),
        )
    )


def _max_background_diff(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
    bg_rgb: tuple[int, int, int],
) -> int:
    max_diff = 0
    for y in range(crop_height):
        row_base = y * crop_width * 3
        for x in range(crop_width):
            off = row_base + x * 3
            diff = max(
                abs(crop_rgb[off] - bg_rgb[0]),
                abs(crop_rgb[off + 1] - bg_rgb[1]),
                abs(crop_rgb[off + 2] - bg_rgb[2]),
            )
            if diff > max_diff:
                max_diff = diff
    return max_diff


def _trim_near_black_matte(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
    *,
    threshold: int = 20,
) -> tuple[bytes, int, int]:
    def _is_near_black(x: int, y: int) -> bool:
        off = (y * crop_width + x) * 3
        r, g, b = crop_rgb[off : off + 3]
        return r <= threshold and g <= threshold and b <= threshold

    def _row_is_near_black(y: int) -> bool:
        return all(_is_near_black(x, y) for x in range(crop_width))

    def _col_is_near_black(x: int) -> bool:
        return all(_is_near_black(x, y) for y in range(crop_height))

    top = 0
    while top < crop_height - 1 and _row_is_near_black(top):
        top += 1

    bottom = crop_height - 1
    while bottom > top and _row_is_near_black(bottom):
        bottom -= 1

    left = 0
    while left < crop_width - 1 and _col_is_near_black(left):
        left += 1

    right = crop_width - 1
    while right > left and _col_is_near_black(right):
        right -= 1

    return _slice_crop_rgb(crop_rgb, crop_width, crop_height, left, top, right, bottom)


def _trim_uniform_paper_matte(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
    *,
    color_tolerance: int = 28,
    coverage: float = 0.92,
) -> tuple[bytes, int, int]:
    def _rgb(x: int, y: int) -> tuple[int, int, int]:
        off = (y * crop_width + x) * 3
        return tuple(crop_rgb[off : off + 3])

    def _close(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
        return (
            abs(a[0] - b[0]) <= color_tolerance
            and abs(a[1] - b[1]) <= color_tolerance
            and abs(a[2] - b[2]) <= color_tolerance
        )

    top_left = _rgb(0, 0)
    top_right = _rgb(crop_width - 1, 0)
    bottom_left = _rgb(0, crop_height - 1)
    bottom_right = _rgb(crop_width - 1, crop_height - 1)
    center = _rgb(crop_width // 2, crop_height // 2)

    if (
        _close(top_left, center)
        and _close(top_right, center)
        and _close(bottom_left, center)
        and _close(bottom_right, center)
    ):
        return crop_rgb, crop_width, crop_height

    def _row_matches(y: int, sample: tuple[int, int, int]) -> bool:
        hits = sum(1 for x in range(crop_width) if _close(_rgb(x, y), sample))
        return hits / max(1, crop_width) >= coverage

    def _col_matches(x: int, sample: tuple[int, int, int]) -> bool:
        hits = sum(1 for y in range(crop_height) if _close(_rgb(x, y), sample))
        return hits / max(1, crop_height) >= coverage

    top = 0
    while top < crop_height - 1 and _row_matches(top, top_left):
        top += 1

    bottom = crop_height - 1
    while bottom > top and _row_matches(bottom, bottom_left):
        bottom -= 1

    left = 0
    while left < crop_width - 1 and _col_matches(left, top_left):
        left += 1

    right = crop_width - 1
    while right > left and _col_matches(right, top_right):
        right -= 1

    return _slice_crop_rgb(crop_rgb, crop_width, crop_height, left, top, right, bottom)


def _slice_crop_rgb(
    crop_rgb: bytes,
    crop_width: int,
    crop_height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[bytes, int, int]:
    if top == 0 and left == 0 and right == crop_width - 1 and bottom == crop_height - 1:
        return crop_rgb, crop_width, crop_height

    new_w = right - left + 1
    new_h = bottom - top + 1
    trimmed = bytearray(new_w * new_h * 3)
    for row in range(new_h):
        src_start = ((top + row) * crop_width + left) * 3
        src_end = src_start + new_w * 3
        dst_start = row * new_w * 3
        trimmed[dst_start : dst_start + new_w * 3] = crop_rgb[src_start:src_end]
    return bytes(trimmed), new_w, new_h


def _png_to_pixmap(png_bytes: bytes) -> fitz.Pixmap:
    return fitz.Pixmap(png_bytes)
