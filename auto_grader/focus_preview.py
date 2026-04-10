from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


_BACKGROUND_RGB = (8, 10, 14)
_OUTLINE_RGB = (64, 84, 120)
_PADDING_PX = 8
_VIGNETTE_PX = 10.0
_CORNER_RADIUS_PX = 4.0
_CORNER_SOFTEN_PX = 1.5


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    The crop is placed inside a thin outline box, then feathered toward the
    terminal-matched background near the edges with only slightly softened
    corners.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    out_width = crop_width + (2 * _PADDING_PX)
    out_height = crop_height + (2 * _PADDING_PX)
    out = bytearray(_BACKGROUND_RGB * (out_width * out_height))

    for y in range(crop_height):
        for x in range(crop_width):
            alpha = _combined_alpha(x, y, crop_width, crop_height)
            src_offset = (y * crop_width + x) * 3
            dest_x = x + _PADDING_PX
            dest_y = y + _PADDING_PX
            dest_offset = (dest_y * out_width + dest_x) * 3

            for channel in range(3):
                src_value = crop_rgb[src_offset + channel]
                bg_value = _BACKGROUND_RGB[channel]
                value = (bg_value * (1.0 - alpha)) + (src_value * alpha)
                out[dest_offset + channel] = int(round(value))

    _draw_outline(out, out_width, out_height)
    pix = fitz.Pixmap(fitz.csRGB, out_width, out_height, bytes(out), False)
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
    x1 = max(x0 + 1, min(page_width, int(math.ceil((focus_region.x + focus_region.width) * page_width))))
    y1 = max(y0 + 1, min(page_height, int(math.ceil((focus_region.y + focus_region.height) * page_height))))

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


def _png_to_pixmap(png_bytes: bytes) -> fitz.Pixmap:
    return fitz.Pixmap(png_bytes)


def _combined_alpha(x: int, y: int, width: int, height: int) -> float:
    edge_alpha = _edge_vignette_alpha(x, y, width, height)
    corner_alpha = _rounded_corner_alpha(x, y, width, height)
    return max(0.0, min(1.0, edge_alpha * corner_alpha))


def _edge_vignette_alpha(x: int, y: int, width: int, height: int) -> float:
    distance = min(
        x + 0.5,
        y + 0.5,
        width - (x + 0.5),
        height - (y + 0.5),
    )
    return _smoothstep(0.0, _VIGNETTE_PX, distance)


def _rounded_corner_alpha(x: int, y: int, width: int, height: int) -> float:
    px = x + 0.5
    py = y + 0.5
    half_w = width / 2.0
    half_h = height / 2.0
    qx = abs(px - half_w) - (half_w - _CORNER_RADIUS_PX)
    qy = abs(py - half_h) - (half_h - _CORNER_RADIUS_PX)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - _CORNER_RADIUS_PX
    return 1.0 - _smoothstep(-_CORNER_SOFTEN_PX, _CORNER_SOFTEN_PX, outside)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = (value - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _draw_outline(buffer: bytearray, width: int, height: int) -> None:
    for x in range(width):
        _set_pixel(buffer, width, x, 0, _OUTLINE_RGB)
        _set_pixel(buffer, width, x, height - 1, _OUTLINE_RGB)
    for y in range(height):
        _set_pixel(buffer, width, 0, y, _OUTLINE_RGB)
        _set_pixel(buffer, width, width - 1, y, _OUTLINE_RGB)


def _set_pixel(
    buffer: bytearray,
    width: int,
    x: int,
    y: int,
    rgb: tuple[int, int, int],
) -> None:
    offset = (y * width + x) * 3
    buffer[offset : offset + 3] = bytes(rgb)
