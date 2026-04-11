from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


_BACKGROUND_RGB = (8, 10, 14)

#: Padding around the crop in screen pixels before the feathered edge
#: blends into the background. At 800 DPI source, 40 px reads as a
#: comfortable border and gives the fade room to land.
_PADDING_PX = 40

#: Edge effects scale with the smaller image dimension so the visual
#: result stays consistent across source DPIs.
_VIGNETTE_FRACTION = 0.05
_CORNER_RADIUS_FRACTION = 0.025
_CORNER_SOFTEN_FRACTION = 0.004


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    The crop is feathered toward the terminal-matched background near
    the edges with softened rounded corners. Edge-effect sizes scale
    with the smaller image dimension so the result reads similarly
    across source DPIs.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    out_width = crop_width + (2 * _PADDING_PX)
    out_height = crop_height + (2 * _PADDING_PX)
    out = bytearray(_BACKGROUND_RGB * (out_width * out_height))

    smaller_dim = min(crop_width, crop_height)
    vignette_px = max(4.0, smaller_dim * _VIGNETTE_FRACTION)
    corner_radius_px = max(2.0, smaller_dim * _CORNER_RADIUS_FRACTION)
    corner_soften_px = max(0.5, smaller_dim * _CORNER_SOFTEN_FRACTION)

    for y in range(crop_height):
        for x in range(crop_width):
            alpha = _combined_alpha(
                x,
                y,
                crop_width,
                crop_height,
                vignette_px=vignette_px,
                corner_radius_px=corner_radius_px,
                corner_soften_px=corner_soften_px,
            )
            src_offset = (y * crop_width + x) * 3
            dest_x = x + _PADDING_PX
            dest_y = y + _PADDING_PX
            dest_offset = (dest_y * out_width + dest_x) * 3

            for channel in range(3):
                src_value = crop_rgb[src_offset + channel]
                bg_value = _BACKGROUND_RGB[channel]
                value = (bg_value * (1.0 - alpha)) + (src_value * alpha)
                out[dest_offset + channel] = int(round(value))

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


def _combined_alpha(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    vignette_px: float,
    corner_radius_px: float,
    corner_soften_px: float,
) -> float:
    edge_alpha = _edge_vignette_alpha(x, y, width, height, vignette_px=vignette_px)
    corner_alpha = _rounded_corner_alpha(
        x,
        y,
        width,
        height,
        corner_radius_px=corner_radius_px,
        corner_soften_px=corner_soften_px,
    )
    return max(0.0, min(1.0, edge_alpha * corner_alpha))


def _edge_vignette_alpha(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    vignette_px: float,
) -> float:
    distance = min(
        x + 0.5,
        y + 0.5,
        width - (x + 0.5),
        height - (y + 0.5),
    )
    return _smoothstep(0.0, vignette_px, distance)


def _rounded_corner_alpha(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    corner_radius_px: float,
    corner_soften_px: float,
) -> float:
    px = x + 0.5
    py = y + 0.5
    half_w = width / 2.0
    half_h = height / 2.0
    qx = abs(px - half_w) - (half_w - corner_radius_px)
    qy = abs(py - half_h) - (half_h - corner_radius_px)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - corner_radius_px
    return 1.0 - _smoothstep(-corner_soften_px, corner_soften_px, outside)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = (value - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
