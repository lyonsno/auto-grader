from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


_BACKGROUND_RGB = (8, 10, 14)
#: Subtle cool accent for the soft outline band. Sits one outline
#: width inside the image edge, fades with the same vignette alpha
#: as the content so the outline itself blends into the background
#: at the corners.
_OUTLINE_RGB = (72, 96, 144)

#: Padding around the crop in screen pixels before the feathered edge
#: blends into the background. Needs to be large enough to hold the
#: full vignette width plus a little extra, otherwise the feather
#: gets clipped at the output boundary. At 800 DPI with a vignette
#: fraction of 0.25, this works out to roughly 500 px on a 2000-px
#: crop — so we need padding of at least ~250 px to contain it.
_PADDING_PX = 300

#: Vignette edge width as a FRACTION of the smaller image dimension.
#: Deep feather — roughly 25% of the smaller dim blends into the
#: panel background. On a 2000-px-tall crop that's ~500 px of fade
#: on each edge, which reads as a generous smoke edge that softens
#: the transition into the surrounding terminal.
_VIGNETTE_FRACTION = 0.25

#: Corner radius as a fraction of the smaller image dimension. Larger
#: than the vignette is narrow because we want the rounded shape to
#: be clearly visible inside the fade — the vignette rounds the edge
#: in one way (luminance) and the corner rounds it in another
#: (geometry), and both have to read at the same visual scale.
_CORNER_RADIUS_FRACTION = 0.05

#: Corner soften width — the smoothstep over which the signed-
#: distance field transitions from "inside the rounded rect" to
#: "outside." Bigger soften = smoother antialiasing at the cost of
#: corner sharpness. At the 0.004 we had before, the transition was
#: ~8 px on a 2000-px crop which reads as jagged. At 0.025 it's
#: ~50 px which reads as a clean curve.
_CORNER_SOFTEN_FRACTION = 0.025

#: Outline band thickness as a fraction of the smaller image
#: dimension. The soft outline sits as a thin ring of accent color
#: just inside the rounded-corner content edge, at the location
#: where the content has mostly faded out. Visible enough to frame
#: the image, but integrated with the vignette so it doesn't look
#: like a hard rectangular stroke.
_OUTLINE_BAND_FRACTION = 0.008

#: How far inside the content edge the outline band sits. Negative
#: values push the outline into the visible part of the image;
#: positive values push it into the faded edge. We want it right
#: at the transition where the eye reads the image boundary.
_OUTLINE_INSET_FRACTION = 0.01


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    The crop is placed inside a padded canvas, feathered toward the
    terminal-matched background near the edges, rounded at the
    corners, and gets a soft accent outline ring just inside the
    content edge. All edge-effect sizes are proportional to the
    smaller image dimension so the visual result is consistent
    across source DPIs.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    out_width = crop_width + (2 * _PADDING_PX)
    out_height = crop_height + (2 * _PADDING_PX)
    out = bytearray(_BACKGROUND_RGB * (out_width * out_height))

    # Compute edge-effect sizes from the smaller crop dimension so
    # the visual result scales with the source image.
    smaller_dim = min(crop_width, crop_height)
    vignette_px = max(4.0, smaller_dim * _VIGNETTE_FRACTION)
    corner_radius_px = max(2.0, smaller_dim * _CORNER_RADIUS_FRACTION)
    corner_soften_px = max(0.5, smaller_dim * _CORNER_SOFTEN_FRACTION)
    outline_band_px = max(1.0, smaller_dim * _OUTLINE_BAND_FRACTION)
    outline_inset_px = smaller_dim * _OUTLINE_INSET_FRACTION

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
            # Outline band: a thin ring of accent color placed at the
            # rounded-content edge, inset by outline_inset_px.
            # Mathematically: we want a narrow band of alpha where
            # the distance from the rounded edge is within
            # [inset - band/2, inset + band/2]. Approximated via a
            # hat function over the edge distance from the same
            # rounded-rect SDF used for the corner geometry.
            outline_mix = _outline_band_alpha(
                x,
                y,
                crop_width,
                crop_height,
                corner_radius_px=corner_radius_px,
                band_width_px=outline_band_px,
                inset_px=outline_inset_px,
            )
            src_offset = (y * crop_width + x) * 3
            dest_x = x + _PADDING_PX
            dest_y = y + _PADDING_PX
            dest_offset = (dest_y * out_width + dest_x) * 3

            for channel in range(3):
                src_value = crop_rgb[src_offset + channel]
                # Blend the source with outline accent where the
                # outline band says we should.
                tinted = (
                    src_value * (1.0 - outline_mix)
                    + _OUTLINE_RGB[channel] * outline_mix
                )
                bg_value = _BACKGROUND_RGB[channel]
                value = (bg_value * (1.0 - alpha)) + (tinted * alpha)
                out[dest_offset + channel] = int(round(value))

    pix = fitz.Pixmap(fitz.csRGB, out_width, out_height, bytes(out), False)
    return pix.tobytes("png")


def _outline_band_alpha(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    corner_radius_px: float,
    band_width_px: float,
    inset_px: float,
) -> float:
    """Return the outline band intensity at (x, y).

    The outline sits at a specific inset inside the rounded-rect
    content edge, with a narrow band_width_px falloff on each side.
    Uses the same signed-distance-field rounded-rect math as
    :func:`_rounded_corner_alpha` so the outline automatically
    follows the corner curvature.
    """
    px = x + 0.5
    py = y + 0.5
    half_w = width / 2.0
    half_h = height / 2.0
    qx = abs(px - half_w) - (half_w - corner_radius_px)
    qy = abs(py - half_h) - (half_h - corner_radius_px)
    # Signed distance from the rounded-rect edge: negative inside,
    # positive outside.
    sdf = (
        math.hypot(max(qx, 0.0), max(qy, 0.0))
        + min(max(qx, qy), 0.0)
        - corner_radius_px
    )
    # We want the band centered at sdf == -inset_px (inside the
    # edge by that much). A hat function of width band_width_px
    # peaks at that distance and fades to zero at the edges.
    distance_from_band_center = abs(sdf - (-inset_px))
    half_band = band_width_px / 2.0
    if distance_from_band_center >= half_band:
        return 0.0
    # Smoothstep-ish falloff: peak 1.0 at center, 0.0 at edges.
    t = 1.0 - (distance_from_band_center / half_band)
    return t * t * (3.0 - 2.0 * t)


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
