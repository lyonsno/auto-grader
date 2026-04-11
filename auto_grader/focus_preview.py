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

#: No padding around the crop. The vignette fades from content to
#: background *within* the crop rectangle (color interpolation in
#: pixel space), so there's no need for a "safe landing zone"
#: outside the content. Any padding would inflate the canvas aspect
#: ratio away from the content aspect ratio, which causes WezTerm's
#: aspect-preserving place command to letterbox the image inside
#: the cell box we request.
_PADDING_PX = 0

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
    outline_band_px = max(1.0, smaller_dim * _OUTLINE_BAND_FRACTION)
    outline_inset_px = smaller_dim * _OUTLINE_INSET_FRACTION

    # Rounded-rect signed distance field half-dimensions. The SDF
    # is computed in the crop's own coordinate space, centered on
    # the crop center. The content region is a rounded rectangle
    # `crop_width × crop_height` with `corner_radius_px` radius.
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0
    inner_half_w = half_w - corner_radius_px
    inner_half_h = half_h - corner_radius_px

    for y in range(crop_height):
        for x in range(crop_width):
            # Signed distance from the rounded-rect edge.
            # Negative inside, positive outside, value is in
            # crop-space pixels. This is the ONLY shape function
            # for the vignette — no multiply against a separate
            # 1D edge mask, so there are no seams or axis-aligned
            # stripes in the corners.
            px = x + 0.5 - half_w
            py = y + 0.5 - half_h
            qx = abs(px) - inner_half_w
            qy = abs(py) - inner_half_h
            sdf = (
                math.hypot(max(qx, 0.0), max(qy, 0.0))
                + min(max(qx, qy), 0.0)
                - corner_radius_px
            )
            # Fade from fully-opaque image (deep inside, sdf <=
            # -vignette_px) to fully-background (at the edge,
            # sdf >= 0). Smoothstep on the normalized depth.
            if sdf >= 0.0:
                alpha = 0.0
            elif sdf <= -vignette_px:
                alpha = 1.0
            else:
                t = -sdf / vignette_px
                alpha = t * t * (3.0 - 2.0 * t)

            # Outline accent band: a narrow ring of accent color
            # centered at sdf == -outline_inset_px, width
            # outline_band_px on each side. Blended into the
            # source color before the vignette alpha is applied,
            # so the outline also fades with the vignette and
            # doesn't leave a crisp rectangle inside the feather.
            distance_from_band_center = abs(sdf - (-outline_inset_px))
            half_band = outline_band_px / 2.0
            if distance_from_band_center >= half_band:
                outline_mix = 0.0
            else:
                t = 1.0 - (distance_from_band_center / half_band)
                outline_mix = t * t * (3.0 - 2.0 * t)

            src_offset = (y * crop_width + x) * 3
            dest_x = x + _PADDING_PX
            dest_y = y + _PADDING_PX
            dest_offset = (dest_y * out_width + dest_x) * 3

            for channel in range(3):
                src_value = crop_rgb[src_offset + channel]
                tinted = (
                    src_value * (1.0 - outline_mix)
                    + _OUTLINE_RGB[channel] * outline_mix
                )
                bg_value = _BACKGROUND_RGB[channel]
                value = (bg_value * (1.0 - alpha)) + (tinted * alpha)
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


