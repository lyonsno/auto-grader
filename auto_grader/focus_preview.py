from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


_BACKGROUND_RGB = (8, 10, 14)

#: No padding around the crop. The vignette fades from content to
#: background *within* the crop rectangle, so there's no need for
#: a "safe landing zone" outside the content. Padding would
#: inflate the canvas aspect ratio and cause Kitty's aspect-
#: preserving place command to letterbox the image inside the
#: cell box we request.
_PADDING_PX = 0

#: Vertical vignette edge width as a FRACTION of the crop's height.
#: The image's top and bottom edges fade toward the panel background
#: over this fraction of the crop height. Only top/bottom — the
#: left/right edges of the image are NOT vignetted, because in the
#: band layout the character-space texture handles the horizontal
#: transition via a solid-block core hugging the image edge.
_TOP_BOTTOM_VIGNETTE_FRACTION = 0.15

#: Sepia tone-map target color. Source pixels are interpolated
#: toward this warm off-white by _SEPIA_MIX, which softens the
#: harsh paper-white contrast against the dark terminal background.
_SEPIA_TARGET_RGB = (236, 208, 164)

#: How strongly the sepia tone-map pulls source pixels toward the
#: warm target. 0.0 = no tint (raw scan), 1.0 = fully tinted.
#: ~0.15 is subtle — the image still reads as a scan but feels
#: warmer and less paper-white against the dark UI.
_SEPIA_MIX = 0.15


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    Applies a subtle sepia warm-tone to every source pixel, then
    feathers the top and bottom edges toward the panel background
    (vertical vignette only — the left and right edges are left
    sharp so the band's character-space texture can hug them
    directly). No corner rounding, no outline ring; those
    responsibilities now live in the textured-band renderer.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    out_width = crop_width
    out_height = crop_height
    out = bytearray(_BACKGROUND_RGB * (out_width * out_height))

    # Vertical vignette is a fraction of crop height so it scales
    # with the source regardless of DPI.
    vignette_px = max(2.0, crop_height * _TOP_BOTTOM_VIGNETTE_FRACTION)

    for y in range(crop_height):
        # Vertical alpha: fully opaque in the middle rows, fading
        # to 0 at the very top and bottom. No horizontal fade —
        # left and right edges are at full opacity to meet the
        # band's solid-block core cleanly.
        dist_from_top = y + 0.5
        dist_from_bottom = crop_height - (y + 0.5)
        edge_dist = min(dist_from_top, dist_from_bottom)
        if edge_dist >= vignette_px:
            alpha = 1.0
        else:
            t = edge_dist / vignette_px
            alpha = t * t * (3.0 - 2.0 * t)

        for x in range(crop_width):
            src_offset = (y * crop_width + x) * 3
            dest_offset = (y * out_width + x) * 3
            for channel in range(3):
                src_value = crop_rgb[src_offset + channel]
                # Apply sepia tint: lerp source toward warm target.
                tinted = (
                    src_value * (1.0 - _SEPIA_MIX)
                    + _SEPIA_TARGET_RGB[channel] * _SEPIA_MIX
                )
                # Vertical vignette: lerp tinted toward background
                # based on distance from top/bottom edge.
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
