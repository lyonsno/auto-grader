from __future__ import annotations

import math

import fitz

from auto_grader.eval_harness import FocusRegion


#: Warm-tone target color — bone from the narrator's moss/bone
#: palette. Source pixels are interpolated toward this warm off-white
#: by _WARM_MIX, which softens the harsh paper-white contrast
#: against the dark terminal background.
_WARM_TARGET_RGB = (220, 205, 180)

#: How strongly the warm tone-map pulls source pixels toward the
#: target. 0.0 = no tint (raw scan), 1.0 = fully tinted.
#: ~0.35 gives the paper a visible parchment warmth without
#: washing out ink contrast.
_WARM_MIX = 0.35


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    Applies a bone warm-tone to every source pixel. No vignette,
    no corner rounding, no outline ring — edge treatment is handled
    entirely by the textured-band renderer in narrator_reader.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    out = bytearray(crop_width * crop_height * 3)

    for y in range(crop_height):
        for x in range(crop_width):
            offset = (y * crop_width + x) * 3
            for channel in range(3):
                src_value = crop_rgb[offset + channel]
                # Lerp source toward warm bone target.
                value = (
                    src_value * (1.0 - _WARM_MIX)
                    + _WARM_TARGET_RGB[channel] * _WARM_MIX
                )
                out[offset + channel] = int(round(value))

    pix = fitz.Pixmap(fitz.csRGB, crop_width, crop_height, bytes(out), False)
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
