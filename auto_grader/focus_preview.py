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
_WARM_MIX = 0.35

#: Ink softening — pull pure blacks this fraction toward a dark
#: warm gray so ink strokes feel printed-on-paper rather than
#: punching through. Only affects dark pixels (luminance < threshold).
_INK_SOFTEN_TARGET_RGB = (45, 38, 32)
_INK_SOFTEN_MIX = 0.30
_INK_SOFTEN_LUMINANCE_THRESHOLD = 80

#: Paper grain — low-amplitude deterministic noise overlaid on
#: every pixel to break up flat paper regions and give the image
#: a matte-print texture. Amplitude is in [0, 255] units.
_GRAIN_AMPLITUDE = 8
#: Grain is applied more strongly to light pixels (paper) and
#: almost not at all to dark pixels (ink), so it reads as paper
#: texture rather than signal noise.
_GRAIN_INK_IMMUNITY = 0.85

#: Paint Dry is a terminal surface, not an archival image viewer. Keep
#: preview crops bounded so the sink never tries to push poster-sized PNGs
#: through the fifo.
_FOCUS_PREVIEW_MAX_LONG_EDGE = 1400
_FOCUS_PREVIEW_MAX_TOTAL_PIXELS = 600_000


def render_focus_preview(page_png: bytes, focus_region: FocusRegion) -> bytes:
    """Render a terminal-friendly preview crop for a focus region.

    Pipeline: crop → warm tone-map → ink softening → paper grain.
    No vignette, no corner rounding, no outline ring — edge treatment
    is handled entirely by the textured-band renderer.
    """
    crop_rgb, crop_width, crop_height = _crop_focus_region(page_png, focus_region)
    crop_rgb, crop_width, crop_height = _downscale_preview_if_needed(
        crop_rgb,
        crop_width,
        crop_height,
    )
    out = bytearray(crop_width * crop_height * 3)

    for y in range(crop_height):
        for x in range(crop_width):
            offset = (y * crop_width + x) * 3
            r = crop_rgb[offset]
            g = crop_rgb[offset + 1]
            b = crop_rgb[offset + 2]

            # Luminance for ink detection (fast approximation).
            lum = 0.299 * r + 0.587 * g + 0.114 * b

            for channel in range(3):
                src_value = crop_rgb[offset + channel]

                # 1. Warm bone tone-map (all pixels).
                value = (
                    src_value * (1.0 - _WARM_MIX)
                    + _WARM_TARGET_RGB[channel] * _WARM_MIX
                )

                # 2. Ink softening (dark pixels only).
                if lum < _INK_SOFTEN_LUMINANCE_THRESHOLD:
                    ink_t = 1.0 - (lum / _INK_SOFTEN_LUMINANCE_THRESHOLD)
                    value = (
                        value * (1.0 - _INK_SOFTEN_MIX * ink_t)
                        + _INK_SOFTEN_TARGET_RGB[channel] * _INK_SOFTEN_MIX * ink_t
                    )

                # 3. Paper grain (light pixels mostly).
                # Deterministic noise from pixel coordinates.
                grain_seed = ((y * 7919 + x * 104729 + channel * 31) ^ 0xA5A5) & 0xFFFF
                grain_noise = ((grain_seed * 48271) & 0xFFFF) / 65535.0  # [0, 1)
                grain_noise = (grain_noise - 0.5) * 2.0  # [-1, 1)
                # Scale: full amplitude on paper, nearly zero on ink.
                grain_weight = min(1.0, lum / 255.0)
                grain_weight = grain_weight * (1.0 - _GRAIN_INK_IMMUNITY) + _GRAIN_INK_IMMUNITY * grain_weight ** 2
                value += grain_noise * _GRAIN_AMPLITUDE * grain_weight

                out[offset + channel] = max(0, min(255, int(round(value))))

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


def _downscale_preview_if_needed(
    rgb: bytes,
    width: int,
    height: int,
) -> tuple[bytes, int, int]:
    long_edge = max(width, height)
    area = width * height
    if (
        long_edge <= _FOCUS_PREVIEW_MAX_LONG_EDGE
        and area <= _FOCUS_PREVIEW_MAX_TOTAL_PIXELS
    ):
        return rgb, width, height

    scale = min(
        _FOCUS_PREVIEW_MAX_LONG_EDGE / long_edge,
        math.sqrt(_FOCUS_PREVIEW_MAX_TOTAL_PIXELS / area),
        1.0,
    )
    dest_width = max(1, int(math.floor(width * scale)))
    dest_height = max(1, int(math.floor(height * scale)))
    while dest_width * dest_height > _FOCUS_PREVIEW_MAX_TOTAL_PIXELS:
        if dest_width >= dest_height and dest_width > 1:
            dest_width -= 1
            continue
        if dest_height > 1:
            dest_height -= 1
            continue
        break
    dest = bytearray(dest_width * dest_height * 3)

    x_scale = width / dest_width
    y_scale = height / dest_height

    for dest_y in range(dest_height):
        src_y = min(height - 1, int(dest_y * y_scale))
        for dest_x in range(dest_width):
            src_x = min(width - 1, int(dest_x * x_scale))
            src_offset = (src_y * width + src_x) * 3
            dest_offset = (dest_y * dest_width + dest_x) * 3
            dest[dest_offset : dest_offset + 3] = rgb[src_offset : src_offset + 3]

    return bytes(dest), dest_width, dest_height


def _png_to_pixmap(png_bytes: bytes) -> fitz.Pixmap:
    return fitz.Pixmap(png_bytes)
