from __future__ import annotations

import unittest

import fitz

from auto_grader.eval_harness import FocusRegion


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    samples = bytes(rgb * (width * height))
    pix = fitz.Pixmap(fitz.csRGB, width, height, samples, False)
    return pix.tobytes("png")


def _bordered_png(
    width: int,
    height: int,
    *,
    border: int,
    inner_rgb: tuple[int, int, int],
    border_rgb: tuple[int, int, int],
    noisy: bool = False,
) -> bytes:
    rows = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            rgb = (
                border_rgb
                if x < border or x >= width - border or y < border or y >= height - border
                else inner_rgb
            )
            if noisy and rgb == border_rgb and (x + y) % 11 == 0:
                rgb = tuple(max(0, c - 11) for c in border_rgb)
            off = (y * width + x) * 3
            rows[off : off + 3] = bytes(rgb)
    pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(rows), False)
    return pix.tobytes("png")


def _content_box_png(
    width: int,
    height: int,
    *,
    bg_rgb: tuple[int, int, int],
    box_rgb: tuple[int, int, int],
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
) -> bytes:
    rows = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            rgb = (
                box_rgb
                if box_x <= x < box_x + box_w and box_y <= y < box_y + box_h
                else bg_rgb
            )
            off = (y * width + x) * 3
            rows[off : off + 3] = bytes(rgb)
    pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(rows), False)
    return pix.tobytes("png")


def _noisy_margin_content_box_png(
    width: int,
    height: int,
    *,
    bg_rgb: tuple[int, int, int],
    noisy_bg_rgb: tuple[int, int, int],
    box_rgb: tuple[int, int, int],
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
) -> bytes:
    rows = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            in_box = box_x <= x < box_x + box_w and box_y <= y < box_y + box_h
            if in_box:
                rgb = box_rgb
            elif (x * 7 + y * 11) % 5 == 0:
                rgb = noisy_bg_rgb
            else:
                rgb = bg_rgb
            off = (y * width + x) * 3
            rows[off : off + 3] = bytes(rgb)
    pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(rows), False)
    return pix.tobytes("png")


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    pix = fitz.Pixmap(png_bytes)
    return pix.width, pix.height


def _pixel_at(png_bytes: bytes, x: int, y: int) -> tuple[int, int, int]:
    pix = fitz.Pixmap(png_bytes)
    offset = (y * pix.width + x) * pix.n
    sample = pix.samples[offset : offset + pix.n]
    return int(sample[0]), int(sample[1]), int(sample[2])


class FocusPreviewContract(unittest.TestCase):
    def setUp(self) -> None:
        # Use a larger test page so the crop is large enough that any
        # accidental vignette/edge treatment shows up clearly in the
        # sampled pixels.
        self.page_png = _solid_png(1600, 1200, (220, 220, 220))
        self.focus = FocusRegion(
            page=1,
            x=0.25,
            y=0.25,
            width=0.50,
            height=0.50,
            source="template",
        )

    def test_preview_returns_png_at_crop_size_without_padding(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        self.assertIsInstance(preview, bytes)
        width, height = _png_dimensions(preview)
        self.assertEqual(
            (width, height),
            (800, 600),
            "focus preview should render at the crop size itself now that the "
            "edge treatment lives in the textured band, not in preview padding",
        )

    def test_preview_keeps_edges_as_warm_as_center_without_vignette(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        center = _pixel_at(preview, width // 2, height // 2)
        edge_inside = _pixel_at(preview, 2, height // 2)

        self.assertEqual(
            center,
            edge_inside,
            "preview pixels should no longer vignette toward the edges; the "
            "textured band owns edge treatment now",
        )
        self.assertGreater(min(center), 180)

    def test_preview_applies_visible_warm_tone(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        center = _pixel_at(preview, width // 2, height // 2)
        self.assertLess(
            center[2],
            center[0],
            "preview should keep the warm sepia/bone treatment rather than "
            "rendering as a neutral gray scan",
        )
        self.assertLess(
            center[2],
            center[1],
            "preview should skew warm enough that blue remains the coolest "
            "channel even on bright paper",
        )

    def test_preview_trims_uniform_paper_matte_before_tinting(self):
        from auto_grader.focus_preview import render_focus_preview

        page_png = _bordered_png(
            1600,
            1200,
            border=120,
            inner_rgb=(180, 150, 120),
            border_rgb=(231, 221, 199),
        )
        focus = FocusRegion(page=1, x=0.0, y=0.0, width=1.0, height=1.0, source="template")
        preview = render_focus_preview(page_png, focus)

        self.assertEqual(
            _png_dimensions(preview),
            (1360, 960),
            "focus preview should trim the all-around paper-colored scan "
            "matte instead of preserving a thick cream frame around the crop",
        )

    def test_preview_trims_noisy_paper_matte_before_tinting(self):
        from auto_grader.focus_preview import render_focus_preview

        page_png = _bordered_png(
            1600,
            1200,
            border=120,
            inner_rgb=(180, 150, 120),
            border_rgb=(231, 221, 199),
            noisy=True,
        )
        focus = FocusRegion(page=1, x=0.0, y=0.0, width=1.0, height=1.0, source="template")
        preview = render_focus_preview(page_png, focus)

        self.assertEqual(
            _png_dimensions(preview),
            (1360, 960),
            "paper-colored scan borders should still trim even with small "
            "scanner-noise variation along the edge",
        )

    def test_preview_tightens_to_content_bounds_when_margin_dominates(self):
        from auto_grader.focus_preview import render_focus_preview

        page_png = _content_box_png(
            1600,
            1200,
            bg_rgb=(231, 221, 199),
            box_rgb=(40, 35, 28),
            box_x=600,
            box_y=420,
            box_w=400,
            box_h=260,
        )
        focus = FocusRegion(page=1, x=0.0, y=0.0, width=1.0, height=1.0, source="template")
        preview = render_focus_preview(page_png, focus)

        self.assertEqual(
            _png_dimensions(preview),
            (400, 260),
            "when the requested region is mostly uniform paper margin, the "
            "preview should trim cleanly all the way to the real content "
            "island rather than preserving a page-sized cream surround",
        )

    def test_preview_tightens_to_content_bounds_even_with_heavier_margin_wash(self):
        from auto_grader.focus_preview import render_focus_preview

        page_png = _noisy_margin_content_box_png(
            1600,
            1200,
            bg_rgb=(231, 221, 199),
            noisy_bg_rgb=(188, 176, 158),
            box_rgb=(40, 35, 28),
            box_x=600,
            box_y=420,
            box_w=400,
            box_h=260,
        )
        focus = FocusRegion(page=1, x=0.0, y=0.0, width=1.0, height=1.0, source="template")
        preview = render_focus_preview(page_png, focus)

        self.assertEqual(
            _png_dimensions(preview),
            (408, 268),
            "even when scanner wash breaks simple edge-matte detection, the "
            "preview should tighten to the actual content bounds with only a "
            "small pad instead of preserving a page-sized cream surround",
        )


if __name__ == "__main__":
    unittest.main()
