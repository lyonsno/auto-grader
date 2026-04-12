from __future__ import annotations

import unittest

import fitz

from auto_grader.eval_harness import FocusRegion


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    samples = bytes(rgb * (width * height))
    pix = fitz.Pixmap(fitz.csRGB, width, height, samples, False)
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
        # Use a larger test page so the crop dimensions are easy to
        # reason about. At 1600x1200 with a centered 50% region,
        # the crop is exactly 800x600.
        self.page_png = _solid_png(1600, 1200, (220, 220, 220))
        self.focus = FocusRegion(
            page=1,
            x=0.25,
            y=0.25,
            width=0.50,
            height=0.50,
            source="template",
        )

    def test_preview_returns_png_at_exact_crop_size(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        self.assertIsInstance(preview, bytes)
        width, height = _png_dimensions(preview)
        self.assertEqual(width, 800)
        self.assertEqual(height, 600)

    def test_preview_applies_visible_uniform_sepia_to_the_crop(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        center = _pixel_at(preview, width // 2, height // 2)
        # Source is flat 220/220/220, so any warm-tone pass should
        # lower blue more than red and keep the crop bright overall.
        self.assertGreater(center[0], center[1])
        self.assertGreater(center[1], center[2])
        self.assertGreater(sum(center), 3 * 170)
        self.assertNotEqual(center, (220, 220, 220))

    def test_preview_no_longer_vignettes_or_rounds_edges(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        near_top = _pixel_at(preview, width // 2, 2)
        near_side = _pixel_at(preview, 2, height // 2)
        near_corner = _pixel_at(preview, 2, 2)
        center = _pixel_at(preview, width // 2, height // 2)

        self.assertEqual(near_top, center)
        self.assertEqual(near_side, center)
        self.assertEqual(near_corner, center)


if __name__ == "__main__":
    unittest.main()
