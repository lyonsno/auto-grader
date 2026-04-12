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
        # Use a larger test page so the proportional vignette/padding
        # produces meaningfully measurable effects. At 1600x1200 the
        # crop is 800x600, small-dim 600, vignette ~30 px, corner
        # radius ~15 px — visible in the output.
        self.page_png = _solid_png(1600, 1200, (220, 220, 220))
        self.focus = FocusRegion(
            page=1,
            x=0.25,
            y=0.25,
            width=0.50,
            height=0.50,
            source="template",
        )

    def test_preview_returns_png_with_proportional_padding(self):
        from auto_grader.focus_preview import render_focus_preview
        from auto_grader.focus_preview import _PADDING_PX

        preview = render_focus_preview(self.page_png, self.focus)
        self.assertIsInstance(preview, bytes)
        width, height = _png_dimensions(preview)
        # Crop is 800x600 (50% of 1600x1200), plus padding on each side.
        self.assertEqual(width, 800 + 2 * _PADDING_PX)
        self.assertEqual(height, 600 + 2 * _PADDING_PX)

    def test_preview_keeps_center_bright_but_vignettes_top_and_bottom(self):
        from auto_grader.focus_preview import render_focus_preview
        from auto_grader.focus_preview import _PADDING_PX

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        # Center of the image should be the original page color.
        center = _pixel_at(preview, width // 2, height // 2)
        # In the textured-band renderer, only the top and bottom edges
        # vignette toward the background. The left/right edges remain
        # sharp so the character-texture band can hug them directly.
        top_edge_inside = _pixel_at(preview, width // 2, _PADDING_PX + 2)
        side_edge_inside = _pixel_at(preview, _PADDING_PX + 2, height // 2)

        self.assertGreater(sum(center), sum(top_edge_inside))
        self.assertEqual(
            sum(center),
            sum(side_edge_inside),
            "side edges should stay sharp in the band-layout renderer",
        )
        # Center should still be warm and bright after the sepia pass.
        self.assertGreater(min(center), 180)

    def test_preview_softens_corners_into_terminal_background(self):
        from auto_grader.focus_preview import render_focus_preview
        from auto_grader.focus_preview import _PADDING_PX

        preview = render_focus_preview(self.page_png, self.focus)
        width, height = _png_dimensions(preview)
        # A pixel just inside the padded corner region.
        near_corner = _pixel_at(preview, _PADDING_PX + 2, _PADDING_PX + 2)
        center = _pixel_at(preview, width // 2, height // 2)

        self.assertLess(sum(near_corner), sum(center))
        # Corner should be substantially darker than a bright page.
        self.assertLess(sum(near_corner), 3 * 180)


if __name__ == "__main__":
    unittest.main()
