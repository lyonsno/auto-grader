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
        self.page_png = _solid_png(80, 60, (220, 220, 220))
        self.focus = FocusRegion(
            page=1,
            x=0.25,
            y=0.25,
            width=0.50,
            height=0.50,
            source="template",
        )

    def test_preview_returns_png_with_padding_and_box_outline(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        self.assertIsInstance(preview, bytes)
        self.assertEqual(_png_dimensions(preview), (56, 46))
        self.assertEqual(_pixel_at(preview, 0, 10), (64, 84, 120))

    def test_preview_keeps_center_bright_but_vignettes_edges(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        center = _pixel_at(preview, 28, 23)
        left_edge = _pixel_at(preview, 9, 23)

        self.assertGreater(sum(center), sum(left_edge))
        self.assertGreater(min(center), 180)

    def test_preview_softens_corners_into_terminal_background(self):
        from auto_grader.focus_preview import render_focus_preview

        preview = render_focus_preview(self.page_png, self.focus)
        near_corner = _pixel_at(preview, 9, 9)
        center = _pixel_at(preview, 28, 23)

        self.assertLess(sum(near_corner), sum(center))
        self.assertLess(sum(near_corner), 3 * 120)


if __name__ == "__main__":
    unittest.main()
