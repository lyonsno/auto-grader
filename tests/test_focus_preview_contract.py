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


if __name__ == "__main__":
    unittest.main()
