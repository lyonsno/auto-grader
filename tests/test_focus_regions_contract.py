from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from auto_grader.eval_harness import FocusRegion
from auto_grader.focus_regions import (
    load_focus_regions,
    save_focus_regions,
)


def _load_annotator_module():
    # The annotator script is not an importable package, so load it
    # through importlib. tkinter import at module top is harmless in
    # headless CI — it's instantiating tk.Tk() that would break, and
    # the tests never do that.
    spec = importlib.util.spec_from_file_location(
        "annotate_focus_regions",
        Path(__file__).resolve().parent.parent / "scripts" / "annotate_focus_regions.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FocusRegionsRoundTripContract(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        # A first-ever annotation run must not require the file to exist
        # already. Absent file → empty dict, caller proceeds.
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.yaml"
            self.assertEqual(load_focus_regions(missing), {})

    def test_roundtrip_preserves_every_field(self):
        regions = {
            ("27-blue-2023", "fr-3"): FocusRegion(
                page=1,
                x=0.14,
                y=0.18,
                width=0.66,
                height=0.24,
                source="operator_annotated",
            ),
            ("34-blue", "fr-12a"): FocusRegion(
                page=4,
                x=0.18,
                y=0.07,
                width=0.50,
                height=0.18,
                source="mock_tricky_plus",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "focus_regions.yaml"
            save_focus_regions(path, regions)
            loaded = load_focus_regions(path)

        self.assertEqual(loaded.keys(), regions.keys())
        for key, original in regions.items():
            roundtripped = loaded[key]
            self.assertEqual(roundtripped.page, original.page)
            self.assertAlmostEqual(roundtripped.x, original.x, places=6)
            self.assertAlmostEqual(roundtripped.y, original.y, places=6)
            self.assertAlmostEqual(roundtripped.width, original.width, places=6)
            self.assertAlmostEqual(roundtripped.height, original.height, places=6)
            self.assertEqual(roundtripped.source, original.source)

    def test_save_is_stable_across_runs(self):
        # Re-saving identical state must produce byte-identical output.
        # Otherwise concurrent sessions can't diff their outputs cleanly
        # and the file becomes a diff-churn hazard.
        regions = {
            ("b", "q2"): FocusRegion(
                page=1, x=0.1, y=0.2, width=0.3, height=0.4, source="s"
            ),
            ("a", "q1"): FocusRegion(
                page=2, x=0.5, y=0.6, width=0.2, height=0.1, source="s"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.yaml"
            path_b = Path(tmp) / "b.yaml"
            save_focus_regions(path_a, regions)
            save_focus_regions(path_b, regions)
            content_a = path_a.read_bytes()
            content_b = path_b.read_bytes()
        self.assertEqual(content_a, content_b)

    def test_malformed_top_level_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("- not\n- a\n- mapping\n")
            with self.assertRaises(ValueError):
                load_focus_regions(path)

    def test_parse_targets_expands_dotdot_range_in_ground_truth_order(self):
        annotator = _load_annotator_module()
        # fr-1..fr-3 → fr-1, fr-2, fr-3 (contiguous in the canonical
        # ground-truth sequence).
        result = annotator._parse_targets("15-blue/fr-1..fr-3")
        self.assertEqual(
            result,
            [("15-blue", "fr-1"), ("15-blue", "fr-2"), ("15-blue", "fr-3")],
        )

    def test_parse_targets_range_includes_multi_part_sub_questions(self):
        # fr-5a..fr-6 → fr-5a, fr-5b, fr-5c, fr-6. Sub-parts like
        # "fr-5a/b/c" are distinct question IDs in the sequence and
        # must be included when the range straddles them.
        annotator = _load_annotator_module()
        result = annotator._parse_targets("15-blue/fr-5a..fr-6")
        self.assertEqual(
            result,
            [
                ("15-blue", "fr-5a"),
                ("15-blue", "fr-5b"),
                ("15-blue", "fr-5c"),
                ("15-blue", "fr-6"),
            ],
        )

    def test_parse_targets_mixes_ranges_and_explicit_entries(self):
        annotator = _load_annotator_module()
        result = annotator._parse_targets(
            "15-blue/fr-1..fr-2, 15-blue/fr-12a"
        )
        self.assertEqual(
            result,
            [
                ("15-blue", "fr-1"),
                ("15-blue", "fr-2"),
                ("15-blue", "fr-12a"),
            ],
        )

    def test_parse_targets_explicit_list_still_works(self):
        annotator = _load_annotator_module()
        result = annotator._parse_targets("15-blue/fr-1, 15-blue/fr-3")
        self.assertEqual(
            result,
            [("15-blue", "fr-1"), ("15-blue", "fr-3")],
        )

    def test_parse_targets_rejects_range_with_unknown_endpoint(self):
        annotator = _load_annotator_module()
        with self.assertRaises(SystemExit):
            annotator._parse_targets("15-blue/fr-1..fr-999")

    def test_ditto_box_none_when_no_previous_target(self):
        annotator = _load_annotator_module()
        # First target in sequence — no previous, no ditto.
        result = annotator._ditto_box_for(0, {})
        self.assertIsNone(result)

    def test_ditto_box_none_when_previous_was_not_accepted(self):
        annotator = _load_annotator_module()
        # Current is index 1, but index 0 was skipped (not in session_accepted).
        result = annotator._ditto_box_for(1, {})
        self.assertIsNone(result)

    def test_ditto_box_returns_immediately_previous_accepted(self):
        annotator = _load_annotator_module()
        box_a = FocusRegion(
            page=1, x=0.1, y=0.2, width=0.3, height=0.4, source="operator_annotated"
        )
        result = annotator._ditto_box_for(1, {0: box_a})
        self.assertEqual(result, box_a)

    def test_ditto_box_does_not_walk_back_past_gaps(self):
        annotator = _load_annotator_module()
        # Current is index 3. Index 2 was skipped. Index 0 was accepted.
        # Ditto must NOT silently walk back to index 0 — that would
        # surprise the operator by copying a box from an unrelated
        # earlier target.
        box_a = FocusRegion(
            page=1, x=0.1, y=0.2, width=0.3, height=0.4, source="operator_annotated"
        )
        result = annotator._ditto_box_for(3, {0: box_a})
        self.assertIsNone(result)

    def test_ditto_box_uses_most_recent_accepted_at_previous_index(self):
        annotator = _load_annotator_module()
        box_a = FocusRegion(
            page=1, x=0.1, y=0.2, width=0.3, height=0.4, source="operator_annotated"
        )
        box_c = FocusRegion(
            page=1, x=0.5, y=0.6, width=0.2, height=0.1, source="operator_annotated"
        )
        result = annotator._ditto_box_for(3, {0: box_a, 2: box_c})
        self.assertEqual(result, box_c)

    def test_compute_display_dpi_fits_page_inside_max_bounds(self):
        annotator = _load_annotator_module()
        # US-letter-ish page in points.
        dpi = annotator._compute_display_dpi(
            610.0, 789.0, max_width=1400, max_height=1600
        )
        # The rasterized dimensions (point × dpi / 72) must not exceed
        # max bounds on either axis.
        width_px = 610.0 * dpi / 72.0
        height_px = 789.0 * dpi / 72.0
        self.assertLessEqual(
            int(round(width_px)),
            1400,
            "display DPI must produce a pixmap that fits max width",
        )
        self.assertLessEqual(
            int(round(height_px)),
            1600,
            "display DPI must produce a pixmap that fits max height",
        )
        # At least one axis should be at or near the max, otherwise we
        # picked a DPI lower than necessary.
        self.assertTrue(
            int(round(width_px)) == 1400 or int(round(height_px)) == 1600,
            "display DPI should saturate one of the max bounds",
        )

    def test_rasterize_page_for_display_returns_exact_pixel_dimensions(self):
        # End-to-end contract: the PNG returned by
        # _rasterize_page_for_display must have pixel dimensions equal
        # to the (width_px, height_px) it reports. If the PNG is
        # secretly smaller than the reported canvas size, the operator
        # draws against a sub-region of the canvas and coordinates
        # normalize against the wrong dimensions — that was the bug
        # that corrupted every annotation on 2026-04-10.
        annotator = _load_annotator_module()
        pdf_path = Path(
            "/Users/noahlyons/dev/auto-grader-assets/scans/15 blue.pdf"
        )
        if not pdf_path.exists():
            self.skipTest(f"scan PDF not available at {pdf_path}")
        png_bytes, width_px, height_px = annotator._rasterize_page_for_display(
            pdf_path, 1
        )
        # Read back the actual pixel dimensions of the PNG by loading
        # it through fitz (same library the tool uses).
        import fitz
        pix = fitz.Pixmap(png_bytes)
        self.assertEqual(
            pix.width,
            width_px,
            "reported width_px must equal actual PNG pixel width",
        )
        self.assertEqual(
            pix.height,
            height_px,
            "reported height_px must equal actual PNG pixel height",
        )
        # And both must fit inside the max display bounds.
        self.assertLessEqual(pix.width, annotator._MAX_DISPLAY_WIDTH)
        self.assertLessEqual(pix.height, annotator._MAX_DISPLAY_HEIGHT)

    def test_normalize_drag_box_basic_fully_inside(self):
        annotator = _load_annotator_module()
        # Drag from (100, 200) to (400, 600) inside a 1000x1500 canvas.
        result = annotator._normalize_drag_box(
            (100, 200), (400, 600), display_width=1000, display_height=1500
        )
        self.assertIsNotNone(result)
        x, y, w, h = result
        self.assertAlmostEqual(x, 0.1, places=6)
        self.assertAlmostEqual(y, 200 / 1500, places=6)
        self.assertAlmostEqual(w, 0.3, places=6)
        self.assertAlmostEqual(h, 400 / 1500, places=6)

    def test_normalize_drag_box_tap_returns_none(self):
        annotator = _load_annotator_module()
        # Click-release at almost the same point is a tap, not a drag.
        result = annotator._normalize_drag_box(
            (500, 500), (501, 501), display_width=1000, display_height=1000
        )
        self.assertIsNone(result)

    def test_normalize_drag_box_reversed_drag_is_swapped(self):
        annotator = _load_annotator_module()
        # Drag from bottom-right to top-left — same geometric box.
        forward = annotator._normalize_drag_box(
            (100, 200), (400, 600), display_width=1000, display_height=1500
        )
        reversed_ = annotator._normalize_drag_box(
            (400, 600), (100, 200), display_width=1000, display_height=1500
        )
        self.assertEqual(forward, reversed_)

    def test_normalize_drag_box_clamps_overdraw_to_display_bounds(self):
        annotator = _load_annotator_module()
        # Release at (-50, 2000) on a 1000x1500 canvas: clamp to (0, 1500).
        result = annotator._normalize_drag_box(
            (100, 200), (-50, 2000), display_width=1000, display_height=1500
        )
        self.assertIsNotNone(result)
        x, y, w, h = result
        # After clamping, the box runs from (0, 200) to (100, 1500).
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 200 / 1500, places=6)
        self.assertAlmostEqual(w, 100 / 1000, places=6)
        self.assertAlmostEqual(h, (1500 - 200) / 1500, places=6)

    def test_normalize_drag_box_display_dims_equal_image_dims_invariant(self):
        # Regression guard against the _build_display_image bug.
        # The invariant is: whatever (display_width, display_height)
        # the caller passes is the actual pixel dimensions of the
        # image it will be drawing on. If someone ever re-introduces
        # a scale step that produces a smaller-than-canvas image,
        # this test does not catch that directly — but together with
        # test_rasterize_page_for_display_returns_exact_pixel_dimensions
        # above, the pair forms a pincer: rasterize reports exact
        # dims, normalize uses those dims, therefore stored coords
        # are against actual image pixels.
        annotator = _load_annotator_module()
        # Sanity: normalizing a known drag matches hand-computed values
        # at small and large display sizes.
        for dw, dh in ((100, 100), (611, 789), (1239, 1600), (6784, 8764)):
            result = annotator._normalize_drag_box(
                (dw // 4, dh // 4),
                (3 * dw // 4, 3 * dh // 4),
                display_width=dw,
                display_height=dh,
            )
            self.assertIsNotNone(result)
            x, y, w, h = result
            # Expected: quarter-to-three-quarters on each axis, which
            # is always (0.25, 0.25, 0.5, 0.5) for any display size
            # where the integer division rounds cleanly. Allow a small
            # tolerance for integer rounding.
            self.assertAlmostEqual(x, 0.25, places=2)
            self.assertAlmostEqual(y, 0.25, places=2)
            self.assertAlmostEqual(w, 0.5, places=2)
            self.assertAlmostEqual(h, 0.5, places=2)
    def test_migrated_default_file_loads(self):
        # The on-disk default file must parse cleanly and produce the
        # expected number of entries. This is a minimal reality check
        # that the migration from the old dict literal is not silently
        # dropping entries — a real invariant worth protecting.
        default_path = (
            Path(__file__).resolve().parent.parent
            / "eval"
            / "focus_regions.yaml"
        )
        if not default_path.exists():
            self.skipTest(f"default focus regions file not present at {default_path}")
        regions = load_focus_regions(default_path)
        # The migrated set had 12 entries; if this file shrinks below
        # that in some future edit, that's a loud failure rather than a
        # silent drop.
        self.assertGreaterEqual(len(regions), 12)
        # Spot-check a known entry.
        self.assertIn(("27-blue-2023", "fr-5b"), regions)


if __name__ == "__main__":
    unittest.main()
