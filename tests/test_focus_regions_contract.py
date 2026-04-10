from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_grader.eval_harness import FocusRegion
from auto_grader.focus_regions import (
    load_focus_regions,
    save_focus_regions,
)


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
