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
