"""Viewport contract for in-pane history scrolling (Crispy Drips).

The Paint Dry history pane currently renders "the newest N entries that
fit the visible budget" with no scroll offset. This contract pins the
behaviour of a pure-logic viewport object that carries:

1. An underlying ordered list of history entries (oldest -> newest).
2. A visible budget measured in *visual rows* (not logical entries),
   so wrapped long entries occupy their true on-screen footprint.
3. A scroll offset anchored to the *newest* visual row (offset 0 ==
   live edge; positive offset means "the bottom of the visible window
   is offset rows above the newest row").
4. Auto-follow semantics: appending new history while at the live edge
   keeps the window pinned to newest; appending while scrolled up
   MUST NOT reset the offset.
5. An explicit live-edge return affordance.

Non-goals for this contract: Rich rendering, shimmer, palette, or any
terminal input handling. Those are separate seams and have their own
contracts (or will).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_narrator_reader():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "narrator_reader.py"
    )
    spec = importlib.util.spec_from_file_location("narrator_reader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entry(kind: str, text: str):
    """Shape matches PaintDryDisplay.history tuples: (kind, text, verdict)."""
    return (kind, text, None)


class HistoryViewportContract(unittest.TestCase):
    """Pins the scroll-viewport semantics for the Paint Dry history pane."""

    # --- Construction & live-edge default -------------------------------

    def test_viewport_defaults_to_live_edge_with_zero_offset(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=5, wrap_width=80)
        self.assertEqual(vp.scroll_offset, 0)
        self.assertTrue(vp.at_live_edge)

    # --- Windowing on unwrapped entries ---------------------------------

    def test_live_edge_returns_newest_entries_fitting_visible_budget(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(6):
            vp.append(_entry("line", f"n{i}"))  # all short, 1 visual row each
        visible = vp.visible_entries()
        # Budget = 3 visual rows, so newest 3 entries: n3, n4, n5
        texts = [e[1] for e in visible]
        self.assertEqual(texts, ["n3", "n4", "n5"])

    # --- Wrap-aware visual-row accounting -------------------------------

    def test_wrapped_entries_count_as_multiple_visual_rows(self):
        module = _load_narrator_reader()
        # wrap_width=10 means an entry of length 25 occupies ceil(25/10)=3
        # visual rows. With a 4-row budget, that wrapped entry plus one
        # 1-row entry exactly fills the budget; any older entries drop.
        vp = module.HistoryViewport(visible_rows=4, wrap_width=10)
        vp.append(_entry("line", "older entry"))       # 11 chars -> 2 rows
        vp.append(_entry("line", "x" * 25))            # 25 chars -> 3 rows
        vp.append(_entry("line", "tail"))              # 4 chars  -> 1 row
        visible = vp.visible_entries()
        texts = [e[1] for e in visible]
        # Budget = 4 rows. Walking from newest: "tail" (1) + "x*25" (3) = 4.
        # "older entry" must drop entirely — partial entries are not allowed.
        self.assertEqual(texts, ["x" * 25, "tail"])

    def test_partial_entries_are_never_returned(self):
        module = _load_narrator_reader()
        # 2-row budget with a single 3-row entry should yield either the
        # whole entry or nothing, never a truncated slice.
        vp = module.HistoryViewport(visible_rows=2, wrap_width=10)
        vp.append(_entry("line", "y" * 25))  # 3 visual rows
        visible = vp.visible_entries()
        if visible:
            self.assertEqual(len(visible), 1)
            self.assertEqual(visible[0][1], "y" * 25)

    # --- Auto-follow while at live edge ---------------------------------

    def test_appending_while_at_live_edge_auto_follows_newest(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(3):
            vp.append(_entry("line", f"n{i}"))
        self.assertEqual([e[1] for e in vp.visible_entries()], ["n0", "n1", "n2"])
        vp.append(_entry("line", "n3"))
        self.assertEqual([e[1] for e in vp.visible_entries()], ["n1", "n2", "n3"])
        self.assertTrue(vp.at_live_edge)

    # --- Scrolling up pauses auto-follow --------------------------------

    def test_scrolling_up_changes_visible_slice_away_from_newest(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(6):
            vp.append(_entry("line", f"n{i}"))
        at_edge = [e[1] for e in vp.visible_entries()]
        vp.scroll_up(2)
        scrolled = [e[1] for e in vp.visible_entries()]
        self.assertNotEqual(
            scrolled, at_edge,
            "scrolling up must change the visible slice",
        )
        self.assertFalse(vp.at_live_edge)

    def test_new_history_does_not_reset_nonzero_scroll_offset(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(6):
            vp.append(_entry("line", f"n{i}"))
        vp.scroll_up(2)
        scrolled_before = [e[1] for e in vp.visible_entries()]
        offset_before = vp.scroll_offset
        vp.append(_entry("line", "fresh"))
        vp.append(_entry("line", "fresher"))
        self.assertFalse(vp.at_live_edge)
        self.assertGreaterEqual(vp.scroll_offset, offset_before)
        # Same earlier content remains visible; the viewport stayed
        # anchored to the requested earlier slice, not snapped to newest.
        self.assertEqual(
            [e[1] for e in vp.visible_entries()],
            scrolled_before,
            "new history must not yank a scrolled-up viewport back to live edge",
        )

    # --- Returning to live edge ------------------------------------------

    def test_scroll_to_live_edge_restores_newest_visibility(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(6):
            vp.append(_entry("line", f"n{i}"))
        vp.scroll_up(2)
        vp.append(_entry("line", "fresh"))
        vp.scroll_to_live_edge()
        self.assertTrue(vp.at_live_edge)
        self.assertEqual(vp.scroll_offset, 0)
        visible_texts = [e[1] for e in vp.visible_entries()]
        # Newest entry must be in the window after returning to live edge.
        self.assertIn("fresh", visible_texts)
        self.assertEqual(visible_texts[-1], "fresh")

    # --- Scroll offset is bounded ---------------------------------------

    def test_scroll_up_cannot_exceed_available_history(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(4):
            vp.append(_entry("line", f"n{i}"))
        # Try to scroll far past the oldest row.
        vp.scroll_up(9999)
        self.assertGreater(vp.scroll_offset, 0)
        # After a huge scroll_up, visible_entries must still contain the
        # oldest entry and must still respect the visible budget.
        visible = vp.visible_entries()
        self.assertLessEqual(len(visible), 4)
        self.assertIn("n0", [e[1] for e in visible])

    def test_scroll_down_past_live_edge_clamps_to_live_edge(self):
        module = _load_narrator_reader()
        vp = module.HistoryViewport(visible_rows=3, wrap_width=80)
        for i in range(5):
            vp.append(_entry("line", f"n{i}"))
        vp.scroll_up(2)
        vp.scroll_down(9999)
        self.assertTrue(vp.at_live_edge)
        self.assertEqual(vp.scroll_offset, 0)


if __name__ == "__main__":
    unittest.main()
