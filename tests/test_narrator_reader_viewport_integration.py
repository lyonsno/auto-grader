"""PaintDryDisplay viewport integration contract.

Pins that PaintDryDisplay drives its history pane through a
HistoryViewport rather than a static "newest-N entries" trim:

* `PaintDryDisplay` exposes a viewport accessor and scroll methods
  (scroll_history_up / scroll_history_down / scroll_history_to_live_edge).
* At the live edge, the rendered entries are behaviorally the same
  as the previous newest-first-priority-fill: headers and topics for
  visible items always win over optional narrator lines.
* Calling scroll_history_up actually changes which committed entries
  appear in the visible slice — the pane is no longer pinned to
  newest-only.
* Committing new history while scrolled up does NOT reset the scroll
  state or yank the visible slice back to newest.
* Calling scroll_history_to_live_edge restores newest-first visibility.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from rich.console import Group
from rich.text import Text


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


def _commit_item(display, module, header: str, lines: list[str], topic: str) -> None:
    """Commit a full item to PaintDryDisplay's history the same way the
    sink does: one header, then narrator lines, then a topic."""
    display.history.append(("header", header, None))
    for line in lines:
        display.history.append(("line", line, 0))
    display.history.append(("topic", topic, None))


def _extract_plain(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_extract_plain(child) for child in renderable.renderables)
    if hasattr(renderable, "renderable"):
        return _extract_plain(renderable.renderable)
    return str(renderable)


class PaintDryDisplayViewportIntegration(unittest.TestCase):

    def _display(self, module):
        d = module.PaintDryDisplay()
        # Force a known wrap width without needing a real Rich console.
        d._wrap_width_override = 80
        return d

    def test_display_exposes_history_viewport(self):
        module = _load_narrator_reader()
        display = self._display(module)
        vp = display.history_viewport()
        self.assertIsInstance(vp, module.HistoryViewport)
        self.assertTrue(vp.at_live_edge)

    def test_display_scroll_history_up_changes_visible_slice(self):
        module = _load_narrator_reader()
        display = self._display(module)
        # Commit more items than the visible budget can hold so there
        # is real earlier content to reveal on scroll-up.
        for i in range(8):
            _commit_item(
                display, module,
                header=f"[item {i+1}/8] test-{i}",
                lines=[f"line-{i}-a", f"line-{i}-b"],
                topic=f"topic-{i}",
            )
        edge_texts = [e[1] for e in display.history_viewport().visible_entries()]
        display.scroll_history_up(5)
        scrolled_texts = [e[1] for e in display.history_viewport().visible_entries()]
        self.assertNotEqual(
            scrolled_texts, edge_texts,
            "scroll_history_up must change the visible slice",
        )
        self.assertFalse(display.history_viewport().at_live_edge)

    def test_new_history_does_not_reset_scrolled_viewport(self):
        module = _load_narrator_reader()
        display = self._display(module)
        for i in range(8):
            _commit_item(
                display, module,
                header=f"[item {i+1}/8] test-{i}",
                lines=[f"line-{i}-a"],
                topic=f"topic-{i}",
            )
        display.scroll_history_up(4)
        scrolled_texts_before = [
            e[1] for e in display.history_viewport().visible_entries()
        ]
        self.assertFalse(display.history_viewport().at_live_edge)
        # Commit a fresh item while scrolled up.
        _commit_item(
            display, module,
            header="[item 9/9] fresh",
            lines=["fresh-line"],
            topic="fresh-topic",
        )
        self.assertFalse(
            display.history_viewport().at_live_edge,
            "committing new history must not yank the viewport back to live edge",
        )
        scrolled_texts_after = [
            e[1] for e in display.history_viewport().visible_entries()
        ]
        self.assertEqual(
            scrolled_texts_after, scrolled_texts_before,
            "the earlier visible slice must remain stable across a commit",
        )

    def test_scroll_to_live_edge_shows_newest_entries(self):
        module = _load_narrator_reader()
        display = self._display(module)
        for i in range(8):
            _commit_item(
                display, module,
                header=f"[item {i+1}/8] test-{i}",
                lines=[f"line-{i}"],
                topic=f"topic-{i}",
            )
        display.scroll_history_up(5)
        _commit_item(
            display, module,
            header="[item 9/9] freshest",
            lines=["freshest-line"],
            topic="freshest-topic",
        )
        display.scroll_history_to_live_edge()
        self.assertTrue(display.history_viewport().at_live_edge)
        visible_texts = [
            e[1] for e in display.history_viewport().visible_entries()
        ]
        self.assertIn("[item 9/9] freshest", visible_texts)
        self.assertIn("freshest-topic", visible_texts)

    def test_live_edge_still_prioritizes_essentials_over_narrator_lines(self):
        module = _load_narrator_reader()
        display = self._display(module)
        # One item with many narrator lines, plus several later items.
        # Essentials (headers + topics) for all items must be visible
        # at the live edge even when narrator lines from the chatty
        # item would otherwise consume the budget.
        _commit_item(
            display, module,
            header="[item 1/5] chatty",
            lines=[f"chatty-{i}" for i in range(40)],
            topic="chatty-topic",
        )
        for i in range(2, 6):
            _commit_item(
                display, module,
                header=f"[item {i}/5] short-{i}",
                lines=[f"short-line-{i}"],
                topic=f"short-topic-{i}",
            )
        visible = display.history_viewport().visible_entries()
        visible_texts = [e[1] for e in visible]
        # Every item's header and topic must be present.
        for i in range(2, 6):
            self.assertIn(f"[item {i}/5] short-{i}", visible_texts)
            self.assertIn(f"short-topic-{i}", visible_texts)
        self.assertIn("[item 1/5] chatty", visible_texts)
        self.assertIn("chatty-topic", visible_texts)

    def test_scrollback_can_recover_rows_trimmed_from_live_edge_fill(self):
        module = _load_narrator_reader()
        display = self._display(module)

        _commit_item(
            display,
            module,
            header="[item 1/6] chatty",
            lines=[f"chatty-{i}" for i in range(30)],
            topic="chatty-topic",
        )
        for i in range(2, 7):
            _commit_item(
                display,
                module,
                header=f"[item {i}/6] short-{i}",
                lines=[f"short-line-{i}"],
                topic=f"short-topic-{i}",
            )

        live_edge_texts = [e[1] for e in display.history_viewport().visible_entries()]
        self.assertNotIn(
            "chatty-0",
            live_edge_texts,
            "old optional narrator lines should be allowed to drop out of the "
            "live-edge fill when newer essentials need the budget",
        )

        # Scroll back far enough to leave the curated live edge and
        # enter the older retained narrator rows that were clipped
        # from the default panel fill.
        display.scroll_history_up(10)
        scrolled_texts = [e[1] for e in display.history_viewport().visible_entries()]

        self.assertIn(
            "chatty-0",
            scrolled_texts,
            "scrollback must still be able to recover narrator rows that were "
            "trimmed from the live-edge fill",
        )

    def test_rendered_history_panel_follows_scrolled_viewport(self):
        module = _load_narrator_reader()
        display = self._display(module)

        for i in range(8):
            _commit_item(
                display,
                module,
                header=f"[item {i+1}/8] test-{i}",
                lines=[],
                topic=f"topic-{i}",
            )

        live_edge_text = _extract_plain(display.render().renderables[-1].renderable)
        display.scroll_history_up(5)
        scrolled_text = _extract_plain(display.render().renderables[-1].renderable)

        self.assertNotEqual(
            scrolled_text,
            live_edge_text,
            "render() must consult the viewport after scroll_history_up",
        )
        self.assertIn("[item 1/8] test-0", scrolled_text)
        self.assertNotIn("[item 8/8] test-7", scrolled_text)


if __name__ == "__main__":
    unittest.main()
