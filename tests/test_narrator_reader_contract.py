from __future__ import annotations

import unittest

from rich.console import Group
from rich.text import Text

from scripts.narrator_reader import PaintDryDisplay


def _extract_plain(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_extract_plain(child) for child in renderable.renderables)
    if hasattr(renderable, "renderable"):
        return _extract_plain(renderable.renderable)
    return str(renderable)


class NarratorReaderContract(unittest.TestCase):
    def test_session_end_disables_animation(self):
        display = PaintDryDisplay()

        self.assertTrue(display.should_animate())

        display.session_ended = True

        self.assertFalse(display.should_animate())

    def test_status_commit_updates_sticky_status_without_replacing_frozen_thought(self):
        display = PaintDryDisplay()
        display.streaming_line = "I'm tracing the stoichiometry."
        display.on_commit("thought")
        display.streaming_line = "Tracing the stoichiometry setup."

        display.on_commit("status")

        self.assertEqual(display.status_line, "Tracing the stoichiometry setup.")
        self.assertEqual(display.frozen_line, "I'm tracing the stoichiometry.")
        self.assertEqual(display.streaming_line, "")

    def test_render_shows_status_above_live_thought(self):
        display = PaintDryDisplay()
        display.status_line = "Tracing the stoichiometry setup."
        display.frozen_line = "I'm tracing the stoichiometry."

        group = display.render()
        live_panel = group.renderables[1]
        panel_text = _extract_plain(live_panel.renderable)

        self.assertIn("status + live", live_panel.title)
        self.assertIn("Tracing the stoichiometry setup.", panel_text)
        self.assertIn("I'm tracing the stoichiometry.", panel_text)
        self.assertLess(
            panel_text.index("Tracing the stoichiometry setup."),
            panel_text.index("I'm tracing the stoichiometry."),
        )

    def test_new_header_clears_sticky_status(self):
        display = PaintDryDisplay()
        display.status_line = "Tracing the old question."

        display.on_header("[item 2/6] 15-blue/fr-2")

        self.assertEqual(display.status_line, "")


if __name__ == "__main__":
    unittest.main()
