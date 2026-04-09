"""Contracts for narrator reader item ordering.

The viewer groups narrator history by item header. Within the active
item, the human wants the freshest committed summaries nearest the
header, with older summaries pushed downward. That keeps the most
recent reasoning in the active reading band instead of sinking to the
bottom of the panel.
"""

from __future__ import annotations

import unittest

from rich.console import Console

from scripts.narrator_reader import PaintDryDisplay


class NarratorReaderContract(unittest.TestCase):
    def _make_display(self) -> PaintDryDisplay:
        return PaintDryDisplay(
            console=Console(
                width=100,
                record=True,
                color_system="truecolor",
                force_terminal=True,
            )
        )

    def test_newest_committed_line_renders_closest_to_header(self) -> None:
        """Within one item group, fresher summaries belong above older ones.

        This preserves the tail of the model's current reasoning when the
        history compresses under a sticky item header.
        """
        display = self._make_display()

        display.on_header("[item 2/6] FR-5(b)")
        display.on_delta("first committed summary")
        display.on_commit()
        display.on_delta("second committed summary")
        display.on_commit()

        history_panel = display.render().renderables[2]
        history_lines = history_panel.renderable.plain.splitlines()

        self.assertEqual(history_lines[0], "─ [item 2/6] FR-5(b)")
        self.assertEqual(history_lines[1].strip(), "second committed summary")
        self.assertEqual(history_lines[2].strip(), "first committed summary")


if __name__ == "__main__":
    unittest.main()
