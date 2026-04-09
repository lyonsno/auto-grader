from __future__ import annotations

import time
import unittest

from rich.console import Console
from rich.text import Text

from scripts.narrator_reader import (
    PaintDryDisplay,
    _LIVE_FREEZE_FADE_S,
    _apply_shimmer,
)


class NarratorReaderContract(unittest.TestCase):
    @staticmethod
    def _hex_luminance(style: str) -> int:
        style = style.lstrip("#")
        r = int(style[0:2], 16)
        g = int(style[2:4], 16)
        b = int(style[4:6], 16)
        return r + g + b

    def _make_display(self) -> PaintDryDisplay:
        return PaintDryDisplay(
            console=Console(
                width=100,
                record=True,
                color_system="truecolor",
                force_terminal=True,
            )
        )

    def test_lower_history_tiers_render_dimmer_than_top_tier(self) -> None:
        """History tiers should visibly fade as they descend.

        Pin the actual rendered base color, not just helper math:
        with shimmer phase forced away from the highlight, a deeper
        history layer should use a darker style than the top layer.
        """
        top_text = Text()
        _apply_shimmer(top_text, "ABC", "line", 0, phase_override=0.0)

        lower_text = Text()
        _apply_shimmer(lower_text, "ABC", "line", 5, phase_override=0.0)

        top_style = top_text.spans[0].style
        lower_style = lower_text.spans[0].style

        self.assertNotEqual(lower_style, top_style)
        self.assertLess(
            self._hex_luminance(lower_style),
            self._hex_luminance(top_style),
        )

    def test_reader_animates_during_live_freeze_fade_then_goes_idle(self) -> None:
        """The live-line freeze fade should animate briefly, not forever."""
        display = self._make_display()
        display.on_delta("fresh line")
        display.on_commit()

        self.assertTrue(display.should_animate())
        self.assertTrue(
            display.should_animate(
                now=display._freeze_started_at + (_LIVE_FREEZE_FADE_S / 2)
            )
        )
        self.assertFalse(
            display.should_animate(
                now=display._freeze_started_at + _LIVE_FREEZE_FADE_S + 0.1
            )
        )

    def test_session_end_stops_animation(self) -> None:
        """End-of-run footer should be static while waiting for Enter."""
        display = self._make_display()
        display.on_delta("fresh line")
        display.on_commit()
        display.session_ended = True

        self.assertFalse(display.should_animate(now=time.monotonic()))

    def test_status_commit_updates_sticky_status_without_replacing_frozen_thought(self) -> None:
        display = self._make_display()
        display.on_delta("I'm tracing the unit slip.")
        display.on_commit(mode="thought")
        display.on_delta("Rechecking the same unit slip.")
        display.on_commit(mode="status")

        self.assertEqual(display.status_line, "Rechecking the same unit slip.")
        self.assertEqual(display.frozen_line, "I'm tracing the unit slip.")

    def test_render_shows_status_above_live_thought(self) -> None:
        display = self._make_display()
        display.status_line = "Rechecking the unit conversion."
        display.frozen_line = "I'm tracing the denominator swap."

        live_panel = display.render().renderables[1]
        panel_text = live_panel.renderable.plain

        self.assertIn("status + live", live_panel.title)
        self.assertIn("Rechecking the unit conversion.", panel_text)
        self.assertIn("I'm tracing the denominator swap.", panel_text)
        self.assertLess(
            panel_text.index("Rechecking the unit conversion."),
            panel_text.index("I'm tracing the denominator swap."),
        )

    def test_new_header_clears_sticky_status(self) -> None:
        display = self._make_display()
        display.status_line = "Rechecking the unit conversion."

        display.on_header("[item 2/6] 15-blue/fr-2")

        self.assertEqual(display.status_line, "")


if __name__ == "__main__":
    unittest.main()
