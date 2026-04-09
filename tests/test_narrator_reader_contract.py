from __future__ import annotations

import time
import unittest
from unittest import mock

from rich.console import Console, Group
from rich.text import Text

from scripts.narrator_reader import (
    PaintDryDisplay,
    _LIVE_FREEZE_FADE_S,
    _apply_shimmer,
    _history_tier_dim_factor,
    _message_requires_immediate_refresh,
    _render_layer_index,
)


def _extract_plain(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_extract_plain(child) for child in renderable.renderables)
    if hasattr(renderable, "renderable"):
        return _extract_plain(renderable.renderable)
    return str(renderable)


class NarratorReaderContract(unittest.TestCase):
    @staticmethod
    def _hex_luminance(style: str) -> int:
        style = style.lstrip("#")
        red = int(style[0:2], 16)
        green = int(style[2:4], 16)
        blue = int(style[4:6], 16)
        return red + green + blue

    def _make_display(self) -> PaintDryDisplay:
        return PaintDryDisplay(
            console=Console(
                width=100,
                record=True,
                color_system="truecolor",
                force_terminal=True,
            )
        )

    @staticmethod
    def _rgb_from_hex(style: str) -> tuple[int, int, int]:
        style = style.lstrip("#")
        return (
            int(style[0:2], 16),
            int(style[2:4], 16),
            int(style[4:6], 16),
        )

    @staticmethod
    def _first_content_hex(text: Text) -> str:
        for span in text.spans:
            if (
                isinstance(span.style, str)
                and span.style.startswith("#")
                and span.start >= 2
            ):
                return span.style
        raise AssertionError("no hex content style found after cursor")

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

    def test_group_depth_resets_at_each_header(self):
        display = self._make_display()
        display.history.append(("header", "[item 2/6] second", None))
        display.history.append(("line", "second line", 0))
        display.history.append(("topic", "second topic", "match"))
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("line", "first line", 0))

        entries = display._build_display_entries()
        depths = [(entry[0], entry[1], group_depth) for entry, _recent, group_depth in entries]

        self.assertEqual(depths[0], ("header", "[item 1/6] first", 0))
        self.assertEqual(depths[1], ("line", "first line", 1))
        self.assertEqual(depths[2], ("header", "[item 2/6] second", 0))
        self.assertEqual(depths[3], ("line", "second line", 1))
        self.assertEqual(depths[4], ("topic", "second topic", 2))

    def test_lines_render_newest_first_within_each_header(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("line", "older line", 0))
        display.history.append(("line", "newer line", 1))
        display.history.append(("topic", "topic line", "match"))

        entries = display._build_display_entries()
        summary = [(entry[0], entry[1]) for entry, _recent, _depth in entries]

        self.assertEqual(
            summary,
            [
                ("header", "[item 1/6] first"),
                ("line", "newer line"),
                ("line", "older line"),
                ("topic", "topic line"),
            ],
        )

    def test_top_panel_uses_cool_status_and_warm_live_colors(self):
        display = self._make_display()
        display.status_line = "Tracing the stoichiometry setup."
        display.frozen_line = "I'm tracing the stoichiometry."

        group = display.render()
        live_panel = group.renderables[1]
        status_text, live_text = live_panel.renderable.renderables

        status_red, status_green, status_blue = self._rgb_from_hex(
            self._first_content_hex(status_text)
        )
        live_red, live_green, live_blue = self._rgb_from_hex(
            self._first_content_hex(live_text)
        )

        self.assertGreater(
            status_blue,
            status_red,
            "sticky status should read as the calmer cool rail",
        )
        self.assertGreater(
            live_red,
            live_blue,
            "live first-person line should stay on the warm ember side, not cool blue",
        )

    def test_new_header_clears_sticky_status(self):
        display = PaintDryDisplay()
        display.status_line = "Tracing the old question."

        display.on_header("[item 2/6] 15-blue/fr-2")

        self.assertEqual(display.status_line, "")

    def test_status_retry_dedup_counts_toward_dedup_stats(self):
        display = PaintDryDisplay()

        display.on_drop("dedup-status", "Rechecking the same unit conversion.")

        self.assertEqual(display.stat_dropped_dedup, 1)

    def test_header_starts_total_and_turn_timers(self):
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=100.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=107.0):
            header_text = _extract_plain(display.render().renderables[0].renderable)
        self.assertIn("total=7s", header_text)
        self.assertIn("turn=7s", header_text)

    def test_turn_timer_persists_within_item_and_resets_on_next_header(self):
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=200.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=201.0):
            display.on_delta("Tracing")
        display.on_commit("status")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=205.0):
            header_text = _extract_plain(display.render().renderables[0].renderable)
        self.assertIn("turn=5s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=206.0):
            display.on_delta("Rechecking")
        display.on_drop("dedup", "Rechecking the same unit conversion.")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=208.0):
            header_text = _extract_plain(display.render().renderables[0].renderable)
        self.assertIn("turn=8s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=209.0):
            display.on_delta("Tracing")
        display.on_rollback_live()
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=211.0):
            header_text = _extract_plain(display.render().renderables[0].renderable)
        self.assertIn("turn=11s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=212.0):
            display.on_header("[item 2/6] 15-blue/fr-2")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=215.0):
            header_text = _extract_plain(display.render().renderables[0].renderable)
        self.assertIn("total=15s", header_text)
        self.assertIn("turn=3s", header_text)

    def test_local_group_dim_factor_descends_materially_per_line(self):
        self.assertEqual(_history_tier_dim_factor(0), 1.0)
        self.assertLess(_history_tier_dim_factor(1), 0.93)
        self.assertLess(_history_tier_dim_factor(2), _history_tier_dim_factor(1))
        self.assertLess(_history_tier_dim_factor(3), _history_tier_dim_factor(2))
        self.assertLess(_history_tier_dim_factor(4), _history_tier_dim_factor(3))
        self.assertLess(_history_tier_dim_factor(5), _history_tier_dim_factor(4))
        self.assertLess(_history_tier_dim_factor(6), _history_tier_dim_factor(5))
        self.assertEqual(_history_tier_dim_factor(6), _history_tier_dim_factor(7))

    def test_only_reasoning_lines_use_group_depth_for_fade(self):
        self.assertEqual(_render_layer_index("line", 2), 2)
        self.assertEqual(
            _render_layer_index("topic", 2),
            0,
            "resolution/topic lines should stay full-strength instead of fading with the thought stack",
        )
        self.assertEqual(_render_layer_index("header", 3), 0)

    def test_stream_events_do_not_force_immediate_repaint(self):
        for msg_type in (
            "header",
            "delta",
            "commit",
            "rollback_live",
            "topic",
            "drop",
            "wrap_up_pending",
        ):
            self.assertFalse(_message_requires_immediate_refresh(msg_type))

        for msg_type in ("wrap_up", "end"):
            self.assertTrue(_message_requires_immediate_refresh(msg_type))

    def test_lower_history_tiers_render_dimmer_than_top_tier(self) -> None:
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

    def test_active_session_keeps_animating_after_live_freeze_fade(self) -> None:
        display = self._make_display()
        display.on_delta("fresh line")
        display.on_commit()

        self.assertTrue(display.should_animate())
        self.assertTrue(
            display.should_animate(
                now=display._freeze_started_at + (_LIVE_FREEZE_FADE_S / 2)
            )
        )
        self.assertTrue(
            display.should_animate(
                now=display._freeze_started_at + _LIVE_FREEZE_FADE_S + 0.1
            )
        )

    def test_session_end_stops_animation(self) -> None:
        display = self._make_display()
        display.on_delta("fresh line")
        display.on_commit()
        display.session_ended = True

        self.assertFalse(display.should_animate(now=time.monotonic()))


if __name__ == "__main__":
    unittest.main()
