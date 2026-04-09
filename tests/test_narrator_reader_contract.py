from __future__ import annotations

import math
import time
import unittest
from unittest import mock

from rich.console import Console, Group
from rich.text import Text

from scripts.narrator_reader import (
    _ACTIVE_ANIMATION_FPS,
    _SESSION_END_ANIMATION_LINGER_S,
    _VISIBLE_HISTORY_ROWS,
    PaintDryDisplay,
    _LIVE_FREEZE_FADE_S,
    _LIVE_PER_CHAR_PHASE_OFFSET,
    _LIVE_UNDULATION_CYCLE_S,
    _STATUS_PER_CHAR_PHASE_OFFSET,
    _STATUS_UNDULATION_CYCLE_S,
    _apply_shimmer,
    _render_status_undulating,
    _history_tier_dim_factor,
    _message_requires_immediate_refresh,
    _undulation_hue_deg,
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
        style = style.split()[-1].lstrip("#")
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

    @staticmethod
    def _style_for_substring(text: Text, needle: str) -> str:
        start = text.plain.index(needle)
        for span in text.spans:
            if (
                isinstance(span.style, str)
                and span.start <= start < span.end
            ):
                return span.style
        raise AssertionError(f"no style span found for substring {needle!r}")

    @staticmethod
    def _content_hexes(text: Text) -> list[str]:
        return [
            span.style
            for span in text.spans
            if isinstance(span.style, str) and span.style.startswith("#")
        ]

    def test_status_commit_updates_sticky_status_without_replacing_frozen_thought(self):
        display = PaintDryDisplay()
        display.on_delta("I'm tracing the stoichiometry.")
        display.on_commit("thought")
        display.on_delta("Tracing the stoichiometry setup.", mode="status")

        display.on_commit("status")

        self.assertEqual(display.status_line, "Tracing the stoichiometry setup.")
        self.assertEqual(display.frozen_line, "I'm tracing the stoichiometry.")
        self.assertEqual(display.streaming_line, "")
        self.assertEqual(display.status_streaming_line, "")

    def test_active_animation_fps_is_doubled_for_smoother_motion(self):
        self.assertEqual(
            _ACTIVE_ANIMATION_FPS,
            24.0,
            "top-band motion should redraw at 24 FPS instead of the old 12 FPS",
        )

    def test_status_delta_types_in_status_lane_without_overwriting_live_line(self):
        display = PaintDryDisplay()
        display.on_delta("I'm tracing the stoichiometry.")
        display.on_commit("thought")

        display.on_delta("Tracing", mode="status")

        group = display.render()
        live_panel = group.renderables[1]
        panel_text = _extract_plain(live_panel.renderable)

        self.assertIn("TRACING", panel_text)
        self.assertIn("I'm tracing the stoichiometry.", panel_text)
        self.assertLess(
            panel_text.index("TRACING"),
            panel_text.index("I'm tracing the stoichiometry."),
        )

    def test_render_shows_status_above_live_thought(self):
        display = PaintDryDisplay()
        display.status_line = "Tracing the stoichiometry setup."
        display.frozen_line = "I'm tracing the stoichiometry."

        group = display.render()
        live_panel = group.renderables[1]
        panel_text = _extract_plain(live_panel.renderable)

        self.assertIn("status + live", live_panel.title)
        self.assertIn("TRACING THE STOICHIOMETRY SETUP.", panel_text)
        self.assertIn("I'm tracing the stoichiometry.", panel_text)
        self.assertLess(
            panel_text.index("TRACING THE STOICHIOMETRY SETUP."),
            panel_text.index("I'm tracing the stoichiometry."),
        )

    def test_new_header_clears_stale_frozen_line_and_shows_placeholders(self):
        display = self._make_display()
        display.frozen_line = "I'm tracing the previous item."

        display.on_header("[item 2/6] 15-blue/fr-2")

        group = display.render()
        live_panel = next(
            panel for panel in group.renderables
            if getattr(panel, "title", None) == "[grey50]status + live[/grey50]"
        )
        panel_text = _extract_plain(live_panel.renderable)

        self.assertEqual(display.frozen_line, "")
        self.assertIn("AWAITING STATUS", panel_text)
        self.assertNotIn("I'm tracing the previous item.", panel_text)

    def test_status_renders_in_all_caps_without_mutating_stored_text(self):
        display = self._make_display()
        display.status_line = "Rechecking the stoichiometry setup."

        group = display.render()
        live_panel = group.renderables[1]
        panel_text = _extract_plain(live_panel.renderable)

        self.assertEqual(display.status_line, "Rechecking the stoichiometry setup.")
        self.assertIn("RECHECKING THE STOICHIOMETRY SETUP.", panel_text)
        self.assertNotIn("Rechecking the stoichiometry setup.", panel_text)

    def test_status_rail_uses_ember_body_with_cool_and_bone_glints(self):
        status_text = Text(no_wrap=False, overflow="fold")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=0.0):
            _render_status_undulating(
                status_text,
                "RECHECKING THE STUDENT'S CALCULATION.",
                indent_width=2,
                wrap_width=80,
            )

        content_hexes = self._content_hexes(status_text)
        rgbs = [self._rgb_from_hex(style) for style in content_hexes]

        self.assertTrue(
            any(red > green > blue for red, green, blue in rgbs),
            "status rail should keep an ember-led body rather than flattening into cool-only text",
        )
        self.assertTrue(
            any(blue > green > red for red, green, blue in rgbs),
            "status rail should pick up restrained deep-blue glints inside the warm wave",
        )
        self.assertTrue(
            any(
                max(red, green, blue) >= 160
                and (max(red, green, blue) - min(red, green, blue)) <= 55
                for red, green, blue in rgbs
            ),
            "status rail should also catch pale bone highlights rather than only warm/cool saturated colors",
        )

    def test_empty_live_lane_shows_rotating_placeholder_copy(self):
        display = self._make_display()
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=0.0):
            first_panel = display.render().renderables[1]
            first_text = _extract_plain(first_panel.renderable)
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=19.0):
            later_panel = display.render().renderables[1]
            later_text = _extract_plain(later_panel.renderable)

        self.assertNotEqual(first_text, later_text)
        self.assertTrue(
            any(
                phrase in first_text
                for phrase in (
                    "thinking",
                    "review",
                    "grading",
                    "replay",
                    "chain-of-thought",
                )
            ),
            "empty live lane should use mildly witty waiting copy instead of a dead cursor",
        )

    def test_history_visual_row_budget_is_reduced_for_scorebug_strip(self):
        self.assertEqual(
            _VISIBLE_HISTORY_ROWS,
            30,
            "history budget should return to the original length while staying wrap-aware",
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
        self.assertEqual(depths[3], ("topic", "second topic", 1))
        self.assertEqual(depths[4], ("line", "second line", 2))

    def test_lines_render_newest_first_beneath_topic_within_each_header(self):
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
                ("topic", "topic line"),
                ("line", "newer line"),
                ("line", "older line"),
            ],
        )

    def test_topic_renders_directly_under_header_before_thought_lines(self):
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
                ("topic", "topic line"),
                ("line", "newer line"),
                ("line", "older line"),
            ],
        )

    def test_wrapped_history_lines_consume_visual_row_budget(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("topic", "brief topic", "match"))
        display.history.append(("line", "x" * 620, 0))
        display.history.append(("line", "newest short line", 1))

        entries = display._build_display_entries(wrap_width=20)
        summary = [(entry[0], entry[1]) for entry, _recent, _depth in entries]

        self.assertEqual(
            summary,
            [
                ("header", "[item 1/6] first"),
                ("topic", "brief topic"),
                ("line", "newest short line"),
            ],
            "a heavily wrapped older line should consume the visual-row budget and drop before pushing out newer visible context",
        )

    def test_top_panel_uses_warm_status_gutter_with_umber_status_and_cool_live_text(self):
        display = self._make_display()
        display.status_line = "Tracing the stoichiometry setup."
        display.frozen_line = "I'm tracing the stoichiometry."

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=0.0):
            group = display.render()
        live_panel = group.renderables[1]
        status_text, live_text = live_panel.renderable.renderables

        status_gutter = status_text.spans[0].style
        status_rgbs = [
            self._rgb_from_hex(style)
            for style in self._content_hexes(status_text)
        ]
        live_red, live_green, live_blue = self._rgb_from_hex(
            self._first_content_hex(live_text)
        )

        gutter_red, gutter_green, gutter_blue = self._rgb_from_hex(status_gutter)
        self.assertGreater(
            gutter_red,
            gutter_green,
            "status gutter should now join the warm status family instead of staying blue",
        )
        self.assertGreater(
            gutter_green,
            gutter_blue,
            "status gutter should read as ember/umber rather than magenta or blue",
        )
        self.assertTrue(
            any(red > green > blue for red, green, blue in status_rgbs),
            "sticky status text should still contain an ember-led body, not only cool glints",
        )
        self.assertTrue(
            any(blue > green > red for red, green, blue in status_rgbs),
            "sticky status text should be allowed to pick up restrained cool glints inside the warmer rail",
        )
        self.assertGreater(
            live_blue,
            live_red,
            "live first-person line should now shift into the cooler structural family",
        )
        self.assertGreater(
            live_green,
            live_red,
            "live first-person line should pick up some moss/green body instead of fiery red",
        )
        self.assertGreater(
            live_blue,
            live_green,
            "live first-person line should lean more steel-blue than moss-green",
        )

    def test_render_inserts_scorebug_strip_between_header_and_status_when_model_known(self):
        display = self._make_display()
        display.current_model = "qwen3p5-35B-A3B"
        display.on_header("[item 3/6] 15-blue/fr-5b")

        group = display.render()

        scorebug_panel = group.renderables[1]
        live_panel = group.renderables[2]
        scorebug_text = _extract_plain(scorebug_panel.renderable)
        scorebug_text_obj = scorebug_panel.renderable.renderable

        self.assertIn("CURRENT MODEL", scorebug_text)
        self.assertIn("qwen3p5-35B-A3B", scorebug_text)
        self.assertIn("ITEM", scorebug_text)
        self.assertIn("3/6", scorebug_text)
        self.assertIn("status + live", live_panel.title)
        self.assertIn(
            "on #",
            self._style_for_substring(scorebug_text_obj, "CURRENT MODEL"),
            "scorebug labels should now read like scoreboard capsules, not plain metadata text",
        )
        self.assertIn(
            "on #",
            self._style_for_substring(scorebug_text_obj, "ITEM"),
            "item indicator should live in its own scorebug cell",
        )

    def test_live_undulation_drifts_leftward(self):
        dt = _LIVE_PER_CHAR_PHASE_OFFSET / (2 * math.pi / _LIVE_UNDULATION_CYCLE_S)

        hue_at_char_1_now = _undulation_hue_deg(
            0.0,
            1,
            cycle_s=_LIVE_UNDULATION_CYCLE_S,
            center_deg=18,
            range_deg=22,
            per_char_phase_offset=_LIVE_PER_CHAR_PHASE_OFFSET,
            phase_offset_rad=0.0,
            direction=-1.0,
        )
        hue_at_char_0_later = _undulation_hue_deg(
            dt,
            0,
            cycle_s=_LIVE_UNDULATION_CYCLE_S,
            center_deg=18,
            range_deg=22,
            per_char_phase_offset=_LIVE_PER_CHAR_PHASE_OFFSET,
            phase_offset_rad=0.0,
            direction=-1.0,
        )

        self.assertAlmostEqual(hue_at_char_0_later, hue_at_char_1_now, places=6)

    def test_status_undulation_drifts_rightward_with_phase_offset(self):
        dt = _STATUS_PER_CHAR_PHASE_OFFSET / (2 * math.pi / _STATUS_UNDULATION_CYCLE_S)

        hue_at_char_0_now = _undulation_hue_deg(
            0.0,
            0,
            cycle_s=_STATUS_UNDULATION_CYCLE_S,
            center_deg=12,
            range_deg=14,
            per_char_phase_offset=_STATUS_PER_CHAR_PHASE_OFFSET,
            phase_offset_rad=0.85,
            direction=1.0,
        )
        hue_at_char_1_later = _undulation_hue_deg(
            dt,
            1,
            cycle_s=_STATUS_UNDULATION_CYCLE_S,
            center_deg=12,
            range_deg=14,
            per_char_phase_offset=_STATUS_PER_CHAR_PHASE_OFFSET,
            phase_offset_rad=0.85,
            direction=1.0,
        )

        self.assertAlmostEqual(hue_at_char_1_later, hue_at_char_0_now, places=6)

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
        self.assertLess(_history_tier_dim_factor(1), 0.96)
        self.assertLess(_history_tier_dim_factor(2), _history_tier_dim_factor(1))
        self.assertLess(_history_tier_dim_factor(3), _history_tier_dim_factor(2))
        self.assertLess(_history_tier_dim_factor(4), _history_tier_dim_factor(3))
        self.assertLess(_history_tier_dim_factor(5), _history_tier_dim_factor(4))
        self.assertLess(_history_tier_dim_factor(6), _history_tier_dim_factor(5))
        self.assertLess(_history_tier_dim_factor(7), _history_tier_dim_factor(6))
        self.assertLess(_history_tier_dim_factor(8), _history_tier_dim_factor(7))
        self.assertEqual(_history_tier_dim_factor(9), _history_tier_dim_factor(10))

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

    def test_faded_summary_lines_get_less_shimmer_boost(self) -> None:
        top_static = Text()
        _apply_shimmer(top_static, "A", "line", 0, phase_override=0.0)
        top_shimmer = Text()
        _apply_shimmer(top_shimmer, "A", "line", 0, phase_override=(12 / 13))

        lower_static = Text()
        _apply_shimmer(lower_static, "A", "line", 5, phase_override=0.0)
        lower_shimmer = Text()
        _apply_shimmer(lower_shimmer, "A", "line", 5, phase_override=(12 / 13))

        top_boost = self._hex_luminance(top_shimmer.spans[0].style) - self._hex_luminance(
            top_static.spans[0].style
        )
        lower_boost = self._hex_luminance(
            lower_shimmer.spans[0].style
        ) - self._hex_luminance(lower_static.spans[0].style)

        self.assertGreater(top_boost, lower_boost)

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
        display._session_ended_at = 100.0

        self.assertTrue(display.should_animate(now=100.0 + 60.0))
        self.assertFalse(
            display.should_animate(
                now=100.0 + _SESSION_END_ANIMATION_LINGER_S + 0.1
            )
        )


if __name__ == "__main__":
    unittest.main()
