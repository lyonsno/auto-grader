from __future__ import annotations

import math
import time
import unittest
from unittest import mock

from rich.align import Align
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
    _scorebug_big_value_rows,
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

    @staticmethod
    def _styles_in_range(text: Text, start: int, end: int) -> set[str]:
        styles: set[str] = set()
        for span in text.spans:
            if not isinstance(span.style, str):
                continue
            if span.end <= start or span.start >= end:
                continue
            styles.add(span.style)
        return styles

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

    def test_thought_commit_freezes_live_line_without_persisting_history_row(self):
        display = PaintDryDisplay()
        display.on_header("[item 1/6] first")
        display.on_delta("I'm tracing the stoichiometry.")

        display.on_commit("thought")

        self.assertEqual(display.frozen_line, "I'm tracing the stoichiometry.")
        self.assertEqual(
            list(display.history),
            [("header", "[item 1/6] first", None)],
            "live thought commits should stay in the live lane and no longer persist line rows into history",
        )

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

    def test_checkpoint_renders_directly_under_header_before_thought_lines(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("line", "older line", 0))
        display.history.append(
            ("checkpoint", "Core issue: ozone drawing misses resonance.", None)
        )
        display.history.append(("line", "newer line", 1))

        entries = display._build_display_entries()
        summary = [(entry[0], entry[1]) for entry, _recent, _depth in entries]

        self.assertEqual(
            summary,
            [
                ("header", "[item 1/6] first"),
                ("checkpoint", "Core issue: ozone drawing misses resonance."),
                ("line", "newer line"),
                ("line", "older line"),
            ],
        )

    def test_checkpoint_uses_structural_mark_and_history_family_ink(self):
        import scripts.narrator_reader as module

        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(
            ("checkpoint", "Core issue: ozone drawing misses resonance.", 0)
        )
        display.history.append(("line", "I'm tracing the ozone structure.", 0))
        display.history.append(("line", "I'm weighing the resonance form.", 1))

        group = display.render()
        history_panel = group.renderables[-1]
        history_text = history_panel.renderable

        checkpoint_mark_style = self._style_for_substring(history_text, "≈")
        checkpoint_text_style = self._style_for_substring(
            history_text,
            "Core issue: ozone drawing misses resonance.",
        )
        line_style = self._style_for_substring(
            history_text,
            "I'm tracing the ozone structure.",
        )
        line_alt_style = self._style_for_substring(
            history_text,
            "I'm weighing the resonance form.",
        )
        checkpoint_rgb = self._rgb_from_hex(checkpoint_text_style.split()[-1])
        line_rgb = self._rgb_from_hex(line_style.split()[-1])
        line_alt_rgb = self._rgb_from_hex(line_alt_style.split()[-1])
        checkpoint_family_distance = min(
            sum(abs(a - b) for a, b in zip(checkpoint_rgb, line_rgb)),
            sum(abs(a - b) for a, b in zip(checkpoint_rgb, line_alt_rgb)),
        )
        legacy_steel_rgb = (172, 186, 198)
        steel_distance = sum(
            abs(a - b) for a, b in zip(checkpoint_rgb, legacy_steel_rgb)
        )

        self.assertNotEqual(
            checkpoint_mark_style,
            "grey50",
            "checkpoint mark should carry its own structural ink instead of reading like a dead placeholder gutter",
        )
        self.assertLess(
            checkpoint_family_distance,
            steel_distance,
            "checkpoint text should stay in the muted history family instead of reading like a separate steel annotation layer",
        )
        self.assertLess(
            self._hex_luminance(checkpoint_text_style),
            max(
                self._hex_luminance(line_style),
                self._hex_luminance(line_alt_style),
            ) + 50,
            "checkpoint text should stay close to the history-row value range instead of jumping to a much brighter anchored-steel tier",
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

    def test_top_panel_uses_warm_status_with_alternating_cool_and_soft_warm_live_text(self):
        cool_display = self._make_display()
        cool_display.status_line = "Tracing the stoichiometry setup."
        cool_display.frozen_line = "I'm tracing the stoichiometry."
        cool_display._frozen_line_parity = 0

        warm_display = self._make_display()
        warm_display.status_line = "Tracing the stoichiometry setup."
        warm_display.frozen_line = "I'm tracing the stoichiometry."
        warm_display._frozen_line_parity = 1

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=0.0):
            cool_group = cool_display.render()
            warm_group = warm_display.render()
        cool_live_panel = cool_group.renderables[1]
        warm_live_panel = warm_group.renderables[1]
        cool_status_text, cool_live_text = cool_live_panel.renderable.renderables
        warm_status_text, warm_live_text = warm_live_panel.renderable.renderables

        status_gutter = cool_status_text.spans[0].style
        status_rgbs = [
            self._rgb_from_hex(style)
            for style in self._content_hexes(warm_status_text)
        ]
        cool_red, cool_green, cool_blue = self._rgb_from_hex(
            self._first_content_hex(cool_live_text)
        )
        warm_red, warm_green, warm_blue = self._rgb_from_hex(
            self._first_content_hex(warm_live_text)
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
            cool_blue,
            cool_red,
            "even-parity live line should stay on the cooler structural family",
        )
        self.assertGreater(
            cool_green,
            cool_red,
            "cool live line should keep some moss/green body",
        )
        self.assertGreater(
            cool_green,
            0.75 * cool_blue,
            "cool live line should lean a bit greener than the earlier pure steel-blue pass",
        )
        self.assertGreater(
            warm_red,
            warm_blue,
            "odd-parity live line should flip into a softened warm family",
        )
        self.assertGreater(
            warm_red,
            warm_green,
            "soft-warm live line should still read red-led rather than beige/white-led",
        )
        self.assertGreater(
            warm_green,
            warm_blue,
            "soft-warm live line should still stay pastel and friendly instead of pure hot red",
        )

    def test_render_places_project_paint_dry_header_above_scorebug_when_model_known(self):
        display = self._make_display()
        display.current_model = "qwen3p5-35B-A3B"
        display.on_header("[item 3/6] 15-blue/fr-5b")

        group = display.render()

        header_panel = group.renderables[0]
        scorebug_panel = group.renderables[1]
        live_panel = group.renderables[2]
        scorebug_text = _extract_plain(scorebug_panel.renderable)
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        scorebug_text_obj = scorebug_renderable.renderables[0]
        spacer_row = scorebug_renderable.renderables[1]

        self.assertIn("CURRENT MODEL", scorebug_text)
        self.assertIn("qwen3p5-35B-A3B", scorebug_text)
        self.assertIn("ITEM", scorebug_text)
        self.assertIn("3/6", scorebug_text)
        self.assertIn("PROJECT PAINT DRY", _extract_plain(header_panel.renderable))
        self.assertIn("status + live", live_panel.title)
        self.assertEqual(
            spacer_row.plain.strip(),
            "",
            "scorebug should separate model/set/item from the big tally row with a blank spacer line",
        )
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

    def test_header_title_uses_visible_lacquer_gradient_in_rendered_surface(self):
        display = self._make_display()
        display.current_model = "qwen3p5-35B-A3B"
        display.on_header("[item 4/6] 15-blue/fr-10a")

        group = display.render()

        header_panel = group.renderables[0]
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable

        title_end = header_renderable.plain.index(" · sumi-e")
        title_styles = {
            style
            for style in self._styles_in_range(header_renderable, 0, title_end)
            if "#" in style
        }

        self.assertGreaterEqual(
            len(title_styles),
            2,
            "PROJECT PAINT DRY should carry multiple lacquer-family spans so the header accent actually reads in the smoked surface",
        )
        self.assertNotIn(
            "bold bright_white",
            self._styles_in_range(header_renderable, 0, title_end),
            "PROJECT PAINT DRY should no longer render as a single flat bright-white title",
        )

    def test_scorebug_can_render_set_and_running_tally_cells(self):
        display = self._make_display()
        display.on_session_meta(
            model="gemma-4-26b-a4b-it-bf16",
            set_label="TRICKY",
            subset_count=6,
        )
        display.on_header("[item 4/6] 15-blue/fr-10a")
        display.on_topic(
            "35s · Grader: 2/2. Prof: 2/2.",
            verdict="match",
            grader_score=2.0,
            truth_score=2.0,
            max_points=2.0,
        )
        display.on_topic(
            "53s · Grader: 0/4. Prof: 1/4.",
            verdict="undershoot",
            grader_score=0.0,
            truth_score=1.0,
            max_points=4.0,
        )
        display.on_topic(
            "82s · Grader: 3/3. Prof: 1.5/3.",
            verdict="overshoot",
            grader_score=3.0,
            truth_score=1.5,
            max_points=3.0,
        )

        group = display.render()

        header_panel = group.renderables[0]
        scorebug_panel = group.renderables[1]
        scorebug_text = _extract_plain(scorebug_panel.renderable)
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        scorebug_text_obj = scorebug_renderable.renderables[0]
        spacer_row = scorebug_renderable.renderables[1]
        tally_text_obj = scorebug_renderable.renderables[2]
        tally_value_top = scorebug_renderable.renderables[3]
        tally_value_mid = scorebug_renderable.renderables[4]
        tally_value_bottom = scorebug_renderable.renderables[5]
        expected_on_target = _scorebug_big_value_rows("2.0/9.0")
        expected_left = _scorebug_big_value_rows("1.0/1.0")
        expected_bad = _scorebug_big_value_rows("1.5/1.5")

        self.assertIn("PROJECT PAINT DRY", _extract_plain(header_panel.renderable))
        self.assertIn("CURRENT MODEL", scorebug_text)
        self.assertIn("SET", scorebug_text)
        self.assertIn("TRICKY", scorebug_text)
        self.assertIn("ITEM", scorebug_text)
        self.assertIn("4/6", scorebug_text)
        self.assertIn("ON TARGET", scorebug_text)
        self.assertIn("LEFT ON TABLE", scorebug_text)
        self.assertIn("BAD CALLS", scorebug_text)
        self.assertEqual(spacer_row.plain.strip(), "")
        self.assertIn(expected_on_target[0], tally_value_top.plain)
        self.assertIn(expected_on_target[1], tally_value_mid.plain)
        self.assertIn(expected_on_target[2], tally_value_bottom.plain)
        self.assertIn(expected_left[0], tally_value_top.plain)
        self.assertIn(expected_left[1], tally_value_mid.plain)
        self.assertIn(expected_left[2], tally_value_bottom.plain)
        self.assertIn(expected_bad[0], tally_value_top.plain)
        self.assertIn(expected_bad[1], tally_value_mid.plain)
        self.assertIn(expected_bad[2], tally_value_bottom.plain)
        self.assertIn(
            "on #",
            self._style_for_substring(scorebug_text_obj, "SET"),
            "set label should also read like a scoreboard cell",
        )
        self.assertIn(
            "on #",
            self._style_for_substring(tally_text_obj, "ON TARGET"),
            "running tally labels should render as scorebug cells, not plain text",
        )
        cell_width = len(f" {expected_on_target[0]} ")
        separator_width = 2
        self.assertEqual(
            tally_text_obj.plain.index("ON TARGET"),
            0,
            "ON TARGET label should start at the left edge of its score cell",
        )
        self.assertEqual(
            tally_text_obj.plain.index("LEFT ON TABLE"),
            cell_width + separator_width,
            "LEFT ON TABLE label should start at the left edge of its score cell",
        )
        self.assertEqual(
            tally_text_obj.plain.index("BAD CALLS"),
            (2 * cell_width) + (2 * separator_width),
            "BAD CALLS label should start at the left edge of its score cell",
        )
        on_target_styles = self._styles_in_range(
            tally_value_top,
            0,
            cell_width,
        )
        self.assertGreaterEqual(
            len(on_target_styles),
            2,
            "scorebug numerals should use at least two stroke weights/colors inside a single value cell",
        )
        on_target_top_strong = tally_value_top.spans[1].style
        on_target_bottom_style = self._style_for_substring(
            tally_value_bottom,
            expected_on_target[2].strip(),
        )
        self.assertNotEqual(
            on_target_top_strong,
            on_target_bottom_style,
            "scorebug value rows should now drift tonally across the board instead of sitting on one flat background",
        )
        left_top_style = self._style_for_substring(
            tally_value_top,
            expected_left[0].strip(),
        )
        left_bottom_style = self._style_for_substring(
            tally_value_bottom,
            expected_left[2].strip(),
        )
        bad_top_style = self._style_for_substring(
            tally_value_top,
            expected_bad[0].strip(),
        )
        bad_bottom_style = self._style_for_substring(
            tally_value_bottom,
            expected_bad[2].strip(),
        )
        self.assertNotEqual(
            left_top_style,
            left_bottom_style,
            "left-on-table board should also drift tonally instead of reading as a flat tech slab",
        )
        self.assertNotEqual(
            bad_top_style,
            bad_bottom_style,
            "bad-calls board should also drift tonally instead of reading as a flat tech slab",
        )
        self.assertEqual(
            tally_value_top.spans[2].style,
            on_target_top_strong,
            "top-row horizontal bars should stay on the strong stroke tier so the numerals read chunkier",
        )

    def test_scorebug_rendered_four_uses_tapered_open_foot_in_context(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY+",
            subset_count=12,
        )
        display.on_header("[item 4/12] 39-blue-redacted/fr-10a")
        display.on_topic(
            "23s · Grader: 4/4. Prof: 4/4.",
            verdict="match",
            grader_score=4.0,
            truth_score=4.0,
            max_points=4.0,
        )

        group = display.render()

        scorebug_panel = group.renderables[1]
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        tally_value_mid = scorebug_renderable.renderables[4]
        tally_value_bottom = scorebug_renderable.renderables[5]

        self.assertIn("╚═╣", tally_value_mid.plain)
        self.assertIn(
            "╹ ▪",
            tally_value_bottom.plain,
            "rendered 4.0 cells should taper to an open foot in the bottom row instead of dropping a full tailed stem",
        )
        self.assertNotIn(
            "║ ▪",
            tally_value_bottom.plain,
            "rendered 4.0 cells should not show the old tailed bottom stem",
        )

    def test_scorebug_big_value_rows_render_three_line_scoreboard_digits(self):
        top, middle, bottom = _scorebug_big_value_rows("2.0/9.0")

        self.assertEqual(len(top), len(middle))
        self.assertEqual(len(middle), len(bottom))
        self.assertIn("╔", top)
        self.assertRegex(middle, r"[║╠╣]")
        self.assertIn("╝", bottom)
        self.assertIn("╔═╝", middle, "the 2 glyph should have a chunky middle shoulder, not a skinny bend")
        self.assertNotIn("╱", top)

    def test_scorebug_five_glyph_reads_as_open_then_hooked_five(self):
        top, middle, bottom = _scorebug_big_value_rows("5.0")

        self.assertIn("╔═ ", top)
        self.assertIn(
            "╚═╗",
            middle,
            "the 5 glyph should hook rightward in the middle row so it reads less like a closed box",
        )
        self.assertIn("╚═╝", bottom)
        self.assertIn("▪", bottom)

    def test_scorebug_four_glyph_tapers_to_an_open_foot(self):
        top, middle, bottom = _scorebug_big_value_rows("4.0")

        self.assertIn(
            "╔ ╗",
            top,
            "the 4 glyph should carry a real top cap so it reads less like two bare fenceposts",
        )
        self.assertIn("╚═╣", middle)
        self.assertIn(
            "  ╹",
            bottom,
            "the 4 glyph should taper into an open foot on the bottom row instead of carrying a tailed stem",
        )

    def test_scorebug_zero_glyph_uses_heavier_sidewalls(self):
        top, middle, bottom = _scorebug_big_value_rows("0.0")

        self.assertIn("╔═╗", top)
        self.assertIn(
            "╠ ╣",
            middle,
            "the 0 glyph should carry heavier sidewalls in the middle row so it feels planted instead of airy",
        )
        self.assertIn("╚═╝", bottom)

    def test_scorebug_one_glyph_uses_upper_cap_and_full_stem(self):
        top, middle, bottom = _scorebug_big_value_rows("1.0")

        self.assertIn(
            "╔╗ ",
            top,
            "the 1 glyph should carry its serif/cap in the upper row instead of reading as a thin post with all the weight at the bottom",
        )
        self.assertIn(" ║ ", middle)
        self.assertIn(
            " ╹ ",
            bottom,
            "the 1 glyph should end on a short foot now that the scorebug has a floor gutter, instead of drilling all the way down with another full stem row",
        )

    def test_scorebug_skinny_family_packs_tightly_with_a_fuller_one_cap(self):
        top, middle, bottom = _scorebug_big_value_rows("17.0")

        self.assertIn(
            "╔╗ ╔═╗",
            top,
            "the 1 and 7 should read as one dense scorebug field, with the 1 carrying a fuller top cap instead of a sparse left tick",
        )
        self.assertNotIn("╔╗  ╔═╗", top)
        self.assertIn(" ║ ╔╝", middle)
        self.assertIn(" ╹ ║", bottom)

    def test_scorebug_seven_glyph_keeps_chunky_upper_hook(self):
        top, middle, bottom = _scorebug_big_value_rows("7.0")

        self.assertIn("╔═╗", top)
        self.assertIn(
            "╔╝ ",
            middle,
            "the 7 glyph should keep a chunkier upper hook instead of dropping immediately into a thin centered post",
        )
        self.assertIn("║  ", bottom)

    def test_scorebug_shows_zeroed_tally_row_before_any_topics_arrive(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY",
            subset_count=6,
        )

        group = display.render()

        scorebug_panel = group.renderables[1]
        scorebug_text = _extract_plain(scorebug_panel.renderable)
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        spacer_row = scorebug_renderable.renderables[1]
        tally_value_top = scorebug_renderable.renderables[3]
        tally_value_mid = scorebug_renderable.renderables[4]
        tally_value_bottom = scorebug_renderable.renderables[5]
        value_floor_gap = scorebug_renderable.renderables[6]
        zero_rows = _scorebug_big_value_rows("0.0/0.0")

        self.assertIn("CURRENT MODEL", scorebug_text)
        self.assertIn("SET", scorebug_text)
        self.assertIn("TRICKY", scorebug_text)
        self.assertIn("ON TARGET", scorebug_text)
        self.assertIn("LEFT ON TABLE", scorebug_text)
        self.assertIn("BAD CALLS", scorebug_text)
        self.assertEqual(spacer_row.plain.strip(), "")
        self.assertIn(zero_rows[0], tally_value_top.plain)
        self.assertIn(zero_rows[1], tally_value_mid.plain)
        self.assertIn(zero_rows[2], tally_value_bottom.plain)
        self.assertEqual(
            value_floor_gap.plain.strip(),
            "",
            "scorebug values should keep a blank gutter below the bottom numeral row so the strokes don't slam into the panel floor",
        )

    def test_scorebug_slash_is_a_quieter_middle_row_divider(self):
        top, middle, bottom = _scorebug_big_value_rows("1.0/2.0")

        self.assertNotIn("╱", top)
        self.assertIn("╱", middle)
        self.assertNotIn(
            "╱",
            bottom,
            "the slash should stay in the middle row so it reads as a divider, not a competing full-height digit stroke",
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

    def test_rejected_panel_caps_visible_drops_to_four_lines(self):
        display = self._make_display()
        for idx in range(6):
            display.on_drop("dedup", f"drop line {idx}")

        group = display.render()
        drops_panel = group.renderables[-1]
        drops_text = _extract_plain(drops_panel.renderable)

        self.assertNotIn("drop line 0", drops_text)
        self.assertNotIn("drop line 1", drops_text)
        self.assertIn("drop line 2", drops_text)
        self.assertIn("drop line 5", drops_text)

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
        self.assertLess(
            _history_tier_dim_factor(2),
            0.87,
            "the first couple thought lines should still fall off quickly",
        )
        self.assertLess(
            _history_tier_dim_factor(3),
            0.80,
            "the early fade should keep the old sharper drop before the tail eases out",
        )
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

    def test_history_groups_get_subtle_setback_with_alternating_secondary_field(self):
        import scripts.narrator_reader as module

        self.assertTrue(
            hasattr(module, "_history_entry_phase"),
            "history shimmer should expose a helper that combines header setback, within-group rake, and a subtle alternating field",
        )
        phase0 = module._history_entry_phase(0.25, 0.43, 0, 0)
        phase0_deep = module._history_entry_phase(0.25, 0.43, 0, 2)
        phase1 = module._history_entry_phase(0.25, 0.43, 1, 0)
        phase2 = module._history_entry_phase(0.25, 0.43, 2, 0)

        self.assertLess(
            abs(phase0 - 0.25),
            0.015,
            "the lead item should stay close to its anchor even with the subtle alternating field",
        )
        self.assertLess(
            phase1 - phase0,
            0.0,
            "lower item headers should sit slightly behind the one above them",
        )
        self.assertGreater(
            phase1 - phase0,
            -0.05,
            "between-group setback should stay gentle enough to read as structure rather than a hard jump",
        )
        self.assertLess(
            phase1 - phase0,
            -0.02,
            "adjacent headers should now sit a bit more visibly behind the one above",
        )
        self.assertGreater(
            phase0 - phase0_deep,
            0.0,
            "within-item rake should still exist so each block keeps a coherent local field",
        )
        self.assertLess(
            phase0 - phase0_deep,
            abs(phase1 - phase0),
            "this pass should lean more on header setback than on aggressive within-item rake",
        )
        self.assertNotAlmostEqual(
            phase2 - phase1,
            phase1 - phase0,
            places=6,
            msg="alternating groups should not all ride the exact same header setback",
        )

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

    def test_wrapped_history_line_only_privileges_first_visual_row(self) -> None:
        wrapped_text = Text()
        _apply_shimmer(
            wrapped_text,
            "ABCDEFGHIJKLMNOPQRSTUVWX",
            "line",
            0,
            wrap_width=16,
            phase_override=0.0,
        )

        first_row_style = wrapped_text.spans[0].style
        continuation_style = wrapped_text.spans[16].style

        self.assertLess(
            self._hex_luminance(continuation_style),
            self._hex_luminance(first_row_style),
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
        display._session_ended_at = 100.0

        self.assertTrue(display.should_animate(now=100.0 + 60.0))
        self.assertFalse(
            display.should_animate(
                now=100.0 + _SESSION_END_ANIMATION_LINGER_S + 0.1
            )
        )


if __name__ == "__main__":
    unittest.main()
