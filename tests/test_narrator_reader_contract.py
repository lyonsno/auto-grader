from __future__ import annotations

import inspect
import math
import subprocess
import signal as stdlib_signal
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import fitz
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

import scripts.narrator_reader as narrator_reader
from scripts.narrator_reader import (
    _ACTIVE_ANIMATION_FPS,
    _DEFAULT_TERMINAL_CELL_ASPECT,
    _build_focus_preview_pixels,
    _build_iterm2_inline_image_sequence,
    _build_composite_band_png,
    _build_kitty_place_sequence,
    _build_kitty_transmit_chunks,
    _compute_inline_image_cell_dimensions,
    _focus_preview_budget,
    _KITTY_IMAGE_ID,
    _TEXTURE_BG_RGB,
    _trim_near_black_crop_margins,
    _trim_uniform_edge_margins,
    _render_focus_preview_pixels,
    _scaled_preview_size,
    _supports_inline_images,
    _supports_kitty_graphics,
    _SESSION_END_ANIMATION_LINGER_S,
    _VISIBLE_HISTORY_ROWS,
    FocusPreviewInlineImage,
    FocusPreviewKittyImage,
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
    _live_frame_prefix,
    _otsu_threshold,
    _texture_cell,
    _undulation_hue_deg,
    _render_layer_index,
    _reader_debug,
    HistoryScrollController,
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
    def test_stable_live_paint_prefix_clears_tail_below_shorter_frame(self):
        self.assertEqual(
            _live_frame_prefix((120, 40), (120, 40)),
            "\033[H\033[J",
            "stable repaints must erase from the top of the frame to the end "
            "of the screen so shorter refreshed frames do not leave stale UI "
            "rows behind underneath the current Paint Dry layout",
        )

    def test_resized_live_paint_prefix_keeps_full_alt_screen_clear(self):
        self.assertEqual(
            _live_frame_prefix((120, 40), (121, 40)),
            "\033[2J\033[H",
            "geometry changes still need the full clear path so the alt "
            "screen does not retain old compositor state across resizes",
        )

    def test_reader_module_imports_signal_for_sigwinch_handler(self):
        import scripts.narrator_reader as narrator_reader

        self.assertIs(
            narrator_reader.signal,
            stdlib_signal,
            "narrator_reader.main installs a SIGWINCH handler, so the module "
            "must import the stdlib signal module instead of crashing when "
            "Project Paint Dry launches",
        )

    def test_display_init_does_not_probe_terminal_stdin_for_cell_aspect(self):
        import scripts.narrator_reader as narrator_reader

        with mock.patch.object(
            narrator_reader,
            "_query_terminal_cell_aspect",
            side_effect=AssertionError("startup must not touch interactive stdin"),
        ):
            display = narrator_reader.PaintDryDisplay()

        self.assertEqual(
            display._terminal_cell_aspect,
            narrator_reader._DEFAULT_TERMINAL_CELL_ASPECT,
            "history controls share stdin with the live reader, so startup "
            "must not run terminal queries through that same input path",
        )

    def test_history_scroll_controller_binds_a_to_annotate_current_item(self):
        display = mock.Mock()
        display.annotate_current_focus_item.return_value = True
        controller = HistoryScrollController(display)

        self.assertIn("a", controller.bindings())
        self.assertTrue(controller.handle_key("a"))
        display.annotate_current_focus_item.assert_called_once_with()

    def test_history_scroll_controller_propagates_annotation_failure(self):
        display = mock.Mock()
        display.annotate_current_focus_item.return_value = False
        controller = HistoryScrollController(display)

        self.assertTrue(controller.handle_key("a"))
        display.annotate_current_focus_item.assert_called_once_with()

    def test_reader_debug_writes_to_stderr(self):
        stderr = StringIO()
        with mock.patch("sys.stderr", stderr):
            _reader_debug("scroll thread started")

        self.assertIn("scroll thread started", stderr.getvalue())

    def test_reader_debug_also_writes_to_run_local_debug_log(self):
        stderr = StringIO()
        with mock.patch("sys.stderr", stderr):
            with self.subTest("writes explicit debug log"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    debug_log = Path(tmpdir) / "reader.debug"
                    with mock.patch.dict(
                        "os.environ",
                        {"PAINT_DRY_DEBUG_LOG": str(debug_log)},
                        clear=False,
                    ):
                        _reader_debug("interactive tty ready")

                    self.assertTrue(debug_log.exists())
                    self.assertIn("interactive tty ready", debug_log.read_text())

    class _TTYBuffer(StringIO):
        def isatty(self) -> bool:
            return True

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
    def _make_png(
        *,
        width: int = 16,
        height: int = 10,
        rgb: tuple[int, int, int] = (180, 150, 120),
    ) -> bytes:
        pix = fitz.Pixmap(
            fitz.csRGB,
            width,
            height,
            bytes(rgb * (width * height)),
            False,
        )
        return pix.tobytes("png")

    @staticmethod
    def _make_bordered_png(
        *,
        width: int = 16,
        height: int = 10,
        border: int = 2,
        inner_rgb: tuple[int, int, int] = (180, 150, 120),
        border_rgb: tuple[int, int, int] = (0, 0, 0),
    ) -> bytes:
        channels = 3
        rows = bytearray(width * height * channels)
        for y in range(height):
            for x in range(width):
                rgb = (
                    border_rgb
                    if x < border or x >= width - border or y < border or y >= height - border
                    else inner_rgb
                )
                off = (y * width + x) * channels
                rows[off : off + channels] = bytes(rgb)
        pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(rows), False)
        return pix.tobytes("png")

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

    def _style_for_normalized_scorebug_substring(self, text: Text, needle: str) -> str:
        start = self._normalize_scorebug_texture(text.plain).index(needle)
        for span in text.spans:
            if (
                isinstance(span.style, str)
                and span.start <= start < span.end
            ):
                return span.style
        raise AssertionError(f"no style span found for normalized substring {needle!r}")

    @staticmethod
    def _foreground_hex(style: str) -> str | None:
        for token in style.split():
            if token.startswith("#"):
                return token
        return None

    @staticmethod
    def _background_hex(style: str) -> str | None:
        if " on " not in style:
            return None
        return style.split(" on ", 1)[1].split()[0]

    @staticmethod
    def _content_hexes(text: Text) -> list[str]:
        return [
            span.style
            for span in text.spans
            if isinstance(span.style, str) and span.style.startswith("#")
        ]

    @staticmethod
    def _normalize_scorebug_texture(text: str) -> str:
        return text.translate({
            ord("░"): ord(" "),
            ord("▒"): ord(" "),
            ord("·"): ord(" "),
            ord("┈"): ord(" "),
            ord("╎"): ord(" "),
        })

    @staticmethod
    def _scorebug_texture_count(text: str) -> int:
        return sum(ch in "░▒·┈╎" for ch in text)

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

    def test_focus_preview_panel_renders_between_live_and_history(self):
        # Half-block fallback path — test the panel-title ordering.
        display = self._make_display()
        display._inline_images_supported = False
        display._kitty_graphics_supported = False
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )

        group = display.render()
        titled_panels = [
            getattr(panel, "title", None)
            for panel in group.renderables
            if getattr(panel, "title", None) is not None
        ]

        self.assertIn("[grey50]status + live[/grey50]", titled_panels)
        self.assertIn("[grey50]focus preview · 15-blue/fr-12a[/grey50]", titled_panels)
        self.assertIn("[grey50]history[/grey50]", titled_panels)
        self.assertLess(
            titled_panels.index("[grey50]status + live[/grey50]"),
            titled_panels.index("[grey50]focus preview · 15-blue/fr-12a[/grey50]"),
        )
        self.assertLess(
            titled_panels.index("[grey50]focus preview · 15-blue/fr-12a[/grey50]"),
            titled_panels.index("[grey50]history[/grey50]"),
        )

    def test_kitty_focus_preview_renders_between_live_and_history(self):
        # Kitty image path — verify the FocusPreviewKittyImage
        # renderable lands in the render group between the
        # status+live panel and the history panel, and that
        # on_focus_preview built it (not the inline fallback or
        # the half-block fallback).
        display = self._make_display()
        display._kitty_graphics_supported = True
        display._inline_images_supported = False
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )
        # Kitty renderable must have been built.
        self.assertIsNotNone(display.focus_preview_kitty_renderable)
        self.assertIsNone(display.focus_preview_inline_renderable)
        self.assertIsNone(display.focus_preview_renderable)

        group = display.render()
        renderables = list(group.renderables)
        status_live_idx = None
        history_idx = None
        kitty_idx = None
        for i, r in enumerate(renderables):
            title = getattr(r, "title", None)
            if title == "[grey50]status + live[/grey50]":
                status_live_idx = i
            elif title == "[grey50]history[/grey50]":
                history_idx = i
            elif isinstance(r, FocusPreviewKittyImage):
                kitty_idx = i
        self.assertIsNotNone(status_live_idx, "status+live panel must be present")
        self.assertIsNotNone(history_idx, "history panel must be present")
        self.assertIsNotNone(
            kitty_idx,
            "kitty focus preview renderable must be present in render group",
        )
        self.assertLess(status_live_idx, kitty_idx)
        self.assertLess(kitty_idx, history_idx)

    def test_history_panel_reports_when_no_scroll_overflow_exists(self):
        display = self._make_display()
        display.on_header("[item 1/2] 15-blue/fr-10b (numeric, 1.0 pts)")
        display.on_topic("31s · Grader: 0/1. Prof: 1/1.", verdict="undershoot")

        group = display.render()
        history_panel = next(
            r for r in group.renderables if getattr(r, "title", None) == "[grey50]history[/grey50]"
        )

        self.assertEqual(
            history_panel.subtitle,
            "[grey35]live edge · no overflow yet[/grey35]",
            "short runs should make it explicit that history is not scrollable yet "
            "instead of leaving the operator to guess whether the controls failed",
        )

    def test_history_panel_reports_rows_back_when_scrolled(self):
        display = self._make_display()
        for i in range(24):
            display.on_header(f"[item {i+1}/10] test-{i}")
            display.on_topic(f"{i}s · topic-{i}", verdict="match")
            display.on_basis(f"basis-{i}")
            display.on_checkpoint(f"checkpoint-{i}")

        display.scroll_history_up(7)
        group = display.render()
        history_panel = next(
            r for r in group.renderables if getattr(r, "title", None) == "[grey50]history[/grey50]"
        )

        self.assertEqual(
            history_panel.subtitle,
            "[grey35]7 rows back · j/d forward · 0 latest[/grey35]",
            "once the operator scrolls off the live edge, the panel should say "
            "so explicitly instead of making the result invisible",
        )

    def test_inline_focus_preview_renders_between_live_and_history(self):
        # Inline image path — same ordering invariant as above but
        # the renderable is a FocusPreviewInlineImage rather than a
        # Panel with a title attribute. Verify positional ordering
        # within the Group.
        display = self._make_display()
        display._inline_images_supported = True
        # Disable kitty so this test exclusively exercises the
        # iTerm2 fallback path regardless of the test environment.
        display._kitty_graphics_supported = False
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )
        group = display.render()
        renderables = list(group.renderables)
        status_live_idx = None
        history_idx = None
        inline_idx = None
        for i, r in enumerate(renderables):
            title = getattr(r, "title", None)
            if title == "[grey50]status + live[/grey50]":
                status_live_idx = i
            elif title == "[grey50]history[/grey50]":
                history_idx = i
            elif isinstance(r, FocusPreviewInlineImage):
                inline_idx = i
        self.assertIsNotNone(status_live_idx, "status+live panel must be present")
        self.assertIsNotNone(history_idx, "history panel must be present")
        self.assertIsNotNone(
            inline_idx,
            "inline focus preview renderable must be present in render group",
        )
        self.assertLess(status_live_idx, inline_idx)
        self.assertLess(inline_idx, history_idx)

    def test_new_header_keeps_previous_preview_visible_in_pending_state(self):
        display = self._make_display()
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )

        display.on_header("[item 2/12] 27-blue-2023/fr-3 (balanced_equation, 4.0 pts)")

        group = display.render()
        titled_panels = [
            getattr(panel, "title", None)
            for panel in group.renderables
            if getattr(panel, "title", None) is not None
        ]

        self.assertTrue(display.focus_preview_pending)
        self.assertIn(
            "[grey50]focus preview · pending · 15-blue/fr-12a[/grey50]",
            titled_panels,
        )

    def test_new_focus_preview_clears_pending_transition_state(self):
        display = self._make_display()
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )
        display.on_header("[item 2/12] 27-blue-2023/fr-3 (balanced_equation, 4.0 pts)")

        display.on_focus_preview(
            self._make_png(rgb=(120, 130, 140)),
            label="27-blue-2023/fr-3",
            source="mock_tricky_plus",
        )

        self.assertFalse(display.focus_preview_pending)
        self.assertEqual(display.focus_preview_label, "27-blue-2023/fr-3")

    def test_pending_focus_preview_uses_character_overlay_not_just_block_noise(self):
        renderable = _render_focus_preview_pixels(
            [[(230, 230, 230) for _ in range(12)] for _ in range(8)],
            now=0.0,
            pending=True,
        )
        plain = _extract_plain(renderable)

        self.assertTrue(
            any(ch in plain for ch in "01./:"),
            "pending preview should use a dim character veil rather than only geometric block shading",
        )

    def test_focus_preview_is_rasterized_once_per_item_not_each_frame(self):
        display = self._make_display()
        # Force half-block fallback path so this test exercises the
        # pipeline it's designed to cover regardless of what terminal
        # the test runner's environment looks like to be.
        display._inline_images_supported = False
        display._kitty_graphics_supported = False
        sentinel_pixels = [[(10, 20, 30)]]

        with mock.patch(
            "scripts.narrator_reader._build_focus_preview_pixels",
            return_value=sentinel_pixels,
        ) as build_mock:
            display.on_focus_preview(
                self._make_png(),
                label="15-blue/fr-12a",
                source="mock_tricky",
            )
            display.render()
            display.render()

        self.assertEqual(build_mock.call_count, 1)

    def test_focus_preview_samples_at_double_vertical_density_for_half_blocks(self):
        display = self._make_display()
        display._inline_images_supported = False
        display._kitty_graphics_supported = False

        with mock.patch(
            "scripts.narrator_reader._build_focus_preview_pixels",
            return_value=[[(10, 20, 30)], [(10, 20, 30)]],
        ) as build_mock:
            display.on_focus_preview(
                self._make_png(width=48, height=72),
                label="15-blue/fr-12a",
                source="mock_tricky",
            )

        call = build_mock.call_args.kwargs
        budget_width, budget_height = _focus_preview_budget(
            display._console.width,
        )
        self.assertEqual(call["max_width_chars"], budget_width)
        self.assertEqual(
            call["max_height_rows"],
            2 * budget_height,
            "half-block renderer should sample at 2× the terminal row budget so each terminal row can encode a top/bottom half-pixel pair",
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

    def test_focus_preview_requires_immediate_refresh(self):
        self.assertTrue(_message_requires_immediate_refresh("focus_preview"))

    def test_focus_preview_budget_uses_most_of_terminal_width(self):
        width_chars, height_rows = _focus_preview_budget(100)

        self.assertGreaterEqual(
            width_chars,
            60,
            "the companion preview should still be comfortably readable on a 100-column terminal",
        )
        self.assertLessEqual(
            width_chars,
            70,
            "the preview should no longer sprawl across most of the terminal width now that it is a calmer companion surface",
        )
        self.assertGreaterEqual(
            height_rows,
            18,
            "preview should get enough rows that the vignette and handwriting survive downsampling",
        )

    def test_focus_preview_budget_uses_a_smaller_companion_surface_on_wide_terminals(self):
        width_chars, height_rows = _focus_preview_budget(
            140,
            source_width_px=1600,
            source_height_px=900,
        )

        self.assertGreaterEqual(
            width_chars,
            80,
            "wide terminals should still get a preview that reads clearly instead of collapsing back to a postage stamp",
        )
        self.assertLessEqual(
            width_chars,
            92,
            "the focus preview should read as a companion surface, not a giant wall of glyphs that dominates the terminal",
        )
        self.assertGreaterEqual(
            height_rows,
            20,
            "the smaller companion surface still needs enough rows to preserve the handwriting and vignette",
        )
        self.assertLessEqual(
            height_rows,
            24,
            "the preview should stay closer to the 70% scale the operator asked for instead of ballooning back upward",
        )

    def test_focus_preview_budget_scales_with_source_detail_instead_of_staying_fixed(self):
        small_width, small_height = _focus_preview_budget(
            140,
            source_width_px=240,
            source_height_px=120,
        )
        large_width, large_height = _focus_preview_budget(
            140,
            source_width_px=1600,
            source_height_px=900,
        )

        self.assertGreater(
            large_width,
            small_width,
            "high-detail crops should be allowed a denser preview raster than tiny crops on the same terminal",
        )
        self.assertGreater(
            large_height,
            small_height,
            "source-aware budgeting should buy more vertical detail too, not only horizontal width",
        )

    def test_focus_preview_steady_state_uses_half_block_image_cells(self):
        # 16 sampled rows = 8 terminal rows when half-blocks pair them up.
        pixels = []
        for row in range(16):
            rows: list[tuple[int, int, int]] = []
            for col in range(24):
                value = 42 + ((row * 17 + col * 11) % 180)
                rows.append((value, value - 6, value - 10))
            pixels.append(rows)

        renderable = _render_focus_preview_pixels(
            pixels,
            now=0.0,
            pending=False,
        )
        plain = _extract_plain(renderable)

        # Steady state must be a static image surface, not animated glyph
        # texture: only half-block characters (and spaces for all-paper
        # regions) are allowed.
        self.assertTrue(
            set(plain.replace("\n", "")) <= {" ", "\u2580"},
            "steady-state previews should render as half-block image cells, not as visible texture glyphs",
        )
        self.assertEqual(
            len(plain.splitlines()),
            8,
            "half-block renderer should collapse each pair of sampled image rows into one terminal row",
        )

    def test_focus_preview_steady_state_emits_foreground_and_background_per_cell(self):
        # A vertical 2-column ink bar on paper should produce cells that
        # carry *both* a foreground (top half-pixel) and a background
        # (bottom half-pixel) color — that's what half-blocks are for.
        pixels = []
        for row in range(16):
            rows: list[tuple[int, int, int]] = []
            for col in range(24):
                if col in {12, 13}:
                    rows.append((30, 30, 30))
                else:
                    rows.append((230, 225, 215))
            pixels.append(rows)

        renderable = _render_focus_preview_pixels(
            pixels,
            now=0.0,
            pending=False,
        )
        styles_with_both = [
            span.style
            for row in renderable.renderables
            for span in row.spans
            if isinstance(span.style, str) and " on " in span.style
        ]
        self.assertTrue(
            styles_with_both,
            "half-block cells should carry fg+bg style pairs representing the top and bottom sampled rows",
        )

    def test_focus_preview_steady_state_keeps_paper_in_a_visible_parchment_family(self):
        pixels = [[(228, 218, 204) for _ in range(24)] for _ in range(16)]

        renderable = _render_focus_preview_pixels(
            pixels,
            now=0.0,
            pending=False,
        )
        style_pairs = [
            span.style
            for row in renderable.renderables
            for span in row.spans
            if isinstance(span.style, str) and " on " in span.style
        ]
        self.assertTrue(style_pairs, "steady-state paper rows should emit styled image cells")
        fg_hex, bg_hex = style_pairs[0].split(" on ")
        fg = self._rgb_from_hex(fg_hex)
        bg = self._rgb_from_hex(bg_hex)

        for rgb in (fg, bg):
            self.assertGreater(
                rgb[0],
                rgb[1],
                "paper tone should stay red-led enough to read as parchment, not flat gray",
            )
            self.assertGreater(
                rgb[1],
                rgb[2],
                "paper tone should stay warm through green>blue ordering, not collapse into neutral",
            )
            self.assertGreaterEqual(
                rgb[0] - rgb[2],
                28,
                "paper tone should remain visibly sepia/warm once it hits the steady-state renderer",
            )

    def test_focus_preview_dark_strokes_produce_spatial_variation(self):
        pixels = []
        for row in range(16):
            rows: list[tuple[int, int, int]] = []
            for col in range(24):
                if col in {12, 13} and 2 <= row <= 13:
                    rows.append((70, 72, 76))
                elif col < 12:
                    rows.append((220, 214, 206))
                else:
                    rows.append((190, 184, 176))
            pixels.append(rows)

        renderable = _render_focus_preview_pixels(
            pixels,
            now=0.0,
            pending=False,
        )
        span_styles = [
            span.style
            for row in renderable.renderables
            for span in row.spans
            if isinstance(span.style, str) and " on " in span.style
        ]
        self.assertTrue(
            span_styles,
            "steady-state image cells should carry foreground/background styles so adjacent pixel rows survive as an image",
        )
        # Binary-threshold renderer: a dark stroke against lighter paper must
        # produce AT LEAST two distinct cell styles (the ink cells and the
        # paper cells). Anything less means the stroke was swallowed by
        # thresholding and the renderer is producing a uniform slab again,
        # which would be a regression to the old mush behavior.
        self.assertGreaterEqual(
            len(set(span_styles)),
            2,
            "a dark stroke against lighter paper must produce spatial variation in output cells (ink cells AND paper cells), not a single uniform slab",
        )

    def test_focus_preview_sampling_preserves_thin_dark_strokes(self):
        width = 96
        height = 96
        background = (235, 230, 222)
        ink = (35, 35, 35)
        buf = bytearray(background * (width * height))
        for y in range(height):
            for x in range(47, 49):
                offset = (y * width + x) * 3
                buf[offset : offset + 3] = bytes(ink)

        pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(buf), False)
        pixels = _build_focus_preview_pixels(
            pix.tobytes("png"),
            max_width_chars=12,
            max_height_rows=12,
        )
        center_luminances = [
            ((0.299 * row[len(row) // 2][0]) + (0.587 * row[len(row) // 2][1]) + (0.114 * row[len(row) // 2][2])) / 255.0
            for row in pixels
        ]

        self.assertLess(
            min(center_luminances),
            0.45,
            "downsampling should preserve a materially darker center stripe instead of averaging thin ink strokes back into the paper",
        )

    # ------------------------------------------------------------------
    # Inline image path (iTerm2 protocol, the primary focus-preview
    # renderer on image-capable terminals like WezTerm). The half-block
    # renderer above is the fallback for terminals that don't speak an
    # image protocol. These tests pin the protocol format and the Rich
    # Renderable contract without actually rendering to a real terminal.
    # ------------------------------------------------------------------

    def test_build_iterm2_inline_image_sequence_has_correct_envelope(self):
        png = b"\x89PNG\r\n\x1a\nfake png bytes for test"
        seq = _build_iterm2_inline_image_sequence(
            png, cell_width=40, cell_height=18
        )
        # Envelope: ESC ] 1337 ; File = <args> : <base64> BEL
        self.assertTrue(
            seq.startswith("\x1b]1337;File="),
            "must start with the iTerm2 OSC 1337 File= prefix",
        )
        self.assertTrue(
            seq.endswith("\x07"),
            "must terminate with BEL",
        )
        self.assertIn("inline=1", seq, "must request inline rendering")
        self.assertIn("width=40", seq, "must declare requested cell width")
        self.assertIn("height=18", seq, "must declare requested cell height")
        self.assertIn(
            "preserveAspectRatio=1",
            seq,
            "must preserve aspect ratio so images don't stretch",
        )
        # The base64 of the PNG payload must actually appear in the
        # sequence between the ':' and the trailing BEL.
        import base64 as _b64
        encoded = _b64.b64encode(png).decode("ascii")
        self.assertIn(encoded, seq)

    def test_compute_inline_image_cell_dimensions_preserves_aspect(self):
        # Wide crop: 900x300 px → 3:1 aspect. At cell_height=18 and
        # terminal cell aspect in the 2.0-3.0 range (varies per
        # terminal font), the cell_width should be roughly
        # 18 * 3 * tca ≈ 108-162 cells. The exact value tracks
        # the queried-or-fallback terminal cell aspect, which is
        # tuned per deployment.
        cw, ch = _compute_inline_image_cell_dimensions(
            900, 300, max_cell_height=18, max_cell_width=200
        )
        self.assertEqual(ch, 18)
        self.assertGreaterEqual(cw, 100)
        self.assertLessEqual(cw, 170)

    def test_compute_inline_image_cell_dimensions_uses_default_fallback_terminal_cell_aspect(self):
        self.assertAlmostEqual(
            _DEFAULT_TERMINAL_CELL_ASPECT,
            2.1,
            places=2,
            msg="focus preview sizing should keep the 2.1 fallback for cases where the runtime terminal query is unavailable",
        )
        cw, ch = _compute_inline_image_cell_dimensions(
            900, 300, max_cell_height=18, max_cell_width=200
        )
        self.assertEqual((cw, ch), (113, 18))

    def test_compute_inline_image_cell_dimensions_clamps_to_max_width(self):
        # A very wide crop would want more cells than max allows.
        # Output must clamp to max_cell_width and shrink cell_height
        # proportionally so the aspect ratio doesn't get distorted.
        cw, ch = _compute_inline_image_cell_dimensions(
            4000, 200, max_cell_height=30, max_cell_width=100
        )
        self.assertLessEqual(cw, 100)
        # Aspect preserved: ch should have shrunk from 30 to keep
        # the ratio roughly 4000:200 = 20:1 (accounting for cell 2:1).
        self.assertLess(ch, 30)

    def test_compute_inline_image_cell_dimensions_returns_positive_integers(self):
        cw, ch = _compute_inline_image_cell_dimensions(
            1, 1, max_cell_height=18, max_cell_width=100
        )
        self.assertIsInstance(cw, int)
        self.assertIsInstance(ch, int)
        self.assertGreater(cw, 0)
        self.assertGreater(ch, 0)

    def test_supports_inline_images_recognizes_wezterm(self):
        self.assertTrue(_supports_inline_images("WezTerm"))

    def test_supports_inline_images_recognizes_iterm2(self):
        self.assertTrue(_supports_inline_images("iTerm.app"))

    def test_supports_inline_images_rejects_plain_xterm(self):
        self.assertFalse(_supports_inline_images("xterm"))
        self.assertFalse(_supports_inline_images("Apple_Terminal"))
        self.assertFalse(_supports_inline_images(None))
        self.assertFalse(_supports_inline_images(""))

    def test_focus_preview_inline_image_escape_sequence_survives_panel_line_fitting(self):
        # Regression guard against the "Rich truncates the escape
        # sequence mid-base64 because it treats the segment as
        # visible text of length == character count" bug.
        #
        # The iTerm2 inline image protocol encodes the full PNG as
        # base64 inside the escape sequence. For a real-sized crop
        # PNG that's tens of thousands of characters. If Rich's
        # Segment cell_length is computed from the text's string
        # length (instead of being forced to 0 via the control
        # marker), the panel's line-fitting pass will wrap or
        # truncate the escape sequence to the panel's cell width
        # (~70-140 cells), destroying the base64 payload and
        # preventing the terminal from ever seeing a complete
        # File= sequence. The fix: mark the escape-sequence
        # Segment as a control segment so its cell_length is 0.
        # This test renders through render_lines (which is what
        # Rich Live uses) and verifies the FULL base64 payload
        # survives into exactly one line segment.
        from rich.console import Console
        from rich.panel import Panel

        # Build a reasonably-large fake PNG so the base64 payload
        # is definitely longer than any reasonable panel width.
        fake_png = b"\x89PNG\r\n\x1a\n" + b"x" * 4096
        renderable = FocusPreviewInlineImage(
            png_bytes=fake_png, cell_width=70, cell_height=18
        )
        # Wrap in a Panel like the real render path does — that's
        # where the line-fitting truncation happens, not at the
        # bare renderable level.
        panel = Panel(renderable, padding=(0, 1))
        console = Console(
            width=100,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )
        lines = console.render_lines(panel, console.options, pad=True)

        # Find the line containing the escape sequence start.
        escape_start = "\x1b]1337;File="
        found_segments = []
        for line in lines:
            for seg in line:
                if seg.text and escape_start in seg.text:
                    found_segments.append(seg)
        self.assertEqual(
            len(found_segments),
            1,
            "exactly one segment should carry the full escape sequence",
        )
        seg = found_segments[0]
        # The full escape sequence must be intact: starts with the
        # OSC prefix, ends with BEL (\x07), contains the complete
        # base64 payload.
        self.assertTrue(seg.text.startswith(escape_start))
        self.assertTrue(seg.text.endswith("\x07"))
        import base64 as _b64
        expected_b64 = _b64.b64encode(fake_png).decode("ascii")
        self.assertIn(
            expected_b64,
            seg.text,
            "the full base64 payload must survive line-fitting without truncation",
        )
        # And the segment must have zero cell_length so Rich's
        # layout engine doesn't try to wrap it.
        self.assertEqual(
            seg.cell_length,
            0,
            "the escape-sequence segment must be zero cell-width so Rich "
            "doesn't truncate it to fit the panel's visible width",
        )

    def test_focus_preview_inline_image_emits_escape_sequence_every_render(self):
        # Rich's Live.LiveRender.position_cursor() emits
        # ERASE_IN_LINE (CSI 2K) on every row of the previous
        # frame before each refresh, which clears every cell we
        # drew the image into. So we MUST re-emit the iTerm2
        # escape sequence on every __rich_console__ call or the
        # image disappears after the first frame. This test pins
        # that requirement as an invariant so we don't
        # accidentally optimize it away again (I already did once
        # and shipped an empty container to the operator).
        from rich.console import Console

        png = b"\x89PNG\r\n\x1a\nfake png for test"
        renderable = FocusPreviewInlineImage(
            png_bytes=png, cell_width=60, cell_height=18, title="test"
        )
        console = Console(
            width=120,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )

        with console.capture() as capture:
            console.print(renderable)
        first = capture.get()

        with console.capture() as capture:
            console.print(renderable)
        second = capture.get()

        osc = "\x1b]1337;File="
        self.assertEqual(
            first.count(osc),
            1,
            "first render must emit the iTerm2 escape sequence exactly once",
        )
        self.assertEqual(
            second.count(osc),
            1,
            "every subsequent render must also emit the escape sequence; "
            "Rich Live clears the image cells between frames so we need to "
            "repaint on each refresh",
        )

    # ------------------------------------------------------------------
    # Kitty graphics protocol path. This is the durable fix for the
    # flicker problem: transmit the PNG once to the terminal with a
    # numeric image ID, then reference it by ID on every subsequent
    # frame via tiny `a=p` place commands. WezTerm and Kitty both
    # support this protocol. Unlike the iTerm2 path, place-by-ID
    # does not re-parse the PNG payload, so emitting it every frame
    # at 24 FPS costs nothing.
    # ------------------------------------------------------------------

    def test_supports_kitty_graphics_recognizes_wezterm(self):
        self.assertTrue(_supports_kitty_graphics("WezTerm"))

    def test_supports_kitty_graphics_recognizes_kitty(self):
        self.assertTrue(_supports_kitty_graphics("kitty"))

    def test_supports_kitty_graphics_rejects_plain_terminals(self):
        self.assertFalse(_supports_kitty_graphics("xterm"))
        self.assertFalse(_supports_kitty_graphics("Apple_Terminal"))
        self.assertFalse(_supports_kitty_graphics(None))
        self.assertFalse(_supports_kitty_graphics(""))

    def test_build_kitty_transmit_chunks_wraps_in_apc_envelope(self):
        png = b"\x89PNG\r\n\x1a\n" + b"x" * 32
        chunks = _build_kitty_transmit_chunks(png, image_id=1)
        self.assertGreaterEqual(len(chunks), 1)
        # Every chunk is wrapped in ESC_G ... ESC\
        for chunk in chunks:
            self.assertTrue(
                chunk.startswith("\x1b_G"),
                f"chunk must start with APC introducer (ESC_G), got {chunk[:10]!r}",
            )
            self.assertTrue(
                chunk.endswith("\x1b\\"),
                f"chunk must end with string terminator (ESC\\), got {chunk[-5:]!r}",
            )

    def test_build_kitty_transmit_chunks_first_has_control_keys_no_action(self):
        # First chunk carries the full control string: f=100 (PNG),
        # i=<id>, t=d (direct transmission), m=<flag>. No `a=` key
        # so the image is transmitted without immediate display —
        # the caller will use `a=p` later to place by ID.
        png = b"\x89PNG\r\n\x1a\n" + b"x" * 32
        chunks = _build_kitty_transmit_chunks(png, image_id=7)
        first = chunks[0]
        # Control string sits between ESC_G and the semicolon that
        # precedes the payload.
        self.assertIn(";", first)
        header = first.split(";", 1)[0]
        self.assertTrue(header.startswith("\x1b_G"))
        control = header[3:]
        self.assertIn("f=100", control)
        self.assertIn("i=7", control)
        self.assertIn("t=d", control)
        # Must NOT contain an action key — transmit-only (cache but
        # do not display). If this test fails because a future edit
        # adds `a=T` or `a=t`, that's a behavior change that would
        # cause the image to paint at the wrong cursor position
        # (wherever on_focus_preview happens to fire relative to
        # Rich's current cursor).
        self.assertNotIn("a=", control)

    def test_build_kitty_transmit_chunks_m_flag_transitions_from_1_to_0(self):
        # Multi-chunk payload: all chunks except the last must have
        # m=1 (more data coming), last chunk must have m=0. Required
        # by the Kitty protocol for the terminal to know when the
        # upload is complete.
        png = b"\x89PNG\r\n\x1a\n" + b"x" * (4096 * 3)  # big enough to need multiple chunks
        chunks = _build_kitty_transmit_chunks(png, image_id=1)
        self.assertGreater(len(chunks), 1, "test payload should produce multiple chunks")
        for chunk in chunks[:-1]:
            # Each non-last chunk has m=1 in its control string.
            header = chunk.split(";", 1)[0]
            self.assertIn("m=1", header)
        last_header = chunks[-1].split(";", 1)[0]
        self.assertIn("m=0", last_header)

    def test_build_kitty_transmit_chunks_payload_sizes_are_multiples_of_four(self):
        # The Kitty protocol requires all chunks except the last to
        # have payload size that is a multiple of 4 (base64 alignment).
        # The last chunk may be any size.
        png = b"\x89PNG\r\n\x1a\n" + b"x" * (4096 * 3)
        chunks = _build_kitty_transmit_chunks(png, image_id=1)
        for chunk in chunks[:-1]:
            _header, payload_plus_terminator = chunk.split(";", 1)
            # Strip the trailing ESC\
            payload = payload_plus_terminator[: -len("\x1b\\")]
            self.assertEqual(
                len(payload) % 4,
                0,
                f"non-last chunk payload length {len(payload)} must be divisible by 4",
            )

    def test_build_kitty_transmit_chunks_concatenated_payload_is_full_base64(self):
        import base64 as _b64
        png = b"\x89PNG\r\n\x1a\n" + b"arbitrary bytes here for test"
        expected_b64 = _b64.b64encode(png).decode("ascii")
        chunks = _build_kitty_transmit_chunks(png, image_id=1)
        concat = ""
        for chunk in chunks:
            _header, payload_plus_terminator = chunk.split(";", 1)
            concat += payload_plus_terminator[: -len("\x1b\\")]
        self.assertEqual(
            concat,
            expected_b64,
            "concatenated chunk payloads must equal the full base64 encoding of the PNG",
        )

    def test_build_kitty_place_sequence_format(self):
        seq = _build_kitty_place_sequence(image_id=1, cell_width=80, cell_height=20)
        # APC envelope
        self.assertTrue(seq.startswith("\x1b_G"))
        self.assertTrue(seq.endswith("\x1b\\"))
        # Must contain action=place, the ID, and both cell dimensions
        self.assertIn("a=p", seq)
        self.assertIn("i=1", seq)
        self.assertIn("c=80", seq)
        self.assertIn("r=20", seq)
        # Must include C=1 to suppress cursor movement after the
        # place. Without this flag, Kitty's default behavior is to
        # move the cursor right by c and down by r after placing,
        # which scrambles all subsequent output from the renderable
        # (right border, subsequent row borders, bottom border)
        # because Rich's layout accounting doesn't know the cursor
        # just jumped. Regression guard: if someone removes C=1,
        # the frame collapses visually and the image is surrounded
        # by a vertical column of border characters painted below
        # it instead of around it.
        self.assertIn(
            "C=1",
            seq,
            "place sequence must set C=1 to suppress post-place cursor movement",
        )
        # Place sequences have no payload — the semicolon and
        # everything after it is just the terminator. So the
        # sequence body (between ESC_G and ESC\) should contain
        # only control keys, no ';' introducing a payload.
        body = seq[len("\x1b_G") : -len("\x1b\\")]
        self.assertNotIn(
            ";",
            body,
            f"place sequence must not have a payload section, got body={body!r}",
        )

    def test_focus_preview_kitty_image_renderable_yields_only_place(self):
        # FocusPreviewKittyImage does NOT carry the PNG data. The
        # composite is transmitted by on_focus_preview directly to
        # stdout before the next Rich frame. The renderable's job is
        # only to emit the tiny place-by-ID command at the correct
        # cursor position on every Rich refresh, at essentially zero
        # cost (~30 bytes per frame).
        from rich.console import Console

        renderable = FocusPreviewKittyImage(
            image_id=1,
            band_cell_width=120,
            band_cell_height=22,
            title="test",
        )
        console = Console(
            width=120,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )
        with console.capture() as capture:
            console.print(renderable)
        output = capture.get()

        # The output must contain the APC place sequence.
        self.assertIn("\x1b_Ga=p", output)
        # It must NOT contain any transmit sequence (no f=100, no
        # payload data). If this test fails it means the renderable
        # is re-emitting the PNG on every frame which is the
        # flicker bug we're trying to avoid.
        self.assertNotIn("f=100", output)

    def test_focus_preview_kitty_image_places_on_every_frame(self):
        # The Kitty place command (a=p) must fire on every frame.
        # Rich's Live erases each line (CSI 2K) between frames,
        # wiping the image pixels. Without a fresh a=p the image
        # disappears after one frame. The placement is ~30 bytes
        # and references the already-cached image — no PNG data.
        from rich.console import Console

        renderable = FocusPreviewKittyImage(
            image_id=1,
            band_cell_width=120,
            band_cell_height=22,
            title="test",
        )
        console = Console(
            width=120,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )
        with console.capture() as capture:
            console.print(renderable)
        first = capture.get()
        with console.capture() as capture:
            console.print(renderable)
        second = capture.get()
        self.assertEqual(
            first.count("\x1b_Ga=p"), 1,
            "first frame must emit placement")
        self.assertEqual(
            second.count("\x1b_Ga=p"),
            1,
            "every frame must re-place — Rich's per-line erase wipes "
            "the image pixels between frames",
        )

    def test_rich_live_erases_preview_rows_before_kitty_replaces_image(self):
        # Characterization test: Rich Live erases previous rows (CSI 2K)
        # before each refresh. The renderable must now explicitly blank
        # the band rows before the deferred a=p placement so transparent
        # pixels reveal the terminal background rather than stale history
        # text from previous frames.
        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=80,
                force_terminal=True,
                color_system="truecolor",
            )
            renderable = FocusPreviewKittyImage(
                image_id=1,
                band_cell_width=20,
                band_cell_height=4,
                title="test",
                crop_png_bytes=b"raw",
                image_pixel_width=100,
                image_pixel_height=100,
            )
            with Live(console=console, auto_refresh=False, screen=False) as live:
                live.update(renderable, refresh=True)
                buf.seek(0)
                buf.truncate(0)
                live.update(renderable, refresh=True)
                second = buf.getvalue()

        self.assertEqual(
            second.count("\x1b_Ga=p"),
            1,
            "second frame must place exactly once",
        )
        # The a=p must come AFTER the explicit blank-row clear so
        # transparent pixels in the composite reveal clean background
        # instead of stale text that happened to be under the band.
        last_blank = second.rfind(" " * 20)
        place_pos = second.index("\x1b_Ga=p")
        self.assertGreater(
            place_pos,
            last_blank,
            "a=p placement must fire after the band rows are blanked so "
            "transparent pixels don't reveal stale text underneath",
        )

    def test_suppress_live_erase_eliminates_per_row_csi2k(self):
        # The animation loop manages alt-screen and cursor-home itself.
        # Rich's Live.position_cursor() emits CSI 2K per row which
        # destroys Kitty image pixels between frames, causing visible
        # flicker.  suppress_live_erase() must neuter that so steady-
        # state frames contain zero CSI 2K sequences.
        from scripts.narrator_reader import suppress_live_erase

        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=80,
                force_terminal=True,
                color_system="truecolor",
            )
            renderable = FocusPreviewKittyImage(
                image_id=1,
                band_cell_width=20,
                band_cell_height=4,
                title="test",
                crop_png_bytes=b"raw",
                image_pixel_width=100,
                image_pixel_height=100,
            )
            with Live(console=console, auto_refresh=False, screen=False) as live:
                suppress_live_erase(live)
                live.update(renderable, refresh=True)
                buf.seek(0)
                buf.truncate(0)
                live.update(renderable, refresh=True)
                second = buf.getvalue()

        self.assertEqual(
            second.count("\x1b[2K"),
            0,
            "after suppress_live_erase(), Rich must not emit CSI 2K — "
            "the animation loop handles positioning and Kitty pixels "
            "must survive between frames",
        )
        self.assertIn(
            "\x1b_Ga=p",
            second,
            "Kitty placement must still appear in the suppressed frame",
        )
    def test_retransmit_kitty_image_updates_placement_dimensions(self):
        # On resize, retransmit_kitty_image rebuilds the composite at the
        # new console width and updates the renderable's placement
        # dimensions so the next a=p uses the correct cell footprint.
        import fitz
        from scripts.narrator_reader import (
            PaintDryDisplay,
            _KITTY_IMAGE_ID,
        )

        # TERM must be non-dumb so Rich respects the explicit width=
        # parameter; without it console.size.width falls back to 80
        # regardless of the width= kwarg.
        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=120,
                force_terminal=True,
                color_system="truecolor",
            )
            display = PaintDryDisplay(console=console)
            display._kitty_graphics_supported = True

            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 300), 1)
            pix.clear_with(128)
            valid_png = pix.tobytes("png")

            display.on_focus_preview(valid_png, label="test", source="cache")
            rend = display.focus_preview_kitty_renderable
            if rend is None:
                self.skipTest("Kitty path not taken (no renderable created)")

            original_image_id = rend._image_id
            original_width = rend._band_cell_width
            self.assertEqual(original_width, 120,
                "initial band width must match console width")

            # Simulate resize to narrower terminal.
            console._width = 80
            buf.seek(0)
            buf.truncate(0)
            display.retransmit_kitty_image()

        self.assertEqual(
            rend._band_cell_width,
            80,
            "retransmit must update _band_cell_width to the new console width",
        )
        self.assertGreater(
            rend._image_id,
            original_image_id,
            "resize retransmit should allocate a fresh Kitty image ID so "
            "transparent regions don't rely on in-place overwrite of the "
            "previous cached image",
        )
        transmitted = buf.getvalue()
        self.assertIn(
            "f=100",
            transmitted,
            "retransmit must upload new image data (PNG format marker)",
        )

    def test_successive_focus_previews_allocate_fresh_kitty_image_ids(self):
        import fitz
        from scripts.narrator_reader import PaintDryDisplay

        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=120,
                force_terminal=True,
                color_system="truecolor",
            )
            display = PaintDryDisplay(console=console)
            display._kitty_graphics_supported = True

            first_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 300), 1)
            first_pix.clear_with(128)
            second_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 400), 1)
            second_pix.clear_with(192)

            display.on_focus_preview(first_pix.tobytes("png"), label="first", source="cache")
            first_rend = display.focus_preview_kitty_renderable
            if first_rend is None:
                self.skipTest("Kitty path not taken")
            first_id = first_rend._image_id

            display.on_focus_preview(second_pix.tobytes("png"), label="second", source="cache")
            second_rend = display.focus_preview_kitty_renderable
            if second_rend is None:
                self.skipTest("Kitty path not taken on second preview")

        self.assertGreater(
            second_rend._image_id,
            first_id,
            "each new focus preview should mint a fresh Kitty image ID "
            "instead of reusing the previous cached surface",
        )

    def test_resize_placement_matches_rebuilt_dimensions(self):
        # End-to-end resize contract: after retransmit_kitty_image at a
        # new width, the a=p placement emitted by __rich_console__ must
        # use the new dimensions, not the old ones.
        import re
        import fitz
        from scripts.narrator_reader import (
            PaintDryDisplay,
            _KITTY_IMAGE_ID,
        )

        # TERM must be non-dumb so Rich respects the explicit width=
        # parameter; without it console.size.width falls back to 80
        # regardless of the width= kwarg.
        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=120,
                force_terminal=True,
                color_system="truecolor",
            )
            display = PaintDryDisplay(console=console)
            display._kitty_graphics_supported = True

            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 300), 1)
            pix.clear_with(128)
            valid_png = pix.tobytes("png")

            display.on_focus_preview(valid_png, label="test", source="cache")
            rend = display.focus_preview_kitty_renderable
            if rend is None:
                self.skipTest("Kitty path not taken")

            # Resize to 80 columns.
            console._width = 80
            display.retransmit_kitty_image()

            # Render the placement as __rich_console__ would.
            options = console.options
            segments = list(rend.__rich_console__(console, options))
            place_text = "".join(seg.text for seg in segments)

        match = re.search(r"c=(\d+),r=(\d+)", place_text)
        self.assertIsNotNone(match, "a=p must contain c= and r= parameters")
        placed_width = int(match.group(1))
        self.assertEqual(
            placed_width,
            80,
            "a=p placement width must match the post-resize console width",
        )

    def test_retransmit_kitty_image_falls_back_when_rebuild_fails(self):
        # If the composite rebuild fails during resize, we must not
        # leave the operator stuck with a stale stretched Kitty band.
        # The display should degrade to the non-Kitty preview path so
        # a real preview remains visible.
        import fitz
        from scripts.narrator_reader import PaintDryDisplay

        with mock.patch.dict("os.environ", {"TERM": "xterm-256color"}):
            buf = self._TTYBuffer()
            console = Console(
                file=buf,
                width=120,
                force_terminal=True,
                color_system="truecolor",
            )
            display = PaintDryDisplay(console=console)
            display._kitty_graphics_supported = True
            display._inline_images_supported = False

            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 300), 1)
            pix.clear_with(128)
            valid_png = pix.tobytes("png")

            display.on_focus_preview(valid_png, label="test", source="cache")
            self.assertIsNotNone(
                display.focus_preview_kitty_renderable,
                "test precondition: Kitty path should be active before resize",
            )

            with mock.patch(
                "scripts.narrator_reader._build_composite_band_png",
                side_effect=RuntimeError("boom"),
            ):
                display.retransmit_kitty_image()

        self.assertFalse(
            display._kitty_graphics_supported,
            "failed Kitty resize rebuild should disable Kitty mode for the session",
        )
        self.assertIsNone(
            display.focus_preview_kitty_renderable,
            "failed Kitty resize rebuild must clear the stale Kitty renderable",
        )
        self.assertIsNotNone(
            display.focus_preview_renderable,
            "failed Kitty resize rebuild must fall back to a non-Kitty preview renderable",
        )
        self.assertFalse(
            display.focus_preview_pending,
            "fallback preview should be immediately renderable after recovery",
        )

    def test_focus_preview_kitty_composite_emits_no_texture_segments(self):
        # The precomposed Kitty band includes the ornate texture/border
        # baked into the composite image. The renderable must therefore
        # emit NO styled text segments for texture or border — only the
        # Kitty place command, cursor-forward escapes for vertical
        # height, and newlines. Any styled text segment means the band
        # is still being drawn as Rich text, which defeats the
        # precomposition and keeps ~12KB of ANSI per frame.
        from rich.console import Console
        from rich.segment import Segment

        renderable = FocusPreviewKittyImage(
            image_id=1,
            band_cell_width=120,
            band_cell_height=22,
            title="test",
        )
        console = Console(
            width=120,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )
        options = console.options.update(width=120)
        segments = list(
            renderable.__rich_console__(console, options)
        )
        styled_text_segments = [
            s for s in segments
            if isinstance(s, Segment)
            and s.style is not None
            and s.control is None  # not a control segment
            and s.text.strip()  # has visible text content
        ]
        self.assertEqual(
            len(styled_text_segments),
            0,
            f"precomposed Kitty band should emit no styled text "
            f"segments but got {len(styled_text_segments)}: "
            f"{[s.text[:20] for s in styled_text_segments[:5]]}",
        )

    def test_focus_preview_kitty_composite_cover_fills_wide_crops(self):
        # Wide crops should now cover-fill the image box instead of
        # leaving subtle top/bottom matte inside it.
        wide_png = self._make_png(width=400, height=100)
        comp_png = _build_composite_band_png(
            wide_png,
            term_width=80,
            image_cell_width=30,
            image_cell_height=12,
            image_id=1,
            title="test",
        )
        comp = fitz.Pixmap(comp_png)

        self.assertTrue(
            comp.alpha,
            "composite band PNG should preserve alpha even after the focus "
            "preview switches to cover-fill behavior",
        )

        image_left = (80 - 30) // 2
        crop_x0 = image_left * 8
        crop_y0 = 1 * 16
        crop_target_w = 30 * 8
        probe_x = crop_x0 + crop_target_w // 2
        probe_y = crop_y0 + 8
        off = (probe_y * comp.width + probe_x) * comp.n
        rgba = tuple(comp.samples[off : off + comp.n])
        self.assertEqual(
            rgba[3],
            255,
            "wide crops should reach the top of the image box instead of "
            "leaving a top letterbox band behind",
        )
        self.assertNotEqual(
            rgba[:3],
            _TEXTURE_BG_RGB,
            "top-center pixels inside the image box should come from the "
            "crop once the internal letterbox matte is removed",
        )

    def test_operator_annotated_kitty_preview_preserves_selected_edge_content(self):
        import scripts.narrator_reader as narrator_reader

        display = self._make_display()
        width, height = 200, 160
        rows = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                if y < 40:
                    rgb = (210, 40, 40)
                elif y >= height - 40:
                    rgb = (40, 80, 210)
                else:
                    rgb = (120, 140, 110)
                off = (y * width + x) * 3
                rows[off : off + 3] = bytes(rgb)
        operator_png = fitz.Pixmap(
            fitz.csRGB, width, height, bytes(rows), False
        ).tobytes("png")

        display.on_focus_preview(
            operator_png,
            label="15-blue/fr-11c",
            source="operator_annotated",
        )

        self.assertIsNotNone(
            display._pending_kitty_transmit,
            "operator-annotated preview should still build a Kitty composite "
            "when Kitty is available",
        )
        comp = fitz.Pixmap(display._pending_kitty_transmit)

        console_width = display._console.size.width
        inner_budget = max(1, console_width - 2)
        image_cw, image_ch = _compute_inline_image_cell_dimensions(
            width,
            height,
            max_cell_height=narrator_reader._INLINE_IMAGE_CELL_HEIGHT,
            max_cell_width=min(
                narrator_reader._INLINE_IMAGE_MAX_CELL_WIDTH,
                inner_budget,
            ),
            terminal_cell_aspect=display._terminal_cell_aspect,
        )
        image_left = (console_width - image_cw) // 2
        crop_x0 = image_left * 8
        crop_y0 = 1 * 16
        crop_target_w = image_cw * 8
        crop_target_h = image_ch * 16
        probe_x = crop_x0 + crop_target_w // 2
        top_probe_y = crop_y0 + 8
        bottom_probe_y = crop_y0 + crop_target_h - 9

        top_rgba = tuple(
            comp.samples[(top_probe_y * comp.width + probe_x) * comp.n :][: comp.n]
        )
        bottom_rgba = tuple(
            comp.samples[(bottom_probe_y * comp.width + probe_x) * comp.n :][: comp.n]
        )

        self.assertEqual(
            top_rgba[:3],
            (210, 40, 40),
            "operator-annotated preview should preserve the selected top edge "
            "instead of clipping it away to cover-fill the box",
        )
        self.assertEqual(
            bottom_rgba[:3],
            (40, 80, 210),
            "operator-annotated preview should preserve the selected bottom "
            "edge instead of clipping it away to cover-fill the box",
        )

    def test_focus_preview_kitty_composite_removes_internal_top_bottom_spacer_rows(self):
        # The preview image should now sit directly between the two border
        # rows. Internal top/bottom spacer rows were reading as a persistent
        # letterbox in smoke.
        comp_png = _build_composite_band_png(
            self._make_png(width=500, height=400),
            term_width=80,
            image_cell_width=30,
            image_cell_height=12,
            image_id=1,
            title="test",
        )
        comp = fitz.Pixmap(comp_png)

        image_left = (80 - 30) // 2
        crop_x0 = image_left * 8

        probe_x = crop_x0 + (30 * 8) // 2
        top_image_probe_y = 1 * 16 + 8
        bottom_image_probe_y = 12 * 16 + 8

        for probe_y in (top_image_probe_y, bottom_image_probe_y):
            off = (probe_y * comp.width + probe_x) * comp.n
            rgba = tuple(comp.samples[off : off + comp.n])
            self.assertEqual(
                rgba[3],
                255,
                "image rows should begin immediately below the top border "
                "and run right up to the row above the bottom border",
            )
            self.assertNotEqual(
                rgba[:3],
                _TEXTURE_BG_RGB,
                "the old internal top/bottom spacer rows should be gone; "
                "probing those rows must hit image content now, not the dark matte",
            )

    def test_focus_preview_kitty_composite_matching_aspect_crop_reaches_image_box(self):
        # Matching-aspect crops should now fill the image box directly.
        # We intentionally removed the all-around inner inset because it
        # read like a second frame inside the segmented surround.
        matching_aspect_png = self._make_png(width=500, height=400)
        comp_png = _build_composite_band_png(
            matching_aspect_png,
            term_width=80,
            image_cell_width=30,
            image_cell_height=12,
            image_id=1,
            title="test",
        )
        comp = fitz.Pixmap(comp_png)

        image_left = (80 - 30) // 2
        crop_x0 = image_left * 8
        crop_y0 = 2 * 16
        probe_x = crop_x0 + 2
        probe_y = crop_y0 + 10
        off = (probe_y * comp.width + probe_x) * comp.n
        rgba = tuple(comp.samples[off : off + comp.n])

        self.assertEqual(
            rgba[3],
            255,
            "matching-aspect crops should reach the image-box edge rather "
            "than leaving behind an extra transparent inner frame",
        )

    def test_history_structured_rows_participate_in_depth_dimming(self):
        self.assertEqual(
            _render_layer_index("basis", 3),
            3,
            "basis rows should dim as they sink below the item header "
            "instead of staying pinned at full-strength header intensity",
        )
        self.assertEqual(
            _render_layer_index("checkpoint", 4),
            4,
            "checkpoint rows should follow the same depth fade below the "
            "header instead of reading as a flat full-strength stack",
        )
        self.assertEqual(
            _render_layer_index("topic", 2),
            2,
            "topic rows should still participate in the local depth fade "
            "so the whole post-header block settles as it descends",
        )

    def test_history_depth_fade_reaches_a_clearer_mid_stack_drop(self):
        self.assertLess(
            _history_tier_dim_factor(4),
            0.67,
            "mid-stack reasoning rows should settle more decisively below the "
            "header instead of holding almost the same value and reading like "
            "one dense wall of text",
        )

    def test_history_alt_rows_keep_a_bright_bone_family(self):
        alt_text = Text()
        _apply_shimmer(alt_text, "A", "line_alt", 0, phase_override=0.0)
        alt_rgb = self._rgb_from_hex(alt_text.spans[0].style)

        self.assertGreaterEqual(
            sum(alt_rgb),
            470,
            "the alternating warm history row should stay bright enough to "
            "read as bone rather than collapsing into muddy brown-on-black",
        )
        self.assertGreaterEqual(
            alt_rgb[0],
            alt_rgb[1],
            "the alternating warm row should still lead with a warm bone/red "
            "channel instead of drifting back toward green-gray",
        )
        self.assertGreater(
            alt_rgb[1],
            alt_rgb[2],
            "the alternating warm row should remain in a bone family with "
            "blue clearly trailing the red/green channels",
        )

    def test_focus_preview_side_texture_exits_solid_blocks_earlier(self):
        glyph, _rgb = _texture_cell(
            distance_from_image=2,
            max_distance=12,
            seed_key=("left", 2, 12),
        )

        self.assertNotIn(
            glyph,
            {"█", "▓"},
            "the side rails should step out of the heavy solid-block phase "
            "earlier so the texture frames the crop without overpowering it",
        )

    def test_trim_near_black_crop_margins_removes_dark_scan_frame(self):
        bordered = self._make_bordered_png(
            width=14,
            height=10,
            border=2,
            inner_rgb=(190, 170, 140),
            border_rgb=(0, 0, 0),
        )
        trimmed = _trim_near_black_crop_margins(bordered)
        pix = fitz.Pixmap(trimmed)

        self.assertEqual(
            (pix.width, pix.height),
            (10, 6),
            "focus-preview crop trimming should drop contiguous near-black "
            "edge matte before we scale it into the composite band",
        )

    def test_trim_uniform_edge_margins_removes_paper_colored_scan_frame(self):
        bordered = self._make_bordered_png(
            width=16,
            height=12,
            border=2,
            inner_rgb=(180, 150, 120),
            border_rgb=(231, 221, 199),
        )
        trimmed = _trim_uniform_edge_margins(bordered)
        pix = fitz.Pixmap(trimmed)

        self.assertEqual(
            (pix.width, pix.height),
            (12, 8),
            "focus-preview crop trimming should also drop uniform paper-"
            "colored scan borders instead of keeping a cream frame "
            "inside the composite band",
        )

    def test_trim_uniform_edge_margins_tolerates_minor_edge_noise(self):
        width = 20
        height = 14
        border = 3
        matte = (231, 221, 199)
        body = (180, 150, 120)
        channels = 3
        rows = bytearray(width * height * channels)
        for y in range(height):
            for x in range(width):
                rgb = matte if (
                    x < border or x >= width - border or y < border or y >= height - border
                ) else body
                # Simulate faint scanner noise / pencil dust in the matte.
                if rgb == matte and (x + y) % 11 == 0:
                    rgb = (220, 209, 188)
                off = (y * width + x) * channels
                rows[off : off + channels] = bytes(rgb)
        noisy = fitz.Pixmap(fitz.csRGB, width, height, bytes(rows), False).tobytes("png")
        trimmed = _trim_uniform_edge_margins(noisy)
        pix = fitz.Pixmap(trimmed)

        self.assertEqual(
            (pix.width, pix.height),
            (14, 8),
            "paper-colored edge trim should survive small matte noise so "
            "real scanned page borders don't keep a visible cream frame",
        )

    def test_focus_preview_kitty_composite_keeps_border_row_background_opaque(self):
        # The image-box negative space should be transparent, but the top
        # border/title row must stay opaque. If the border row is left
        # transparent, stale text from the previous frame shows through
        # the title strip when Rich repaints in place.
        comp_png = _build_composite_band_png(
            self._make_png(width=500, height=400),
            term_width=80,
            image_cell_width=30,
            image_cell_height=12,
            image_id=1,
            title="test",
        )
        comp = fitz.Pixmap(comp_png)

        # Probe the top border row away from the line text and border ink.
        probe_x = comp.width - 12
        probe_y = 4
        off = (probe_y * comp.width + probe_x) * comp.n
        rgba = tuple(comp.samples[off : off + comp.n])

        self.assertEqual(
            rgba,
            (*_TEXTURE_BG_RGB, 255),
            "the border/title row background must stay opaque dark so "
            "old frame text cannot ghost through the transparent PNG",
        )

    def test_focus_preview_kitty_composite_keeps_texture_space_cells_opaque(self):
        # The ornate side field should read as a dark textured panel, not as
        # a transparent stencil. When the composite canvas went RGBA we left
        # glyph==" " cells untouched, which turned the gaps between the dots
        # into literal holes where old history text could show through.
        comp_png = _build_composite_band_png(
            self._make_png(width=500, height=400),
            term_width=80,
            image_cell_width=30,
            image_cell_height=12,
            image_id=1,
            title="test",
        )
        comp = fitz.Pixmap(comp_png)

        # Probe an empty background point in the left textured surround,
        # between border rows and outside the image box.
        probe_x = 6
        probe_y = 3 * 16 + 8
        off = (probe_y * comp.width + probe_x) * comp.n
        rgba = tuple(comp.samples[off : off + comp.n])

        self.assertEqual(
            rgba,
            (*_TEXTURE_BG_RGB, 255),
            "texture background cells must stay opaque dark; only the "
            "image-box negative space should be transparent",
        )

    def test_focus_preview_kitty_composite_title_changes_top_strip_pixels(self):
        # The title strip must actually differ when a title is present.
        # We previously rendered the title into an RGB helper pixmap and
        # then tried to blit it into the RGBA composite with 3-tuples,
        # which fitz rejects as a bad color sequence. `_paint_border_row`
        # swallowed the exception, so the titled and untitled composites
        # were literally identical.
        crop_png = self._make_png(width=500, height=400)
        titled = fitz.Pixmap(
            _build_composite_band_png(
                crop_png,
                term_width=80,
                image_cell_width=30,
                image_cell_height=12,
                image_id=1,
                title="focus preview · 15-blue/fr-10b",
            )
        )
        untitled = fitz.Pixmap(
            _build_composite_band_png(
                crop_png,
                term_width=80,
                image_cell_width=30,
                image_cell_height=12,
                image_id=1,
                title="",
            )
        )

        differing_pixels = 0
        for y in range(0, min(16, titled.height)):
            for x in range(0, min(320, titled.width)):
                if titled.pixel(x, y) != untitled.pixel(x, y):
                    differing_pixels += 1

        self.assertGreater(
            differing_pixels,
            0,
            "a titled composite should visibly differ from an untitled one "
            "in the top strip; otherwise the title paint path is dead",
        )
        self.assertGreater(
            differing_pixels,
            2300,
            "the titled top strip should carry enough rasterized ink mass to "
            "stay legible after terminal downscaling, not just a tiny hairline caption",
        )
    def test_focus_preview_inline_image_renderable_declares_cell_height(self):
        # Rich's layout engine measures a renderable's vertical footprint
        # from what it yields. The inline image escape sequence occupies
        # no text rows on its own, so the renderable must yield exactly
        # `cell_height` blank lines alongside the escape sequence, so
        # Rich reserves the right amount of vertical space and positions
        # the next panel below where the image visually lands.
        from rich.console import Console

        png = b"\x89PNG\r\n\x1a\nfake png"
        renderable = FocusPreviewInlineImage(
            png_bytes=png, cell_width=60, cell_height=18
        )
        console = Console(
            width=120,
            record=True,
            color_system="truecolor",
            force_terminal=True,
        )
        with console.capture() as capture:
            console.print(renderable)
        captured = capture.get()
        # Must contain the escape sequence exactly once.
        self.assertEqual(
            captured.count("\x1b]1337;File="),
            1,
            "inline image must emit exactly one OSC 1337 sequence per render",
        )
        # Line count: must produce cell_height rows of vertical footprint
        # so Rich reserves that much space. Rich's print adds a trailing
        # newline; count the non-final newlines.
        lines = captured.splitlines()
        self.assertGreaterEqual(
            len(lines),
            18,
            "renderable must reserve cell_height rows of vertical space",
        )

    def test_otsu_threshold_separates_bimodal_distribution(self):
        # Two clear clusters: 200 samples around 40 (ink), 200 around 220 (paper).
        luminances = [40.0] * 200 + [220.0] * 200
        threshold = _otsu_threshold(luminances)
        self.assertGreater(
            threshold,
            40.0,
            "Otsu threshold must fall above the ink cluster",
        )
        self.assertLess(
            threshold,
            220.0,
            "Otsu threshold must fall below the paper cluster",
        )

    def test_otsu_threshold_handles_uniform_input(self):
        # Degenerate case: all one value. The function must not crash;
        # the exact returned threshold is unconstrained since the input
        # is not bimodal, but it should be a real finite number.
        threshold = _otsu_threshold([128.0] * 50)
        self.assertIsInstance(threshold, (int, float))
        self.assertTrue(math.isfinite(float(threshold)))

    def test_half_block_pipeline_preserves_ink_position(self):
        # End-to-end invariant: if we render a pure-black horizontal strip
        # across the TOP half of a pure-white field, the rendered output
        # must have ink-colored cells in the upper rows and paper-colored
        # cells in the lower rows. This is the single most important
        # legibility guarantee — ink position through the pipeline.
        width = 96
        height = 96
        paper = (235, 230, 222)
        ink = (15, 15, 15)
        buf = bytearray(paper * (width * height))
        for y in range(0, height // 2):
            for x in range(width):
                offset = (y * width + x) * 3
                buf[offset : offset + 3] = bytes(ink)
        pix = fitz.Pixmap(fitz.csRGB, width, height, bytes(buf), False)

        pixels = _build_focus_preview_pixels(
            pix.tobytes("png"),
            max_width_chars=12,
            # 12 terminal rows = 24 sampled rows (half-blocks = 2× vertical)
            max_height_rows=24,
        )
        renderable = _render_focus_preview_pixels(
            pixels,
            now=0.0,
            pending=False,
        )

        def _row_avg_luma(row_text) -> float:
            # Collect all bg colors from all spans in the row; also treat
            # fg (top half-pixel) as contributing to the visible luminance
            # of that terminal row. For a row entirely above the ink/paper
            # boundary, both fg and bg should be ink-dark.
            lumas: list[float] = []
            for span in row_text.spans:
                if not isinstance(span.style, str) or " on " not in span.style:
                    continue
                fg, bg = span.style.split(" on ")
                for hex_color in (fg, bg):
                    hex_color = hex_color.lstrip("#")
                    if len(hex_color) != 6:
                        continue
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    lumas.append(0.299 * r + 0.587 * g + 0.114 * b)
            if not lumas:
                return float("nan")
            return sum(lumas) / len(lumas)

        rows = list(renderable.renderables)
        self.assertGreaterEqual(
            len(rows),
            6,
            "pipeline should emit multiple terminal rows for a 24-sample-row input",
        )
        top_luma = _row_avg_luma(rows[1])  # skip row 0 to avoid boundary
        bottom_luma = _row_avg_luma(rows[-2])
        self.assertLess(
            top_luma,
            bottom_luma - 80,
            "ink strip at the top of the source must render as distinctly darker "
            "terminal rows than the paper at the bottom",
        )

    def test_scaled_preview_size_respects_terminal_row_budget_in_glyph_mode(self):
        width, height = _scaled_preview_size(
            475,
            218,
            max_width_chars=72,
            max_height_rows=18,
        )

        self.assertLessEqual(
            height,
            18,
            "glyph-mode preview sizing should not retain the old half-block *2 row budget and overflow vertically",
        )
        self.assertLessEqual(
            width,
            72,
            "scaled preview width should still honor the horizontal character budget",
        )

    def test_pending_focus_preview_rerender_is_quantized_instead_of_rebuilding_every_tick(self):
        display = self._make_display()
        display._inline_images_supported = False
        display._kitty_graphics_supported = False
        display.on_focus_preview(
            self._make_png(width=64, height=36),
            label="15-blue/fr-12a",
            source="mock_tricky",
        )
        display.on_header("[item 2/12] 27-blue-2023/fr-3 (balanced_equation, 4.0 pts)")

        with mock.patch(
            "scripts.narrator_reader._render_focus_preview_pixels",
            return_value=Group(Text("preview")),
        ) as render_mock:
            with mock.patch("scripts.narrator_reader.time.monotonic", return_value=1.01):
                display.render()
            with mock.patch("scripts.narrator_reader.time.monotonic", return_value=1.05):
                display.render()
            with mock.patch("scripts.narrator_reader.time.monotonic", return_value=1.15):
                display.render()

        self.assertEqual(
            render_mock.call_count,
            2,
            "pending preview frames should be cached within a transition bucket instead of rebuilding on every 24 FPS tick",
        )
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

    def test_basis_and_review_render_directly_under_topic_before_checkpoints(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(
            ("checkpoint", "Core issue: ozone drawing misses resonance.", None)
        )
        display.history.append(("topic", "topic line", "match"))
        display.history.append(
            ("basis", "Correct setup, lost credit for octet violation.", None)
        )
        display.history.append(("review_marker", "Human review warranted.", None))

        entries = display._build_display_entries()
        summary = [(entry[0], entry[1]) for entry, _recent, _depth in entries]

        self.assertEqual(
            summary,
            [
                ("header", "[item 1/6] first"),
                ("topic", "topic line"),
                ("basis", "Correct setup, lost credit for octet violation."),
                ("review_marker", "Human review warranted."),
                ("checkpoint", "Core issue: ozone drawing misses resonance."),
            ],
        )

    def test_display_no_longer_exposes_scrollback_snapshot_affordance(self):
        display = self._make_display()
        self.assertFalse(
            hasattr(display, "take_scrollback_snapshot"),
            "whole-terminal scrollback archiving was only a temporary lookback stopgap and should be removed now that live in-pane scrolling owns the affordance",
        )

    def test_display_no_longer_tracks_scrollback_archive_bookkeeping(self):
        display = self._make_display()
        self.assertFalse(
            hasattr(display, "_scrollback_archived_headers"),
            "the reader should stop carrying per-header archive bookkeeping once the temporary scrollback snapshot path is removed",
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
        self.assertLess(
            cool_red + cool_green + cool_blue,
            545,
            "cool live line should tone down from the brighter electric wash and carry a darker, more pigmented aqua body",
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
        self.assertLess(
            warm_red + warm_green + warm_blue,
            540,
            "soft-warm live line should stop hovering near white and read as an actual peach/apricot note",
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

    def test_header_title_returns_to_white_scene_setter(self):
        display = self._make_display()
        display.current_model = "qwen3p5-35B-A3B"
        display.on_header("[item 4/6] 15-blue/fr-10a")

        group = display.render()

        header_panel = group.renderables[0]
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable

        title_end = header_renderable.plain.index(" · sumi-e")
        title_styles = self._styles_in_range(header_renderable, 0, title_end)

        self.assertIn(
            "bold bright_white",
            title_styles,
            "PROJECT PAINT DRY should go back to a flat bright-white scene-setter instead of participating in the accent system",
        )
        self.assertFalse(
            any("#" in style for style in title_styles),
            "PROJECT PAINT DRY should not carry lacquer-color spans in the rebuilt reference-led surface",
        )

    def test_header_subtitle_uses_generic_narrator_label(self):
        display = self._make_display()
        display.current_model = "Qwen3.6-35B-A3B-oQ8"
        display.on_header("[item 4/6] 15-blue/fr-10a")

        group = display.render()

        header_panel = group.renderables[0]
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable

        self.assertIn(
            "thinking narrator · live",
            header_renderable.plain,
            "the top chrome should stop hard-coding the old Bonsai-specific "
            "label now that Paint Dry regularly runs other narrator backends",
        )
        self.assertNotIn(
            "bonsai narrator",
            header_renderable.plain,
            "the subtitle should be truthful or generic, not claim Bonsai "
            "when the session is running another model surface",
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
            verdict="ceiling",
            grader_score=2.0,
            truth_score=2.0,
            max_points=2.0,
        )
        display.on_topic(
            "44s · Grader: 1/3. Prof: 1.5/3. · Within acceptable range, below ceiling.",
            verdict="within_band",
            grader_score=1.0,
            truth_score=1.5,
            max_points=3.0,
            acceptable_score_floor=1.0,
            acceptable_score_ceiling=1.5,
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
        expected_on_target = _scorebug_big_value_rows("2.0/12.0")
        expected_in_range = _scorebug_big_value_rows("1.0/1.5")
        expected_left = _scorebug_big_value_rows("1.0/1.0")
        expected_bad = _scorebug_big_value_rows("1.5/1.5")

        self.assertIn("PROJECT PAINT DRY", _extract_plain(header_panel.renderable))
        self.assertIn("CURRENT MODEL", scorebug_text)
        self.assertIn("SET", scorebug_text)
        self.assertIn("TRICKY", scorebug_text)
        self.assertIn("ITEM", scorebug_text)
        self.assertIn("4/6", scorebug_text)
        self.assertIn("ON TARGET", scorebug_text)
        self.assertIn("IN RANGE", scorebug_text)
        self.assertIn("LEFT ON TABLE", scorebug_text)
        self.assertIn("BAD CALLS", scorebug_text)
        self.assertEqual(spacer_row.plain.strip(), "")
        normalized_top = self._normalize_scorebug_texture(tally_value_top.plain)
        normalized_mid = self._normalize_scorebug_texture(tally_value_mid.plain)
        normalized_bottom = self._normalize_scorebug_texture(tally_value_bottom.plain)
        self.assertIn(expected_on_target[0], normalized_top)
        self.assertIn(expected_on_target[1], normalized_mid)
        self.assertIn(expected_on_target[2], normalized_bottom)
        self.assertIn(expected_in_range[0], normalized_top)
        self.assertIn(expected_in_range[1], normalized_mid)
        self.assertIn(expected_in_range[2], normalized_bottom)
        self.assertIn(expected_left[0], normalized_top)
        self.assertIn(expected_left[1], normalized_mid)
        self.assertIn(expected_left[2], normalized_bottom)
        self.assertIn(expected_bad[0], normalized_top)
        self.assertIn(expected_bad[1], normalized_mid)
        self.assertIn(expected_bad[2], normalized_bottom)
        current_model_bg = self._background_hex(
            self._style_for_substring(scorebug_text_obj, "CURRENT MODEL")
        )
        current_model_label_fg = self._foreground_hex(
            self._style_for_substring(scorebug_text_obj, "CURRENT MODEL")
        )
        current_model_value_fg = self._foreground_hex(
            self._style_for_substring(scorebug_text_obj, "gemma-4-26b-a4b-it-bf16")
        )
        set_bg = self._background_hex(
            self._style_for_substring(scorebug_text_obj, "SET")
        )
        item_bg = self._background_hex(
            self._style_for_substring(scorebug_text_obj, "ITEM")
        )
        self.assertEqual(
            {current_model_bg, set_bg, item_bg},
            {current_model_bg},
            "the metadata strip should read as one shared smoke field, not three separate categorical capsules",
        )
        self.assertGreater(
            self._hex_luminance(current_model_label_fg),
            545,
            "the top metadata strip should carry a little more white mass so the identity band doesn't feel too slight once the scoreboard settles in",
        )
        self.assertGreater(
            self._hex_luminance(current_model_value_fg),
            655,
            "scorebug metadata values should read closer to blocky white lettering than dim instrument text",
        )
        on_target_label_style = self._style_for_substring(tally_text_obj, "ON TARGET")
        in_range_label_style = self._style_for_substring(tally_text_obj, "IN RANGE")
        left_label_style = self._style_for_substring(tally_text_obj, "LEFT ON TABLE")
        bad_label_style = self._style_for_substring(tally_text_obj, "BAD CALLS")
        on_target_label_bg = self._background_hex(on_target_label_style)
        in_range_label_bg = self._background_hex(in_range_label_style)
        left_label_bg = self._background_hex(left_label_style)
        bad_label_bg = self._background_hex(bad_label_style)
        self.assertEqual(
            {on_target_label_bg, in_range_label_bg, left_label_bg, bad_label_bg},
            {on_target_label_bg},
            "the tally labels should sit on one shared charcoal field instead of three color-coded boards",
        )
        self.assertGreaterEqual(
            len(
                {
                    self._foreground_hex(on_target_label_style),
                    self._foreground_hex(in_range_label_style),
                    self._foreground_hex(left_label_style),
                    self._foreground_hex(bad_label_style),
                }
            ),
            2,
            "the labels can still carry restrained accent differences, but only in the foreground ink, not in separate slab backgrounds",
        )
        cell_width = len(f" {expected_on_target[0]} ")
        separator_width = 2
        # Gauge Saints II: TOTAL and TURN now sit leftmost in the
        # big-value strip as their own scoreboard plates, so ON TARGET
        # no longer starts at offset 0. This test only cares about the
        # relative layout of the three grading plates, so it anchors
        # on wherever ON TARGET lands and walks from there.
        on_target_start = tally_text_obj.plain.index("ON TARGET")
        in_range_start = tally_text_obj.plain.index("IN RANGE")
        left_start = tally_text_obj.plain.index("LEFT ON TABLE")
        bad_start = tally_text_obj.plain.index("BAD CALLS")
        self.assertGreater(
            in_range_start,
            on_target_start,
            "IN RANGE should appear to the right of ON TARGET in the scorebug strip",
        )
        self.assertGreater(
            left_start,
            in_range_start,
            "LEFT ON TABLE should appear to the right of IN RANGE in the scorebug strip",
        )
        self.assertGreater(
            bad_start,
            left_start,
            "BAD CALLS should appear rightmost among the grading tally cells",
        )

        def _first_strong_style_in_cell(row: Text, start: int) -> str:
            end = start + cell_width
            for span in row.spans:
                if not isinstance(span.style, str):
                    continue
                if span.start < start or span.end > end:
                    continue
                if span.style.startswith("bold #"):
                    return span.style
            raise AssertionError(
                f"no strong scorebug span found in cell range {start}:{end}"
            )

        on_target_styles = self._styles_in_range(
            tally_value_top,
            on_target_start,
            on_target_start + cell_width,
        )
        self.assertGreaterEqual(
            len(on_target_styles),
            2,
            "scorebug numerals should use at least two stroke weights/colors inside a single value cell",
        )
        on_target_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_on_target[0].strip(),
        )
        in_range_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_in_range[0].strip(),
        )
        left_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_left[0].strip(),
        )
        bad_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_bad[0].strip(),
        )
        self.assertEqual(
            {
                self._background_hex(on_target_top_style),
                self._background_hex(left_top_style),
                self._background_hex(bad_top_style),
            },
            {None},
            "scorebug value strokes should no longer carry filled backgrounds; "
            "the plate should read through sparse texture and border ink instead",
        )
        self.assertTrue(
            any(ch in tally_value_top.plain for ch in "░▒·┈╎"),
            "the top score field should use visible UTF-8 texture characters instead of smooth colored whitespace",
        )
        self.assertTrue(
            any(ch in tally_value_mid.plain for ch in "░▒·┈╎"),
            "the middle score field should use visible UTF-8 texture characters instead of smooth colored whitespace",
        )
        self.assertTrue(
            any(ch in tally_value_bottom.plain for ch in "░▒·┈╎"),
            "the bottom score field should use visible UTF-8 texture characters instead of smooth colored whitespace",
        )
        self.assertGreaterEqual(
            self._scorebug_texture_count(tally_value_top.plain),
            10,
            "the top score field should carry enough texture mass to register at a glance rather than only in a few isolated specks",
        )
        self.assertGreaterEqual(
            self._scorebug_texture_count(tally_value_mid.plain),
            7,
            "the middle score field should carry enough texture mass to read as a surface, not an almost-empty gradient band",
        )
        self.assertGreaterEqual(
            self._scorebug_texture_count(tally_value_bottom.plain),
            7,
            "the bottom score field should carry enough texture mass to read as a surface, not an almost-empty gradient band",
        )
        self.assertNotIn(
            "╎",
            tally_value_mid.plain + tally_value_top.plain + tally_value_bottom.plain,
            "scorebug texture should avoid thin vertical artifact glyphs that read like accidental banding",
        )
        on_target_mid_style = self._style_for_normalized_scorebug_substring(
            tally_value_mid,
            expected_on_target[1].strip(),
        )
        left_mid_style = self._style_for_normalized_scorebug_substring(
            tally_value_mid,
            expected_left[1].strip(),
        )
        bad_mid_style = self._style_for_normalized_scorebug_substring(
            tally_value_mid,
            expected_bad[1].strip(),
        )
        on_target_bottom_style = self._style_for_normalized_scorebug_substring(
            tally_value_bottom,
            expected_on_target[2].strip(),
        )
        self.assertEqual(
            {
                self._background_hex(on_target_mid_style),
                self._background_hex(on_target_bottom_style),
            },
            {None},
            "middle and bottom numeral strokes should also stay foreground-only "
            "once the scorebug drops its filled slabs",
        )
        fg_lumas = [
            self._hex_luminance(self._foreground_hex(on_target_top_style)),
            self._hex_luminance(self._foreground_hex(on_target_mid_style)),
            self._hex_luminance(self._foreground_hex(on_target_bottom_style)),
        ]
        self.assertGreater(
            max(fg_lumas) - min(fg_lumas),
            18,
            "the scorebug should still keep some row-to-row tonal drift in "
            "the numeral ink after the fill is removed",
        )
        # Find the first two strong-stroke spans inside the ON TARGET
        # cell by walking the tally_value_top spans from on_target_start
        # rather than indexing by absolute span index (which would now
        # point at the TOTAL/TURN plates' spans).
        on_target_spans = [
            span
            for span in tally_value_top.spans
            if (
                isinstance(span.style, str)
                and span.start >= on_target_start
                and span.end <= on_target_start + cell_width
            )
        ]
        self.assertGreaterEqual(
            len(on_target_spans),
            3,
            "ON TARGET value cell should carry multiple styled spans",
        )
        on_target_top_strong = on_target_spans[1].style
        on_target_bottom_style = self._style_for_normalized_scorebug_substring(
            tally_value_bottom,
            expected_on_target[2].strip(),
        )
        self.assertNotEqual(
            on_target_top_strong,
            on_target_bottom_style,
            "scorebug value rows should now drift tonally across the board instead of sitting on one flat background",
        )
        left_bottom_style = self._style_for_normalized_scorebug_substring(
            tally_value_bottom,
            expected_left[2].strip(),
        )
        bad_bottom_style = self._style_for_normalized_scorebug_substring(
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
        top_board_styles = {
            _first_strong_style_in_cell(tally_value_top, on_target_start),
            _first_strong_style_in_cell(tally_value_top, in_range_start),
            _first_strong_style_in_cell(tally_value_top, left_start),
            _first_strong_style_in_cell(tally_value_top, bad_start),
        }
        mid_board_styles = {
            _first_strong_style_in_cell(tally_value_mid, on_target_start),
            _first_strong_style_in_cell(tally_value_mid, in_range_start),
            _first_strong_style_in_cell(tally_value_mid, left_start),
            _first_strong_style_in_cell(tally_value_mid, bad_start),
        }
        bottom_board_styles = {
            _first_strong_style_in_cell(tally_value_bottom, on_target_start),
            _first_strong_style_in_cell(tally_value_bottom, in_range_start),
            _first_strong_style_in_cell(tally_value_bottom, left_start),
            _first_strong_style_in_cell(tally_value_bottom, bad_start),
        }
        self.assertEqual(
            len(top_board_styles),
            4,
            "AT CEILING, IN RANGE, LEFT ON TABLE, and BAD CALLS should each keep a distinct top-row board family",
        )
        self.assertEqual(
            len(mid_board_styles),
            4,
            "the middle row should also keep distinct board families so lawful-in-range calls do not collapse into either ceiling hits or miss buckets",
        )
        self.assertEqual(
            len(bottom_board_styles),
            4,
            "the bottom row should keep distinct board families too, not just row-to-row drift inside one shared palette",
        )
        self.assertTrue(
            all(
                self._hex_luminance(
                    self._foreground_hex(style)
                ) > 635
                for style in {
                    on_target_top_strong,
                    in_range_top_style,
                    left_top_style,
                    bad_top_style,
                }
            ),
            "the main numeral strokes should now carry more white weight so the scorebug can sit in the same bold white language as the surrounding interface",
        )
        self.assertEqual(
            on_target_spans[2].style,
            on_target_top_strong,
            "top-row horizontal bars should stay on the strong stroke tier so the numerals read chunkier",
        )

    def test_scorebug_keeps_band_ceiling_hits_out_of_on_target(self):
        display = self._make_display()

        display.on_topic(
            "41s · Grader: 1.5/2. Prof: 1/2. · Acceptable band: 1/2 to 1.5/2.",
            verdict="within_band",
            grader_score=1.5,
            truth_score=1.0,
            max_points=2.0,
            acceptable_score_floor=1.0,
            acceptable_score_ceiling=1.5,
        )

        self.assertEqual(
            display.score_on_target_points,
            0.0,
            "lawful band hits at the acceptable ceiling should not be treated "
            "as exact truth hits when truth_score is lower",
        )
        self.assertEqual(
            display.score_within_band_points,
            1.5,
            "lawful band hits should accrue in the IN RANGE bucket even when "
            "they sit at the acceptable ceiling",
        )
        self.assertEqual(
            display.score_within_band_potential,
            1.5,
            "the IN RANGE bucket should still report the lawful ceiling as "
            "its potential when the exact truth stays lower",
        )

    def test_scorebug_counts_truth_match_as_on_target_even_with_band(self):
        display = self._make_display()

        display.on_topic(
            "41s · Grader: 1/2. Prof: 1/2. · Acceptable band: 1/2 to 1.5/2.",
            verdict="match",
            grader_score=1.0,
            truth_score=1.0,
            max_points=2.0,
            acceptable_score_floor=1.0,
            acceptable_score_ceiling=1.5,
        )

        self.assertEqual(
            display.score_on_target_points,
            1.0,
            "truth_score remains the exact-hit target even when a lawful "
            "acceptable band is also present",
        )
        self.assertEqual(
            display.score_within_band_points,
            0.0,
            "exact truth matches should not be diverted into the lawful-band "
            "bucket",
        )

    def test_annotate_current_focus_item_relaunches_annotator_and_refreshes_preview(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY++",
            subset_count=15,
            scans_dir="/tmp/scans",
            focus_regions_path="/tmp/focus-regions.yaml",
        )
        display.on_focus_preview(
            self._make_png(rgb=(10, 20, 30)),
            label="15-blue/fr-11c",
            source="operator_annotated",
        )
        gt_item = mock.Mock(exam_id="15-blue", question_id="fr-11c", page=3)
        updated_region = mock.Mock(
            page=3,
            x=0.1,
            y=0.2,
            width=0.3,
            height=0.4,
            source="operator_annotated",
        )

        with (
            mock.patch(
                "scripts.narrator_reader.load_ground_truth",
                return_value=[gt_item],
            ),
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch("scripts.narrator_reader.subprocess.run") as run_mock,
            mock.patch(
                "scripts.narrator_reader.load_focus_regions",
                return_value={("15-blue", "fr-11c"): updated_region},
            ),
            mock.patch(
                "scripts.narrator_reader.extract_page_image",
                return_value=b"page-png",
            ),
            mock.patch(
                "scripts.narrator_reader.render_focus_preview",
                return_value=self._make_png(rgb=(40, 50, 60)),
            ),
        ):
            refreshed = display.annotate_current_focus_item()

        self.assertTrue(refreshed)
        run_mock.assert_called_once()
        cmd = run_mock.call_args.args[0]
        self.assertTrue(
            run_mock.call_args.kwargs.get("capture_output"),
            "annotate-current-item should capture annotator stdout/stderr so the "
            "tool cannot scribble on the live Paint Dry terminal",
        )
        self.assertTrue(
            run_mock.call_args.kwargs.get("text"),
            "captured annotator output should be decoded for diagnostic logging",
        )
        self.assertIn("annotate_focus_regions.py", cmd[1])
        self.assertIn("--pdf", cmd)
        self.assertIn("/tmp/scans/15 blue.pdf", cmd)
        self.assertIn("--page", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--targets", cmd)
        self.assertIn("15-blue/fr-11c", cmd)
        self.assertIn("--config", cmd)
        self.assertIn("/tmp/focus-regions.yaml", cmd)
        self.assertEqual(display.focus_preview_png, self._make_png(rgb=(40, 50, 60)))
        self.assertEqual(display.focus_preview_label, "15-blue/fr-11c")
        self.assertEqual(display.focus_preview_source, "operator_annotated")
        self.assertEqual(display.status_line, "Updated focus preview for 15-blue/fr-11c.")

    def test_session_meta_message_handler_preserves_scans_and_focus_region_paths(self):
        source = inspect.getsource(narrator_reader.main)

        self.assertIn(
            'scans_dir=msg.get("scans_dir")',
            source,
            "the live reader must forward scans_dir from session_meta events "
            "into on_session_meta() so annotate-current-item can find the PDFs",
        )
        self.assertIn(
            'focus_regions_path=msg.get("focus_regions_path")',
            source,
            "the live reader must forward focus_regions_path from session_meta "
            "events into on_session_meta() so in-window annotation refreshes "
            "edit the right config file",
        )

    def test_annotate_current_focus_item_reports_missing_focus_target(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY++",
            subset_count=15,
            scans_dir="/tmp/scans",
        )

        refreshed = display.annotate_current_focus_item()

        self.assertFalse(refreshed)
        self.assertEqual(
            display.status_line,
            "Annotate current item unavailable: no focus target.",
        )

    def test_annotate_current_focus_item_reports_annotator_failure(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY++",
            subset_count=15,
            scans_dir="/tmp/scans",
            focus_regions_path="/tmp/focus-regions.yaml",
        )
        display.on_focus_preview(
            self._make_png(rgb=(10, 20, 30)),
            label="15-blue/fr-11c",
            source="operator_annotated",
        )
        gt_item = mock.Mock(exam_id="15-blue", question_id="fr-11c", page=3)

        with (
            mock.patch(
                "scripts.narrator_reader.load_ground_truth",
                return_value=[gt_item],
            ),
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch(
                "scripts.narrator_reader.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["annotate"]),
            ),
        ):
            refreshed = display.annotate_current_focus_item()

        self.assertFalse(refreshed)
        self.assertEqual(
            display.status_line,
            "Annotate current item failed: annotator subprocess failed for 15-blue/fr-11c.",
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
        self.assertIn("IN RANGE", scorebug_text)
        self.assertIn("LEFT ON TABLE", scorebug_text)
        self.assertIn("BAD CALLS", scorebug_text)
        self.assertEqual(spacer_row.plain.strip(), "")
        self.assertIn(
            zero_rows[0],
            self._normalize_scorebug_texture(tally_value_top.plain),
        )
        self.assertIn(
            zero_rows[1],
            self._normalize_scorebug_texture(tally_value_mid.plain),
        )
        self.assertIn(
            zero_rows[2],
            self._normalize_scorebug_texture(tally_value_bottom.plain),
        )
        self.assertTrue(
            any(ch in tally_value_top.plain for ch in "░▒·┈╎"),
            "even the zeroed tally surface should keep the scorebug's UTF-8 field texture alive instead of reverting to smooth whitespace",
        )
        self.assertEqual(
            value_floor_gap.plain.strip(),
            "",
            "scorebug values should keep a blank gutter below the bottom numeral row so the strokes don't slam into the panel floor",
        )

    def test_topic_kind_helper_distinguishes_ceiling_and_within_band_success(self):
        import scripts.narrator_reader as module

        self.assertEqual(module._topic_kind_for_verdict("ceiling"), "topic_match")
        self.assertEqual(
            module._topic_kind_for_verdict("within_band"),
            "topic_within_band",
        )
        self.assertEqual(
            module._topic_kind_for_verdict("overshoot"),
            "topic_overshoot",
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

    @staticmethod
    def _scorebug_plain(display) -> str:
        """Concat every row of the scorebug panel into one plain string."""
        scorebug_panel = display.render().renderables[1]
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        return "\n".join(
            row.plain if isinstance(row, Text) else ""
            for row in scorebug_renderable.renderables
        )

    @staticmethod
    def _scorebug_value_rows(display) -> tuple[Text, Text, Text, Text]:
        """Return the (labels, value_top, value_middle, value_bottom)
        row Text objects for the big-value plate strip."""
        scorebug_panel = display.render().renderables[1]
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        rows = scorebug_renderable.renderables
        # rows[0] = model/set/item, rows[1] = spacer,
        # rows[2] = labels, rows[3..5] = value rows.
        return rows[2], rows[3], rows[4], rows[5]

    def _plate_tall_value(self, display, label: str) -> str:
        """Extract the digit string currently sitting inside the named
        big-value plate by slicing the three value rows at the label's
        column range and reverse-looking-up each 3-character glyph
        column against the ``_SCOREBUG_BIG_DIGITS`` table."""
        from scripts.narrator_reader import _SCOREBUG_BIG_DIGITS

        labels_row, value_top, value_middle, value_bottom = (
            self._scorebug_value_rows(display)
        )
        label_start = labels_row.plain.index(label)
        # Find the next plate's label start (or the end of the row) so
        # this plate's cell range is bounded.
        remaining = labels_row.plain[label_start + len(label):]
        next_non_space = None
        for offset, ch in enumerate(remaining):
            if ch != " ":
                next_non_space = offset
                break
        if next_non_space is None:
            label_end = len(labels_row.plain)
        else:
            label_end = label_start + len(label) + next_non_space

        # Pull the value-row slices in this plate's cell range.
        top_plain = self._normalize_scorebug_texture(value_top.plain)
        middle_plain = self._normalize_scorebug_texture(value_middle.plain)
        bottom_plain = self._normalize_scorebug_texture(value_bottom.plain)
        top_cell = top_plain[label_start:label_end]
        middle_cell = middle_plain[label_start:label_end]
        bottom_cell = bottom_plain[label_start:label_end]

        # The top row is the most reliable anchor because digit glyphs
        # always carry visible ink in their top row (╔, ═, ╗, or a
        # combination). Find the first non-space column in the top row
        # and walk 3-char glyph columns from there. Middle and bottom
        # rows must be read at the SAME column range without stripping,
        # because some glyph bottoms (e.g. 7: "║  ") legitimately carry
        # trailing spaces that would be destroyed by strip().
        glyph_start = len(top_cell) - len(top_cell.lstrip(" "))
        glyph_to_char = {rows: ch for ch, rows in _SCOREBUG_BIG_DIGITS.items()}
        result: list[str] = []
        col = glyph_start
        while col + 3 <= len(top_cell):
            top_glyph = top_cell[col:col + 3]
            if top_glyph == "   ":
                break  # hit the trailing padding after the last digit
            middle_glyph = middle_cell[col:col + 3] if col + 3 <= len(middle_cell) else "   "
            bottom_glyph = bottom_cell[col:col + 3] if col + 3 <= len(bottom_cell) else "   "
            triple = (top_glyph, middle_glyph, bottom_glyph)
            ch = glyph_to_char.get(triple)
            if ch is None:
                break
            result.append(ch)
            col += 3
        return "".join(result)

    def test_header_starts_total_and_turn_timers(self):
        """After the Gauge Saints II promotion, TOTAL and TURN render
        as tall scorebug plates rather than flat header telemetry. The
        timer values should still reflect elapsed time since the first
        header fired."""
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=100.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=107.0):
            self.assertEqual(self._plate_tall_value(display, "TOTAL"), "7")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=107.0):
            self.assertEqual(self._plate_tall_value(display, "TURN"), "7")

    def test_turn_timer_persists_within_item_and_resets_on_next_header(self):
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=200.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=201.0):
            display.on_delta("Tracing")
        display.on_commit("status")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=205.0):
            self.assertEqual(self._plate_tall_value(display, "TURN"), "5")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=205.0):
            self.assertEqual(self._plate_tall_value(display, "TOTAL"), "5")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=206.0):
            display.on_delta("Rechecking")
        display.on_drop("dedup", "Rechecking the same unit conversion.")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=208.0):
            self.assertEqual(self._plate_tall_value(display, "TURN"), "8")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=208.0):
            self.assertEqual(self._plate_tall_value(display, "TOTAL"), "8")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=209.0):
            display.on_delta("Tracing")
        display.on_rollback_live()
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=211.0):
            self.assertEqual(self._plate_tall_value(display, "TURN"), "11")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=211.0):
            self.assertEqual(self._plate_tall_value(display, "TOTAL"), "11")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=212.0):
            display.on_header("[item 2/6] 15-blue/fr-2")
        # After the fr-2 header at t=212.0, TURN resets so at t=215.0
        # we have TOTAL=15 but TURN=3 — timers in distinct plates, no
        # longer collapsed into a shared flat line.
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=215.0):
            self.assertEqual(self._plate_tall_value(display, "TOTAL"), "15")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=215.0):
            self.assertEqual(self._plate_tall_value(display, "TURN"), "3")

    def test_run_counters_render_as_scoreboard_dials_not_flat_telemetry(self):
        """The three event-count run quantities (emitted, dedup, empty)
        should read as scoreboard-language dials in the top header
        band, not as a flat ``emitted=N  dedup=N  empty=N`` telemetry
        tail. TOTAL and TURN are handled by
        ``test_timers_render_as_tall_scorebug_plates_not_small_capsules``
        because they are promoted into the tall scorebug-plate
        treatment below.

        Falsifiable shape:
          * EMITTED / DEDUP / EMPTY each carry an uppercase dial label
            in the header panel's plain text.
          * The legacy flat ``total=``/``turn=``/``emitted=``/``dedup=``/
            ``empty=`` substring prefixes are gone from the header panel
            plain text — they are the exact shape the attractor is
            retiring.
          * Each of the three event-count dial labels lives inside a
            capsule cell (same ``on #...`` background idiom used by
            the existing scorebug cells), so a future pass cannot
            silently collapse the dials back into unstyled flat text.
          * The underlying counts are still legible in the header plain
            text.
          * The rejected/drops panel title no longer carries the literal
            ``dedup=``/``empty=`` flat telemetry either, because the
            attractor condition #1 covers the whole "flat line of plain
            telemetry" surface rather than only the top-band header.
        """
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=500.0):
            display.on_header("[item 1/6] 15-blue/fr-1")

        display.on_delta("I'm tracing the stoichiometry.")
        display.on_commit("thought")
        display.on_delta("Tracing the stoichiometry setup.", mode="status")
        display.on_commit("status")
        display.on_drop("dedup", "Rechecking the same unit conversion.")
        display.on_drop("dedup", "Rechecking the same unit conversion.")
        display.on_drop("empty", "")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=507.0):
            group = display.render()

        header_panel = group.renderables[0]
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable
        header_text_obj = header_renderable
        if isinstance(header_text_obj, Group):
            header_text_obj = header_text_obj.renderables[0]
        self.assertIsInstance(
            header_text_obj,
            Text,
            "header panel should still render a Text row carrying the dial labels",
        )
        header_plain = header_text_obj.plain

        for label in ("EMITTED", "DEDUP", "EMPTY"):
            self.assertIn(label, header_plain)

        for legacy in ("total=", "turn=", "emitted=", "dedup=", "empty="):
            self.assertNotIn(legacy, header_plain)

        emitted_style = self._style_for_substring(header_text_obj, "EMITTED")
        dedup_style = self._style_for_substring(header_text_obj, "DEDUP")
        empty_style = self._style_for_substring(header_text_obj, "EMPTY")
        self.assertEqual(
            self._background_hex(emitted_style),
            "#4a9838",
            "EMITTED should keep the restored bright green board, not the "
            "older muted telemetry green",
        )
        self.assertEqual(
            self._foreground_hex(emitted_style),
            "#f4ffea",
            "EMITTED label ink should stay on the hotter green-white accent",
        )
        self.assertEqual(
            self._background_hex(dedup_style),
            "#809326",
            "DEDUP should read as the restored chartreuse board rather than "
            "the duller amber telemetry pass",
        )
        self.assertEqual(
            self._foreground_hex(dedup_style),
            "#fbffd7",
            "DEDUP label ink should stay in the brighter yellow-chartreuse family",
        )
        self.assertEqual(
            self._background_hex(empty_style),
            "#a13f2f",
            "EMPTY should keep the restored ember-red board rather than a "
            "muddier brown-red telemetry pass",
        )
        self.assertEqual(
            self._foreground_hex(empty_style),
            "#ffe4dc",
            "EMPTY label ink should stay on the bright red-white accent",
        )

    def test_zero_run_counters_keep_semantic_hues_instead_of_dead_gray(self):
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=0.0):
            group = display.render()

        header_panel = group.renderables[0]
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable
        header_text_obj = header_renderable
        if isinstance(header_text_obj, Group):
            header_text_obj = header_text_obj.renderables[0]

        emitted_style = self._style_for_substring(header_text_obj, "EMITTED")
        dedup_style = self._style_for_substring(header_text_obj, "DEDUP")
        empty_style = self._style_for_substring(header_text_obj, "EMPTY")

        self.assertNotEqual(self._background_hex(emitted_style), "#2a2d34")
        self.assertNotEqual(self._background_hex(dedup_style), "#2a2d34")
        self.assertNotEqual(self._background_hex(empty_style), "#2a2d34")

    def test_scorebug_uses_sparse_separator_dots_and_foreground_only_digits(self):
        display = self._make_display()
        display.on_session_meta(
            model="qwen3p5-35B-A3B",
            set_label="TRICKY",
            subset_count=6,
        )
        display.on_header("[item 4/6] 15-blue/fr-10b")
        display.on_topic(
            "45s · Grader: 1.0/1.0. Prof: 0.0/1.0.",
            verdict="overshoot",
            grader_score=1.0,
            truth_score=0.0,
            max_points=1.0,
        )

        scorebug_panel = display.render().renderables[1]
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        tally_text_obj = scorebug_renderable.renderables[2]
        tally_value_top = scorebug_renderable.renderables[3]

        self.assertIn(
            "·",
            tally_text_obj.plain,
            "the big-value strip should use sparse separator dots instead of "
            "reverting to blocky gutters",
        )
        self.assertIn("·", tally_value_top.plain)
        self.assertIsNone(
            self._background_hex(self._style_for_substring(tally_text_obj, "·")),
            "separator dots should read as foreground ink, not as another slab",
        )

        expected_on_target = _scorebug_big_value_rows("0.0/1.0")
        on_target_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_on_target[0].strip(),
        )
        self.assertIsNone(
            self._background_hex(on_target_top_style),
            "scorebug numeral strokes should stay foreground-only once the "
            "lighter plate treatment is restored",
        )

    def test_timers_render_as_tall_scorebug_plates_not_small_capsules(self):
        """TOTAL and TURN must render as full three-row
        ``_append_scorebug_big_value_cell`` plates inside the scorebug
        panel, next to ON TARGET / LEFT ON TABLE / BAD CALLS — not as
        small single-row ``_append_scorebug_cell`` capsules in the top
        header band.

        The previous Gauge Saints slice put all five run counters into
        the small header-band capsule treatment. On review that made
        the timer promotion invisible: the eye reads only the three
        existing grading plates as "scoreboard dials" and the small
        header capsules read as unchanged metadata chrome. The fix is
        to promote TOTAL and TURN into the tall scoreboard-plate
        treatment so there is an actual dial-shape promotion. The
        event-count trio (EMITTED / DEDUP / EMPTY) stays as small
        header capsules because the attractor only asked for timer
        promotion.

        Falsifiable shape:
          * TOTAL and TURN labels appear in the scorebug panel's plain
            text, not only in the header panel.
          * The tall-digit three-row glyph signature produced by
            ``_scorebug_big_value_rows`` for the TOTAL value (e.g.
            ``2963``) appears in the scorebug panel rendering, so the
            timer values are actually walking through the big-digit
            renderer rather than being printed in bare text.
          * Each of TOTAL and TURN has an ``on #...`` capsule label
            style matching the ON TARGET / LEFT ON TABLE / BAD CALLS
            label cell idiom.
          * TOTAL and TURN are gone from the header panel plain text —
            no duplicated small-capsule version in the top band.
        """
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=1000.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        # Emit some activity so a realistic render is exercised and
        # no panel collapses to a degenerate empty state.
        display.on_delta("I'm tracing the stoichiometry.")
        display.on_commit("thought")
        display.on_delta("Tracing the stoichiometry setup.", mode="status")
        display.on_commit("status")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=3963.0):
            group = display.render()

        header_panel = group.renderables[0]
        scorebug_panel = group.renderables[1]

        # Collect the scorebug panel plain text from every row so the
        # assertions don't depend on the exact row index of the new
        # timer plates.
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        scorebug_rows = scorebug_renderable.renderables
        scorebug_plain = "\n".join(
            self._normalize_scorebug_texture(row.plain) if isinstance(row, Text) else ""
            for row in scorebug_rows
        )

        # Labels present in the scorebug panel.
        self.assertIn(
            "TOTAL",
            scorebug_plain,
            "TOTAL timer should live as a tall scoreboard-plate label "
            "inside the scorebug panel, not as a small top-band capsule",
        )
        self.assertIn(
            "TURN",
            scorebug_plain,
            "TURN timer should live as a tall scoreboard-plate label "
            "inside the scorebug panel, not as a small top-band capsule",
        )

        # TOTAL value 2963 should walk through _scorebug_big_value_rows,
        # so the top row of the three-row tall-digit signature for
        # "2963" should appear verbatim somewhere in the scorebug plain
        # text. Using the top row is the strongest check because the
        # middle/bottom rows of some glyph pairs collide with other
        # plates' glyphs.
        total_top, _total_middle, _total_bottom = _scorebug_big_value_rows("2963")
        self.assertIn(
            total_top,
            scorebug_plain,
            "TOTAL value 2963 should render through _scorebug_big_value_rows "
            "so the three-row tall-digit glyph signature is present in the "
            "scorebug panel — this is how the plate is distinguished from a "
            "small text capsule",
        )

        # TURN value 2963 (same elapsed because the test header fires
        # at t=1000 and renders at t=3963; turn timer started at the
        # header) should likewise walk through the big-digit renderer.
        turn_top, _turn_middle, _turn_bottom = _scorebug_big_value_rows("2963")
        self.assertIn(
            turn_top,
            scorebug_plain,
            "TURN value should render through _scorebug_big_value_rows too",
        )

        # Extract header panel plain text for counter-survival and
        # no-duplication assertions.
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable
        if isinstance(header_renderable, Group):
            header_renderable = header_renderable.renderables[0]
        header_plain = header_renderable.plain

        # Label cells carry the ``on #...`` capsule style.
        # Find whichever scorebug row carries each label and assert
        # the label style has a background component.
        def _label_style_in_scorebug(label: str) -> str:
            for row in scorebug_rows:
                if not isinstance(row, Text):
                    continue
                if label in row.plain:
                    return self._style_for_substring(row, label)
            raise AssertionError(
                f"no scorebug row carried the label {label!r}"
            )

        self.assertIn(
            "EMITTED  1",
            header_plain,
            "EMITTED counter value (1 from on_commit thought) must survive "
            "in header after timer dial promotion",
        )
        self.assertIn(
            "DEDUP  0",
            header_plain,
            "DEDUP counter value (0, no drops in setup) must survive "
            "in header after timer dial promotion",
        )

        for label in ("TOTAL", "TURN"):
            style = _label_style_in_scorebug(label)
            self.assertIn(
                "on #",
                style,
                f"timer plate label {label!r} should live inside a capsule "
                f"cell (``fg on #bghex`` style), matching the existing "
                f"ON TARGET / LEFT ON TABLE / BAD CALLS label cells",
            )

        total_label_bg = self._background_hex(_label_style_in_scorebug("TOTAL"))
        turn_label_bg = self._background_hex(_label_style_in_scorebug("TURN"))
        on_target_label_bg = self._background_hex(_label_style_in_scorebug("ON TARGET"))
        self.assertEqual(
            {total_label_bg, turn_label_bg, on_target_label_bg},
            {on_target_label_bg},
            "TOTAL and TURN should share the same charcoal label field as the rest of the scorebug row rather than living on isolated blue/orange slabs",
        )

        self.assertNotIn(
            "TOTAL",
            header_plain,
            "TOTAL label should no longer appear in the top header band "
            "once it is promoted to a tall scorebug plate — otherwise the "
            "dial is duplicated",
        )
        self.assertNotIn(
            "TURN",
            header_plain,
            "TURN label should no longer appear in the top header band "
            "once it is promoted to a tall scorebug plate — otherwise the "
            "dial is duplicated",
        )

    def test_timer_promotion_preserves_event_counters_in_header(self):
        """The emitted/dedup event counters must survive in the header
        panel after TOTAL and TURN are promoted to tall scorebug plates.

        This was previously a dead test body: Python silently kept only
        the second definition of
        ``test_timers_render_as_tall_scorebug_plates_not_small_capsules``,
        so these assertions (emitted "1" and dedup "2" still in header,
        plus label background equality) never ran. Renamed to restore
        coverage.
        """
        display = self._make_display()

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=1000.0):
            display.on_header("[item 1/6] 15-blue/fr-1")
        # Emit some activity so a realistic render is exercised and
        # no panel collapses to a degenerate empty state.
        display.on_delta("I'm tracing the stoichiometry.")
        display.on_commit("thought")
        display.on_delta("Tracing the stoichiometry setup.", mode="status")
        display.on_commit("status")

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=3963.0):
            group = display.render()

        header_panel = group.renderables[0]
        scorebug_panel = group.renderables[1]

        # Collect the scorebug panel plain text from every row so the
        # assertions don't depend on the exact row index of the new
        # timer plates.
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        scorebug_rows = scorebug_renderable.renderables
        scorebug_plain = "\n".join(
            row.plain if isinstance(row, Text) else ""
            for row in scorebug_rows
        )

        # Labels present in the scorebug panel.
        self.assertIn(
            "TOTAL",
            scorebug_plain,
            "TOTAL timer should live as a tall scoreboard-plate label "
            "inside the scorebug panel, not as a small top-band capsule",
        )
        self.assertIn(
            "TURN",
            scorebug_plain,
            "TURN timer should live as a tall scoreboard-plate label "
            "inside the scorebug panel, not as a small top-band capsule",
        )

        # TOTAL value 2963 should walk through _scorebug_big_value_rows,
        # so the top row of the three-row tall-digit signature for
        # "2963" should appear verbatim somewhere in the scorebug plain
        # text. Using the top row is the strongest check because the
        # middle/bottom rows of some glyph pairs collide with other
        # plates' glyphs.
        total_top, _total_middle, _total_bottom = _scorebug_big_value_rows("2963")
        self.assertIn(
            total_top,
            scorebug_plain,
            "TOTAL value 2963 should render through _scorebug_big_value_rows "
            "so the three-row tall-digit glyph signature is present in the "
            "scorebug panel — this is how the plate is distinguished from a "
            "small text capsule",
        )

        # TURN value 2963 (same elapsed because the test header fires
        # at t=1000 and renders at t=3963; turn timer started at the
        # header) should likewise walk through the big-digit renderer.
        turn_top, _turn_middle, _turn_bottom = _scorebug_big_value_rows("2963")
        self.assertIn(
            turn_top,
            scorebug_plain,
            "TURN value should render through _scorebug_big_value_rows too",
        )

        # Label cells carry the ``on #...`` capsule style.
        # Find whichever scorebug row carries each label and assert
        # the label style has a background component.
        def _label_style_in_scorebug(label: str) -> str:
            for row in scorebug_rows:
                if not isinstance(row, Text):
                    continue
                if label in row.plain:
                    return self._style_for_substring(row, label)
            raise AssertionError(
                f"no scorebug row carried the label {label!r}"
            )

        for label in ("TOTAL", "TURN"):
            style = _label_style_in_scorebug(label)
            self.assertIn(
                "on #",
                style,
                f"timer plate label {label!r} should live inside a capsule "
                f"cell (``fg on #bghex`` style), matching the existing "
                f"ON TARGET / LEFT ON TABLE / BAD CALLS label cells",
            )

        # Header panel no longer carries small TOTAL/TURN capsules —
        # the promotion replaces the top-band treatment rather than
        # duplicating it.
        header_renderable = header_panel.renderable
        if isinstance(header_renderable, Align):
            header_renderable = header_renderable.renderable
        if isinstance(header_renderable, Group):
            header_renderable = header_renderable.renderables[0]
        header_plain = header_renderable.plain
        self.assertNotIn(
            "TOTAL",
            header_plain,
            "TOTAL label should no longer appear in the top header band "
            "once it is promoted to a tall scorebug plate — otherwise the "
            "dial is duplicated",
        )
        self.assertNotIn(
            "TURN",
            header_plain,
            "TURN label should no longer appear in the top header band "
            "once it is promoted to a tall scorebug plate — otherwise the "
            "dial is duplicated",
        )

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

    def test_all_post_header_rows_use_group_depth_for_fade(self):
        self.assertEqual(_render_layer_index("line", 2), 2)
        self.assertEqual(
            _render_layer_index("topic", 2),
            2,
            "topic rows should participate in the local depth fade so the "
            "whole block below a header settles as it descends",
        )
        self.assertEqual(_render_layer_index("basis", 3), 3)
        self.assertEqual(_render_layer_index("checkpoint", 4), 4)
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

    def test_secondary_history_overlay_flips_target_group_parity_each_pass(self):
        import scripts.narrator_reader as module

        self.assertTrue(
            hasattr(module, "_history_secondary_phase"),
            "history shimmer should expose a helper for the quieter second pass so parity-flip timing can stay explicit and testable",
        )
        self.assertTrue(
            hasattr(module, "_history_secondary_group_weight"),
            "history shimmer should expose a helper that selects which heading-group parity gets the secondary pass on a given cycle",
        )

        pass0_index, pass0_phase = module._history_secondary_phase(0.0)
        pass1_index, _pass1_phase = module._history_secondary_phase(
            module._HISTORY_GROUP_SECONDARY_CYCLE_S
        )

        self.assertEqual(
            pass0_index,
            0,
            "the secondary pass should begin on its first parity at time zero",
        )
        self.assertNotAlmostEqual(
            pass0_phase,
            0.0,
            places=6,
            msg="the secondary pass should be phase-offset from the primary shimmer instead of riding exactly on top of it",
        )
        self.assertEqual(
            module._history_secondary_group_weight(0, pass0_index),
            module._HISTORY_GROUP_SECONDARY_BLEND,
            "the first secondary pass should favor the first heading-group parity",
        )
        self.assertEqual(
            module._history_secondary_group_weight(1, pass0_index),
            0.0,
            "neighboring heading groups should not both get the quieter second pass in the same cycle",
        )
        self.assertEqual(
            pass1_index,
            1,
            "one full secondary cycle later, the opposite parity should be active",
        )
        self.assertEqual(
            module._history_secondary_group_weight(0, pass1_index),
            0.0,
            "the previously favored parity should stand down on the next pass",
        )
        self.assertEqual(
            module._history_secondary_group_weight(1, pass1_index),
            module._HISTORY_GROUP_SECONDARY_BLEND,
            "the alternating secondary pass should flip to the neighboring heading-group parity on the next cycle",
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

        for msg_type in ("wrap_up", "basis", "review_marker", "end"):
            self.assertTrue(_message_requires_immediate_refresh(msg_type))

    def test_inline_focus_preview_uses_reduced_animation_cadence(self):
        display = self._make_display()
        display._inline_images_supported = True
        display._kitty_graphics_supported = False
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-10b",
            source="mock_tricky",
        )

        self.assertTrue(
            display.should_animate(),
            "steady inline-image preview should still animate, just at a calmer cadence so the surface does not feel frozen",
        )
        self.assertEqual(
            display.target_animation_fps(),
            _ACTIVE_ANIMATION_FPS,
            "steady preview should run at the same animation cadence as "
            "active mode now that the band is precomposed — the original "
            "throttle made the shimmer look sluggish at 10fps",
        )
        self.assertFalse(
            display.should_refresh_on_event("delta"),
            "inline preview mode should not repaint on every streaming token",
        )
        self.assertTrue(
            display.should_refresh_on_event("commit"),
            "inline preview mode must still repaint on structural boundaries so the visible state keeps moving",
        )
        self.assertTrue(
            display.should_refresh_on_event("focus_preview"),
            "a newly arrived preview image still needs an immediate paint in event-driven mode",
        )
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

    def test_secondary_history_overlay_is_present_but_quieter_than_primary(self):
        base_text = Text()
        _apply_shimmer(base_text, "A", "header", 0, phase_override=0.0)

        primary_text = Text()
        _apply_shimmer(primary_text, "A", "header", 0, phase_override=(12 / 13))

        secondary_text = Text()
        _apply_shimmer(
            secondary_text,
            "A",
            "header",
            0,
            phase_override=0.0,
            secondary_phase_override=(12 / 13),
            secondary_peak_weight=narrator_reader._HISTORY_GROUP_SECONDARY_BLEND,
        )

        base_luma = self._hex_luminance(base_text.spans[0].style)
        primary_boost = self._hex_luminance(primary_text.spans[0].style) - base_luma
        secondary_boost = self._hex_luminance(secondary_text.spans[0].style) - base_luma

        self.assertGreater(
            secondary_boost,
            0.0,
            "the alternating overlay should create a real visible lift when its own pass is on the character",
        )
        self.assertLess(
            secondary_boost,
            primary_boost,
            "the alternating overlay should stay quieter than the main shimmer pass instead of competing with it",
        )

    def test_secondary_history_overlay_targets_header_group_and_child_rows(self):
        display = self._make_display()
        wrap_width = display._compute_wrap_width()
        self.assertIsNotNone(wrap_width)
        secondary_head_col = int(round(
            narrator_reader._HISTORY_GROUP_SECONDARY_PHASE_OFFSET
            * (wrap_width + narrator_reader._SHIMMER_WIDTH)
            - narrator_reader._SHIMMER_WIDTH
        ))
        header_filler = "x" * max(0, secondary_head_col - 13)
        line_filler = "x" * max(0, secondary_head_col - 4)
        display.history.append(
            ("header", "[item 1/2] " + header_filler + "OLDERHDR", None)
        )
        display.history.append(
            ("line", line_filler + "OLDERLINE", 0)
        )
        display.history.append(
            ("header", "[item 2/2] " + header_filler + "NEWHDR", None)
        )
        display.history.append(
            ("line", line_filler + "NEWLINE", 0)
        )

        def _render_history_at(now_s: float) -> Text:
            with mock.patch.object(
                narrator_reader.time,
                "monotonic",
                return_value=now_s,
            ):
                group = display.render()
            history_panel = group.renderables[-1]
            return history_panel.renderable

        first_pass = _render_history_at(0.0)
        second_pass = _render_history_at(
            narrator_reader._HISTORY_GROUP_SECONDARY_CYCLE_S
        )

        new_header_first = self._hex_luminance(
            self._style_for_substring(first_pass, "NEWHDR")
        )
        new_header_second = self._hex_luminance(
            self._style_for_substring(second_pass, "NEWHDR")
        )
        new_line_first = self._hex_luminance(
            self._style_for_substring(first_pass, "NEWLINE")
        )
        new_line_second = self._hex_luminance(
            self._style_for_substring(second_pass, "NEWLINE")
        )
        old_header_first = self._hex_luminance(
            self._style_for_substring(first_pass, "OLDERHDR")
        )
        old_header_second = self._hex_luminance(
            self._style_for_substring(second_pass, "OLDERHDR")
        )
        old_line_first = self._hex_luminance(
            self._style_for_substring(first_pass, "OLDERLINE")
        )
        old_line_second = self._hex_luminance(
            self._style_for_substring(second_pass, "OLDERLINE")
        )

        self.assertGreater(
            new_header_first,
            new_header_second,
            "the first secondary pass should brighten the targeted heading group's header more than the same header on the opposite-parity pass",
        )
        self.assertGreater(
            new_line_first,
            new_line_second,
            "the selected heading group's subordinate row should ride along with the secondary pass instead of leaving the effect stranded on the header only",
        )
        self.assertGreater(
            old_header_second,
            old_header_first,
            "when the parity flips, the neighboring heading group's header should take the quieter second pass instead",
        )
        self.assertGreater(
            old_line_second,
            old_line_first,
            "the parity flip should also carry across the neighboring heading group's child row, not just its header",
        )

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

    def test_kitty_focus_preview_also_uses_reduced_animation_cadence(self) -> None:
        display = self._make_display()
        display._kitty_graphics_supported = True
        display._inline_images_supported = False
        display.on_focus_preview(
            self._make_png(),
            label="15-blue/fr-10b",
            source="mock_tricky",
        )

        self.assertTrue(
            display.should_animate(),
            "steady kitty-image preview should still animate, just at a calmer cadence so the surface does not feel frozen",
        )
        self.assertEqual(
            display.target_animation_fps(),
            _ACTIVE_ANIMATION_FPS,
            "steady kitty preview should run at the same animation cadence "
            "as active mode now that the band is precomposed",
        )

    def test_live_update_gates_global_clear_on_geometry_change(self) -> None:
        source = Path("scripts/narrator_reader.py").read_text()
        live_update = source.split("def _live_update():", 1)[1].split(
            "with Live(",
            1,
        )[0]

        self.assertIn(
            "_live_frame_requires_full_clear(",
            source,
            "steady preview mode should decide explicitly when a global alt-screen clear is necessary instead of hard-clearing every frame",
        )
        self.assertIn(
            "_live_frame_prefix(_last_paint_size, paint_size)",
            live_update,
            "the live paint loop should route its prefix escape sequence "
            "through the helper that distinguishes full geometry clears from "
            "stable-paint tail clears",
        )

    def test_main_routes_message_refreshes_through_live_update(self) -> None:
        source = Path("scripts/narrator_reader.py").read_text()
        main_tail = source.split("with Live(", 1)[1]

        self.assertNotIn(
            "live.update(display.render(), refresh=True)",
            main_tail,
            "once the reader owns repaint semantics via _live_update(), "
            "the fifo message loop must not bypass that helper with raw "
            "live.update(display.render(), refresh=True) calls or frames "
            "will stack instead of repainting in place",
        )
        self.assertIn(
            "_live_update()",
            main_tail,
            "the fifo message loop should route immediate refreshes "
            "through the manual live-update helper that owns cursor-home, "
            "buffering, and Kitty sequencing",
        )

    def test_main_keeps_control_sequences_on_console_file_path(self) -> None:
        source = Path("scripts/narrator_reader.py").read_text()
        main_tail = source.split("def main() -> int:", 1)[1]

        self.assertNotIn(
            "sys.stdout.write(",
            main_tail,
            "the spawned Paint Dry reader must keep alt-screen and cursor-home "
            "control sequences on the same console.file path as the buffered "
            "frame write; mixing raw sys.stdout writes with console.file writes "
            "can leak visible '[H' text instead of repainting in place",
        )

    def test_scroll_thread_reads_raw_tty_bytes_instead_of_buffered_stdin_text(self) -> None:
        source = Path("scripts/narrator_reader.py").read_text()
        main_tail = source.split("def main() -> int:", 1)[1]

        self.assertIn(
            "_read_tty_key(tty_fd)",
            main_tail,
            "the live history controls should read raw bytes from the tty file "
            "descriptor; relying on sys.stdin.read(1) through the buffered text "
            "wrapper is too fragile for spawned cbreak-mode terminals",
        )
        self.assertNotIn(
            "sys.stdin.read(1)",
            main_tail,
            "the scroll thread should not use the buffered sys.stdin text path "
            "once the reader is in cbreak mode",
        )

    def test_main_opens_dev_tty_for_interactive_controls(self) -> None:
        source = Path("scripts/narrator_reader.py").read_text()
        main_tail = source.split("def main() -> int:", 1)[1]

        self.assertIn(
            'os.open("/dev/tty", os.O_RDONLY)',
            main_tail,
            "the live reader should acquire its interactive key path from "
            "/dev/tty directly instead of trusting inherited stdin plumbing",
        )
        self.assertNotIn(
            "stdin_fd = sys.stdin.fileno()",
            main_tail,
            "the live history controls should not rely on inherited "
            "sys.stdin to discover the interactive TTY fd",
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
