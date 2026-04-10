from __future__ import annotations

import math
import time
import unittest
from unittest import mock
from pathlib import Path

import fitz
from rich.align import Align
from rich.console import Console, Group
from rich.text import Text

from scripts.narrator_reader import (
    _ACTIVE_ANIMATION_FPS,
    _DEFAULT_TERMINAL_CELL_ASPECT,
    _build_focus_preview_pixels,
    _build_iterm2_inline_image_sequence,
    _build_kitty_place_sequence,
    _build_kitty_transmit_chunks,
    _compute_inline_image_cell_dimensions,
    _focus_preview_budget,
    _KITTY_IMAGE_ID,
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
    _otsu_threshold,
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
        # transmit is done by the caller (on_focus_preview) directly
        # to stdout before the next Rich frame. The renderable's
        # job is only to emit the tiny place-by-ID command at the
        # correct cursor position inside its own border frame, on
        # every single Rich refresh, at essentially zero cost
        # (~30 bytes per frame).
        from rich.console import Console

        renderable = FocusPreviewKittyImage(
            image_id=1,
            image_pixel_width=1600,
            image_pixel_height=900,
            terminal_cell_aspect=2.1,
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

    def test_focus_preview_kitty_image_shrinks_on_narrow_console(self):
        # The Kitty box must be computed from the currently available
        # width budget, not cached at construction time. A narrower
        # budget must shrink the cell box; a wider budget must let it
        # grow again while keeping the aspect stable.
        renderable = FocusPreviewKittyImage(
            image_id=1,
            image_pixel_width=3000,
            image_pixel_height=1000,
            terminal_cell_aspect=2.1,
            title="test",
        )

        c_narrow, r_narrow = renderable._compute_box(50)
        self.assertLess(
            c_narrow,
            120,
            "narrow width budget must materially shrink the Kitty box rather than keeping a wide-console size",
        )

        c_wide, r_wide = renderable._compute_box(200)
        self.assertGreater(
            c_wide,
            c_narrow,
            "wider width budget must let the box grow larger than the narrow case",
        )

        ratio_narrow = c_narrow / r_narrow
        ratio_wide = c_wide / r_wide
        self.assertAlmostEqual(
            ratio_narrow,
            ratio_wide,
            delta=1.0,
            msg="resizing must preserve image aspect ratio",
        )

    def test_focus_preview_kitty_image_emits_place_every_render(self):
        # Rich Live clears the image region between frames, so we
        # must re-emit the place command on every render. The place
        # command is tiny and WezTerm doesn't re-parse anything, so
        # this is cheap and flicker-free.
        from rich.console import Console

        renderable = FocusPreviewKittyImage(
            image_id=1,
            image_pixel_width=1600,
            image_pixel_height=900,
            terminal_cell_aspect=2.1,
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
        self.assertEqual(first.count("\x1b_Ga=p"), 1)
        self.assertEqual(second.count("\x1b_Ga=p"), 1)

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

    def test_render_keeps_topic_above_core_issue_in_topicless_neighbor_case(self):
        display = self._make_display()
        display.history.append(
            ("header", "[item 2/15] 15-blue/fr-11c (exact_match, 0.5 pts)", None)
        )
        display.history.append(
            ("header", "[item 1/15] 15-blue/fr-10b (numeric, 1.0 pts)", None)
        )
        display.history.append(
            (
                "checkpoint",
                "Core issue: Student used frequency from part (a) in denominator, but question asked for energy per mole of photons.",
                None,
            )
        )
        display.history.append(
            (
                "topic",
                "183s · 15-blue/fr-10b: grader did not commit to a score (truncated)",
                "match",
            )
        )

        history_text = display.render().renderables[-1].renderable.plain
        topic_pos = history_text.index(
            "183s  ·  15-blue/fr-10b: grader did not commit to a score (truncated)"
        )
        core_issue_pos = history_text.index(
            "Core issue: Student used frequency from part (a) in denominator, but question asked for energy per mole of photons."
        )

        self.assertLess(
            topic_pos,
            core_issue_pos,
            "even when a neighbor item above is header-only, the visible topic/timing line must still sit above Core issue rows for its own item",
        )

    def test_render_drops_leading_orphan_rows_until_next_header(self):
        display = self._make_display()

        orphaned_entries = [
            (
                ("checkpoint", "Core issue: orphaned top row should not render.", None),
                False,
                0,
            ),
            (
                ("topic", "42s · orphaned topic should not render.", "match"),
                False,
                0,
            ),
            (
                ("header", "[item 4/15] 27-blue-2023/fr-3 (balanced_equation, 4.0 pts)", None),
                False,
                0,
            ),
            (
                ("topic", "38s · Grader: 4/4 (Correct net ionic equation).", "match"),
                True,
                1,
            ),
        ]

        with mock.patch.object(
            display,
            "_viewport_display_entries",
            return_value=orphaned_entries,
        ):
            history_text = display.render().renderables[-1].renderable.plain

        self.assertNotIn(
            "orphaned top row should not render",
            history_text,
            "the history pane should never begin with a mid-item body row when the owning header is out of view",
        )
        self.assertNotIn(
            "orphaned topic should not render",
            history_text,
            "the history pane should trim leading topic/body fragments until the next visible header",
        )
        self.assertTrue(
            history_text.lstrip().startswith("─ [item 4/15]"),
            "once leading orphans are trimmed, the first visible history line should be the next item header",
        )

    def test_truncated_topic_renders_dimmer_than_normal_match_topic(self):
        regular = self._make_display()
        regular.history.append(("header", "[item 1/6] first", None))
        regular.history.append(
            ("topic", "42s · Grader: 2/2 (Correct stoichiometric calculation with proper charges.)", "match")
        )
        regular_text = regular.render().renderables[-1].renderable
        regular_style = self._style_for_substring(
            regular_text,
            "Grader: 2/2 (Correct stoichiometric calculation with proper charges.)",
        )

        truncated = self._make_display()
        truncated.history.append(("header", "[item 1/6] first", None))
        truncated.history.append(
            ("topic", "190s · 15-blue/fr-10b: grader did not commit to a score (truncated)", "match")
        )
        truncated_text = truncated.render().renderables[-1].renderable
        truncated_style = self._style_for_substring(
            truncated_text,
            "15-blue/fr-10b: grader did not commit to a score (truncated)",
        )

        self.assertLess(
            self._hex_luminance(truncated_style),
            self._hex_luminance(regular_style),
            "truncated fallback topics should read as dimmer placeholders than ordinary match verdict lines",
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

    def test_body_rows_below_header_return_to_bone_family_and_descend_in_value(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("topic", "45s  ·  Grader: 1/2. Prof: 1/2.", "undershoot"))
        display.history.append(("basis", "Correct setup, lost credit for octet violation.", None))
        display.history.append(("checkpoint", "Core issue: ozone drawing misses resonance.", 0))
        display.history.append(("line", "I'm tracing the ozone structure.", 0))

        group = display.render()
        history_text = group.renderables[-1].renderable

        basis_style = self._style_for_substring(
            history_text,
            "Correct setup, lost credit for octet violation.",
        )
        checkpoint_style = self._style_for_substring(
            history_text,
            "Core issue: ozone drawing misses resonance.",
        )
        line_style = self._style_for_substring(
            history_text,
            "I'm tracing the ozone structure.",
        )

        basis_rgb = self._rgb_from_hex(basis_style.split()[-1])
        checkpoint_rgb = self._rgb_from_hex(checkpoint_style.split()[-1])
        line_rgb = self._rgb_from_hex(line_style.split()[-1])
        self.assertGreaterEqual(
            basis_rgb[0],
            basis_rgb[1],
            "basis rows under the heading should return to a red-led bone family instead of green-led body ink",
        )
        self.assertGreater(
            checkpoint_rgb[1],
            checkpoint_rgb[0],
            "the second descendant should now carry the moss alternate lane instead of staying bone-on-bone",
        )
        self.assertLess(
            self._hex_luminance(checkpoint_style),
            self._hex_luminance(basis_style),
            "body descendants should keep the old downward fade instead of sitting on one flat checkpoint tier",
        )
        self.assertLess(
            self._hex_luminance(line_style),
            self._hex_luminance(checkpoint_style),
            "deeper narrator rows should continue dimming below the structured rows",
        )

    def test_second_body_descendant_shifts_toward_moss_without_alternating_item_start(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("topic", "45s  ·  Grader: 1/2. Prof: 1/2.", "undershoot"))
        display.history.append(("basis", "Warm anchor row.", None))
        display.history.append(("checkpoint", "Second descendant should tilt moss.", 0))

        history_text = display.render().renderables[-1].renderable
        basis_style = self._style_for_substring(history_text, "Warm anchor row.")
        checkpoint_style = self._style_for_substring(
            history_text,
            "Second descendant should tilt moss.",
        )
        basis_rgb = self._rgb_from_hex(basis_style.split()[-1])
        checkpoint_rgb = self._rgb_from_hex(checkpoint_style.split()[-1])

        self.assertGreaterEqual(
            basis_rgb[0],
            basis_rgb[1],
            "the first descendant under each item should stay warm instead of alternating item starts",
        )
        self.assertGreater(
            checkpoint_rgb[1],
            checkpoint_rgb[0],
            "the second descendant should lean mossward so short blocks still show the alternate lane",
        )

    def test_no_topic_items_still_alternate_warm_then_moss(self):
        display = self._make_display()
        display.history.append(("header", "[item 1/6] first", None))
        display.history.append(("basis", "Warm anchor row.", None))
        display.history.append(("checkpoint", "Second descendant should tilt moss.", 0))
        display.history.append(("line", "Third descendant should return warm.", 0))

        history_text = display.render().renderables[-1].renderable
        basis_style = self._style_for_substring(history_text, "Warm anchor row.")
        checkpoint_style = self._style_for_substring(
            history_text,
            "Second descendant should tilt moss.",
        )
        line_style = self._style_for_substring(
            history_text,
            "Third descendant should return warm.",
        )
        basis_rgb = self._rgb_from_hex(basis_style.split()[-1])
        checkpoint_rgb = self._rgb_from_hex(checkpoint_style.split()[-1])
        line_rgb = self._rgb_from_hex(line_style.split()[-1])

        self.assertGreaterEqual(
            basis_rgb[0],
            basis_rgb[1],
            "without a topic row, the first visible body descendant should still stay warm",
        )
        self.assertGreater(
            checkpoint_rgb[1],
            checkpoint_rgb[0],
            "without a topic row, the second visible body descendant should still tip mossward",
        )
        self.assertGreaterEqual(
            line_rgb[0],
            line_rgb[1],
            "the third visible body descendant should come back to the warm lane",
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
        normalized_top = self._normalize_scorebug_texture(tally_value_top.plain)
        normalized_mid = self._normalize_scorebug_texture(tally_value_mid.plain)
        normalized_bottom = self._normalize_scorebug_texture(tally_value_bottom.plain)
        self.assertIn(expected_on_target[0], normalized_top)
        self.assertIn(expected_on_target[1], normalized_mid)
        self.assertIn(expected_on_target[2], normalized_bottom)
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
        left_label_style = self._style_for_substring(tally_text_obj, "LEFT ON TABLE")
        bad_label_style = self._style_for_substring(tally_text_obj, "BAD CALLS")
        on_target_label_bg = self._background_hex(on_target_label_style)
        left_label_bg = self._background_hex(left_label_style)
        bad_label_bg = self._background_hex(bad_label_style)
        self.assertEqual(
            {on_target_label_bg, left_label_bg, bad_label_bg},
            {on_target_label_bg},
            "the tally labels should sit on one shared charcoal field instead of three color-coded boards",
        )
        self.assertGreaterEqual(
            len(
                {
                    self._foreground_hex(on_target_label_style),
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
        self.assertEqual(
            tally_text_obj.plain.index("LEFT ON TABLE"),
            on_target_start + cell_width + separator_width,
            "LEFT ON TABLE label should start at the left edge of its score cell",
        )
        self.assertEqual(
            tally_text_obj.plain.index("BAD CALLS"),
            on_target_start + (2 * cell_width) + (2 * separator_width),
            "BAD CALLS label should start at the left edge of its score cell",
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
        on_target_top_bg = self._background_hex(
            self._style_for_normalized_scorebug_substring(
                tally_value_top, expected_on_target[0].strip()
            )
        )
        left_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_left[0].strip(),
        )
        bad_top_style = self._style_for_normalized_scorebug_substring(
            tally_value_top,
            expected_bad[0].strip(),
        )
        left_top_bg = self._background_hex(left_top_style)
        bad_top_bg = self._background_hex(bad_top_style)
        self.assertEqual(
            {on_target_top_bg, left_top_bg, bad_top_bg},
            {on_target_top_bg},
            "the scorebug should go back to one unified field treatment instead of three tinted slabs",
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
        on_target_mid_bg = self._background_hex(
            self._style_for_normalized_scorebug_substring(
                tally_value_mid, expected_on_target[1].strip()
            )
        )
        on_target_bottom_bg = self._background_hex(
            self._style_for_normalized_scorebug_substring(
                tally_value_bottom, expected_on_target[2].strip()
            )
        )
        bg_lumas = [
            self._hex_luminance(on_target_top_bg),
            self._hex_luminance(on_target_mid_bg),
            self._hex_luminance(on_target_bottom_bg),
        ]
        self.assertLess(
            max(bg_lumas) - min(bg_lumas),
            24,
            "the score field should read as a flatter low-contrast texture, not a strong descending row gradient",
        )
        self.assertGreater(
            max(bg_lumas) - min(bg_lumas),
            6,
            "the score field should still have some subtle tonal breathing room instead of collapsing to one dead flat slab",
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
        self.assertTrue(
            all(
                self._hex_luminance(
                    self._foreground_hex(style)
                ) > 635
                for style in {on_target_top_strong, left_top_style, bad_top_style}
            ),
            "the main numeral strokes should now carry more white weight so the scorebug can sit in the same bold white language as the surrounding interface",
        )
        self.assertEqual(
            on_target_spans[2].style,
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

        # Emit some activity so all three event counters are nonzero.
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
            self.assertIn(
                label,
                header_plain,
                f"run counter {label!r} should render as a scoreboard dial "
                f"label, not as flat telemetry",
            )

        for legacy in ("total=", "turn=", "emitted=", "dedup=", "empty="):
            self.assertNotIn(legacy, header_plain)

        self.assertIn("1", header_plain)
        self.assertIn("2", header_plain)

        for label in ("EMITTED", "DEDUP", "EMPTY"):
            style = self._style_for_substring(header_text_obj, label)
            self.assertIn("on #", style)

        drops_panel = group.renderables[-1]
        drops_title = drops_panel.title or ""
        self.assertNotIn("dedup=", drops_title)
        self.assertNotIn("empty=", drops_title)

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
            "1",
            header_plain,
            "emitted count (1) should still be visible after dial treatment",
        )
        self.assertIn(
            "2",
            header_plain,
            "dedup count (2) should still be visible after dial treatment",
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
        self.assertLess(
            _history_tier_dim_factor(1),
            0.92,
            "the first visible drop should be strong enough to read in the common 2-3 row item blocks",
        )

    def test_only_reasoning_lines_use_group_depth_for_fade(self):
        self.assertEqual(
            _render_layer_index("line", 2),
            1,
            "body fade should now start counting after the topic row so the first descendant is the first visible drop",
        )
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

        for msg_type in ("wrap_up", "basis", "review_marker", "end"):
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

    def test_main_returns_error_on_fifo_eof_before_end_event(self) -> None:
        import scripts.narrator_reader as module

        class _FakeFifo:
            def read(self, _size: int) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeLive:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self, *_args, **_kwargs) -> None:
                return None

        class _FakeDisplay:
            def __init__(self, *args, **kwargs):
                self.session_ended = False

            def render(self):
                return Text("frame")

            def should_animate(self) -> bool:
                return False

        with mock.patch.object(module.sys, "argv", ["narrator_reader.py", "/tmp/mock.fifo"]):
            with mock.patch.object(module.Path, "exists", return_value=True):
                with mock.patch.object(module.os, "open", return_value=123):
                    with mock.patch.object(module.os, "fdopen", return_value=_FakeFifo()):
                        with mock.patch.object(module, "Live", _FakeLive):
                            with mock.patch.object(module, "PaintDryDisplay", _FakeDisplay):
                                with mock.patch.object(module.termios, "tcgetattr", side_effect=OSError):
                                    stderr = mock.Mock()
                                    with mock.patch.object(module.sys, "stderr", stderr):
                                        result = module.main()

        self.assertEqual(
            result,
            1,
            "reader should fail loudly if the FIFO closes before an end event arrives",
        )
        printed = "".join(
            call.args[0]
            for call in stderr.write.call_args_list
            if call.args
        )
        self.assertIn(
            "unexpected FIFO EOF before end event",
            printed,
            "reader should explain why it exited instead of disappearing cleanly",
        )


if __name__ == "__main__":
    unittest.main()
