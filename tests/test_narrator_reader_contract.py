from __future__ import annotations

import math
import time
import unittest
from unittest import mock

import fitz
from rich.align import Align
from rich.console import Console, Group
from rich.text import Text

from scripts.narrator_reader import (
    _ACTIVE_ANIMATION_FPS,
    _build_focus_preview_pixels,
    _build_iterm2_inline_image_sequence,
    _compute_inline_image_cell_dimensions,
    _focus_preview_budget,
    _render_focus_preview_pixels,
    _scaled_preview_size,
    _supports_inline_images,
    _SESSION_END_ANIMATION_LINGER_S,
    _VISIBLE_HISTORY_ROWS,
    FocusPreviewInlineImage,
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

    def test_focus_preview_panel_renders_between_live_and_history(self):
        # Half-block fallback path — test the panel-title ordering.
        display = self._make_display()
        display._inline_images_supported = False
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

    def test_inline_focus_preview_renders_between_live_and_history(self):
        # Inline image path — same ordering invariant as above but
        # the renderable is a FocusPreviewInlineImage rather than a
        # Panel with a title attribute. Verify positional ordering
        # within the Group.
        display = self._make_display()
        display._inline_images_supported = True
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
        # terminal cell aspect ~2:1 (tall), the cell_width should be
        # roughly 18 * 3 * 2 = 108 cells.
        cw, ch = _compute_inline_image_cell_dimensions(
            900, 300, max_cell_height=18, max_cell_width=200
        )
        self.assertEqual(ch, 18)
        self.assertGreaterEqual(cw, 90)
        self.assertLessEqual(cw, 120)

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

    def test_focus_preview_inline_image_emits_escape_sequence_only_on_first_render(self):
        # Regression guard against the "Rich redraws at 24 FPS and
        # re-emits the full base64 PNG on every tick, causing
        # seizure-grade flicker" bug. After the first call to
        # __rich_console__, subsequent calls must yield cursor-
        # forwards and borders but NOT the iTerm2 image escape
        # sequence. The terminal's cell buffer retains the image
        # pixels from the first paint because cursor-forward
        # doesn't write to cells.
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
            0,
            "subsequent renders must NOT re-emit the escape sequence "
            "(it causes re-rasterization flicker at narrator refresh rate)",
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

    def test_render_places_scorebug_above_header_when_model_known(self):
        display = self._make_display()
        display.current_model = "qwen3p5-35B-A3B"
        display.on_header("[item 3/6] 15-blue/fr-5b")

        group = display.render()

        scorebug_panel = group.renderables[0]
        header_panel = group.renderables[1]
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

        scorebug_panel = group.renderables[0]
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
        self.assertEqual(
            tally_value_top.spans[2].style,
            on_target_top_strong,
            "top-row horizontal bars should stay on the strong stroke tier so the numerals read chunkier",
        )

    def test_scorebug_big_value_rows_render_three_line_scoreboard_digits(self):
        top, middle, bottom = _scorebug_big_value_rows("2.0/9.0")

        self.assertEqual(len(top), len(middle))
        self.assertEqual(len(middle), len(bottom))
        self.assertIn("╔", top)
        self.assertIn("║", middle)
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

    def test_scorebug_four_glyph_keeps_a_straight_right_stem(self):
        top, middle, bottom = _scorebug_big_value_rows("4.0")

        self.assertIn("║ ║", top)
        self.assertIn("╚═╣", middle)
        self.assertIn(
            "  ║",
            bottom,
            "the 4 glyph should keep a straight right stem on the bottom row instead of flaring into a T-shape",
        )

    def test_scorebug_one_glyph_uses_upper_cap_and_full_stem(self):
        top, middle, bottom = _scorebug_big_value_rows("1.0")

        self.assertIn(
            "╔╗ ",
            top,
            "the 1 glyph should carry its serif/cap in the upper row instead of reading as a thin post with all the weight at the bottom",
        )
        self.assertIn(" ║ ", middle)
        self.assertIn(" ║ ", bottom)

    def test_scorebug_skinny_family_packs_tightly_with_a_fuller_one_cap(self):
        top, middle, bottom = _scorebug_big_value_rows("17.0")

        self.assertIn(
            "╔╗ ╔═╗",
            top,
            "the 1 and 7 should read as one dense scorebug field, with the 1 carrying a fuller top cap instead of a sparse left tick",
        )
        self.assertNotIn("╔╗  ╔═╗", top)
        self.assertIn(" ║ ╔╝", middle)
        self.assertIn(" ║ ║", bottom)

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

        scorebug_panel = group.renderables[0]
        scorebug_text = _extract_plain(scorebug_panel.renderable)
        scorebug_renderable = scorebug_panel.renderable
        if isinstance(scorebug_renderable, Align):
            scorebug_renderable = scorebug_renderable.renderable
        spacer_row = scorebug_renderable.renderables[1]
        tally_value_top = scorebug_renderable.renderables[3]
        tally_value_mid = scorebug_renderable.renderables[4]
        tally_value_bottom = scorebug_renderable.renderables[5]
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
            header_text = _extract_plain(display.render().renderables[1].renderable)
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
            header_text = _extract_plain(display.render().renderables[1].renderable)
        self.assertIn("turn=5s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=206.0):
            display.on_delta("Rechecking")
        display.on_drop("dedup", "Rechecking the same unit conversion.")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=208.0):
            header_text = _extract_plain(display.render().renderables[1].renderable)
        self.assertIn("turn=8s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=209.0):
            display.on_delta("Tracing")
        display.on_rollback_live()
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=211.0):
            header_text = _extract_plain(display.render().renderables[1].renderable)
        self.assertIn("turn=11s", header_text)

        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=212.0):
            display.on_header("[item 2/6] 15-blue/fr-2")
        with mock.patch("scripts.narrator_reader.time.monotonic", return_value=215.0):
            header_text = _extract_plain(display.render().renderables[1].renderable)
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
