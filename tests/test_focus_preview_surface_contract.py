from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FocusPreviewSurfaceContract(unittest.TestCase):
    def test_preview_renderer_does_not_split_rgb_interpolation_policy(self):
        renderer = _load_module("focus_preview_renderer", "scripts/focus_preview_renderer.py")

        self.assertEqual(
            renderer._interp_rgb((13, 13, 13), (10, 10, 10), 0.5),
            renderer._lerp_rgb((13, 13, 13), (10, 10, 10), 0.5),
            "preview renderer should not keep truncating and rounding RGB interpolation as separate policies",
        )

    def test_reader_surface_stays_preview_scoped(self):
        reader = _load_module("narrator_reader", "scripts/narrator_reader.py")

        self.assertFalse(
            hasattr(reader, "_blend_rgb"),
            "preview substrate should not drag unrelated display helpers back into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_history_group_phase"),
            "preview substrate should not reintroduce history-phase helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_history_entry_phase"),
            "preview substrate should not reintroduce history-entry helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_scorebug_big_value_rows"),
            "preview substrate should not reintroduce scorebug-only helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_append_scorebug_value_row"),
            "preview substrate should not reintroduce scorebug rendering helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_live_placeholder"),
            "preview substrate should not reintroduce live-lane helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_scale_rgb"),
            "preview substrate should not reintroduce generic color helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_history_tier_dim_factor"),
            "preview substrate should not reintroduce history dimming helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_render_layer_index"),
            "preview substrate should not reintroduce history layer helpers into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_message_requires_immediate_refresh"),
            "preview substrate should not reintroduce message refresh policy into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_interp_rgb"),
            "preview substrate should not keep a duplicate color interpolation helper in narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_HISTORY_TIER_DIM_FLOOR_DEPTH"),
            "preview substrate should not reintroduce unused history dimming constants into narrator_reader",
        )
        self.assertFalse(
            hasattr(reader, "_HISTORY_TIER_DIM_EASE_POWER"),
            "preview substrate should not reintroduce unused history dimming constants into narrator_reader",
        )

    def test_reader_exports_inline_preview_surface(self):
        reader = _load_module("narrator_reader", "scripts/narrator_reader.py")

        self.assertTrue(
            hasattr(reader, "FocusPreviewInlineImage"),
            "reader should expose the inline focus-preview renderable surface",
        )
        self.assertTrue(
            hasattr(reader, "_build_iterm2_inline_image_sequence"),
            "reader should expose the iTerm2/WezTerm inline image escape builder",
        )
        self.assertTrue(
            hasattr(reader, "_supports_inline_images"),
            "reader should advertise inline-image capability detection",
        )

    def test_reader_reuses_extracted_focus_preview_module(self):
        reader = _load_module("narrator_reader", "scripts/narrator_reader.py")
        self.assertEqual(
            reader.FocusPreviewInlineImage.__module__,
            "scripts.focus_preview_renderer",
            "reader should import the inline preview surface from the extracted module",
        )
        self.assertEqual(
            reader._supports_inline_images.__module__,
            "scripts.focus_preview_renderer",
            "reader should reuse inline-image capability detection from the extracted module",
        )
        self.assertEqual(
            reader._lerp_rgb.__module__,
            "narrator_reader",
            "reader should keep its general shimmer interpolation helper local",
        )
        self.assertEqual(
            reader._rgb_to_hex.__module__,
            "narrator_reader",
            "reader should keep its general style-formatting helper local",
        )

    def test_reader_source_marks_used_preview_imports_honestly(self):
        source = (
            Path(__file__).resolve().parent.parent / "scripts" / "narrator_reader.py"
        ).read_text()

        self.assertNotIn(
            "from scripts.focus_preview_renderer import _rgb_to_hex",
            source,
            "reader should not pull generic style-formatting helpers in from the preview renderer",
        )
        self.assertNotIn(
            "from scripts.focus_preview_renderer import _clamp",
            source,
            "reader should not import generic clamp helpers from the preview renderer",
        )
        self.assertNotIn(
            "from scripts.focus_preview_renderer import _lerp_rgb",
            source,
            "reader should not couple general shimmer interpolation to the preview renderer",
        )
        self.assertNotIn(
            "from scripts.focus_preview_renderer import _pixel_luma",
            source,
            "reader should not import generic luminance helpers from the preview renderer",
        )

    def test_reader_live_surface_uses_manual_alt_screen_not_main_buffer_scroll(self):
        source = (
            Path(__file__).resolve().parent.parent / "scripts" / "narrator_reader.py"
        ).read_text()

        self.assertIn(
            '\\033[?1049h',
            source,
            "reader should enter alt-screen explicitly so live preview does not stack duplicate frames into the main buffer",
        )
        self.assertIn(
            "def _live_update(",
            source,
            "reader should drive paints through a dedicated live-update path instead of a bare 30fps main-buffer redraw loop",
        )
        self.assertNotIn(
            "screen=False — stay in the terminal's main screen buffer",
            source,
            "reader should not retain the legacy main-buffer redraw path that accepts stacked/garbled preview frames",
        )

    def test_kitty_preview_reuses_place_sequence_when_geometry_is_stable(self):
        renderer = _load_module("focus_preview_renderer", "scripts/focus_preview_renderer.py")

        renderable = renderer.FocusPreviewKittyImage(
            image_id=99,
            texture_seed=99,
            image_pixel_width=320,
            image_pixel_height=200,
            terminal_cell_aspect=2.0,
            title="focus preview · test",
        )
        console = renderer.Console(color_system="truecolor", force_terminal=True)
        options = console.options.update(width=80, max_width=80)

        first_segments = list(renderable.__rich_console__(console, options))
        second_segments = list(renderable.__rich_console__(console, options))

        def _count_place(segments):
            return sum(
                1
                for segment in segments
                if "a=p" in getattr(segment, "text", "")
            )

        self.assertEqual(
            _count_place(first_segments),
            1,
            "first steady render should place the kitty image once",
        )
        self.assertEqual(
            _count_place(second_segments),
            0,
            "same-size steady rerender should reuse the existing kitty placement instead of re-placing every frame",
        )


if __name__ == "__main__":
    unittest.main()
