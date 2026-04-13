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
            reader._clamp.__module__,
            "scripts.focus_preview_renderer",
            "reader should re-export _clamp from the extracted preview module",
        )
        self.assertEqual(
            reader._lerp_rgb.__module__,
            "scripts.focus_preview_renderer",
            "reader should re-export _lerp_rgb from the extracted preview module",
        )
        self.assertEqual(
            reader._pixel_luma.__module__,
            "scripts.focus_preview_renderer",
            "reader should re-export _pixel_luma from the extracted preview module",
        )


if __name__ == "__main__":
    unittest.main()
