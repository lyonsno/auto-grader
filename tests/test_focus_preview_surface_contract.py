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


if __name__ == "__main__":
    unittest.main()
