from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_narrator_reader():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "narrator_reader.py"
    )
    spec = importlib.util.spec_from_file_location("narrator_reader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FocusPreviewSurfaceContract(unittest.TestCase):
    def test_reader_exports_inline_preview_surface(self):
        module = _load_narrator_reader()

        self.assertTrue(
            hasattr(module, "FocusPreviewInlineImage"),
            "reader should expose the inline focus-preview renderable surface",
        )
        self.assertTrue(
            hasattr(module, "_build_iterm2_inline_image_sequence"),
            "reader should expose the iTerm2/WezTerm inline image escape builder",
        )
        self.assertTrue(
            hasattr(module, "_supports_inline_images"),
            "reader should advertise inline-image capability detection",
        )


if __name__ == "__main__":
    unittest.main()
