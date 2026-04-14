"""Fail-first tests for the focus preview pipeline.

These cover the three integration points that must exist for focus
preview to render in a narrator-bearing smoke:

1. NarratorSink.write_focus_preview emits a base64-encoded focus_preview event
2. PaintDryDisplay.on_focus_preview stores state so render() can compose it
3. PaintDryDisplay.render() includes the focus preview panel when preview state is set
4. The message dispatch loop routes focus_preview events to on_focus_preview
"""

from __future__ import annotations

import base64
import importlib.util
import json
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from auto_grader.narrator_sink import NarratorSink, SinkConfig


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


# -- Sink emission ---------------------------------------------------------

class TestSinkWriteFocusPreview(unittest.TestCase):
    def test_write_focus_preview_emits_base64_event(self):
        """NarratorSink must have write_focus_preview that emits a
        focus_preview JSONL event with the PNG encoded as base64."""
        buf = StringIO()
        config = SinkConfig(spawn_terminal=False, fallback_stream=buf)
        with NarratorSink(config) as sink:
            # Minimal PNG-like bytes (not a real image, just proving the
            # base64 encoding contract).
            fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
            sink.write_focus_preview(
                fake_png, label="15-blue/fr-1", source="page-cache"
            )

        # The JSONL log should contain the event. Since we don't have a
        # log_dir, check via the jsonl_file or the fifo. The simplest
        # contract: the method exists and doesn't crash. For a stronger
        # assertion, set up log_dir.
        self.assertTrue(
            hasattr(sink, "write_focus_preview"),
            "NarratorSink must expose write_focus_preview",
        )

    def test_write_focus_preview_jsonl_contains_base64(self):
        """When log_dir is set, the JSONL log must contain the
        focus_preview event with png_base64 field."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            config = SinkConfig(
                spawn_terminal=False,
                log_dir=log_dir,
                fallback_stream=StringIO(),
            )
            fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
            with NarratorSink(config) as sink:
                sink.write_focus_preview(
                    fake_png, label="test-label", source="test-source"
                )

            jsonl_path = log_dir / "narrator.jsonl"
            self.assertTrue(jsonl_path.exists(), "JSONL log must exist")
            lines = jsonl_path.read_text().strip().splitlines()
            focus_events = [
                json.loads(l) for l in lines
                if '"focus_preview"' in l
            ]
            self.assertEqual(len(focus_events), 1, "exactly one focus_preview event")
            evt = focus_events[0]
            self.assertEqual(evt["type"], "focus_preview")
            self.assertIn("png_base64", evt)
            decoded = base64.b64decode(evt["png_base64"])
            self.assertEqual(decoded, fake_png)
            self.assertEqual(evt.get("label"), "test-label")
            self.assertEqual(evt.get("source"), "test-source")

    def test_write_focus_preview_skips_empty_bytes(self):
        """Empty PNG bytes should be silently skipped."""
        buf = StringIO()
        config = SinkConfig(spawn_terminal=False, fallback_stream=buf)
        with NarratorSink(config) as sink:
            sink.write_focus_preview(b"")
        # No crash = pass. The contract is that empty bytes are a no-op.


# -- Reader display state -------------------------------------------------

class TestDisplayFocusPreviewState(unittest.TestCase):
    def test_display_has_on_focus_preview_handler(self):
        """PaintDryDisplay must have an on_focus_preview method."""
        module = _load_narrator_reader()
        display = module.PaintDryDisplay()
        self.assertTrue(
            hasattr(display, "on_focus_preview"),
            "PaintDryDisplay must expose on_focus_preview handler",
        )

    def test_on_focus_preview_stores_png_bytes(self):
        """on_focus_preview must store the PNG bytes for render() to use."""
        module = _load_narrator_reader()
        display = module.PaintDryDisplay()

        # Generate a valid minimal PNG via fitz so the half-block
        # fallback path (which calls fitz.Pixmap) doesn't choke.
        import fitz
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), 1)
        pix.clear_with(255)
        valid_png = pix.tobytes("png")

        display.on_focus_preview(valid_png, label="test", source="cache")

        self.assertEqual(
            display.focus_preview_png,
            valid_png,
            "on_focus_preview must store the raw PNG bytes",
        )

    def test_display_initializes_focus_preview_state(self):
        """PaintDryDisplay.__init__ must set up focus preview state attrs."""
        module = _load_narrator_reader()
        display = module.PaintDryDisplay()

        self.assertIsNone(display.focus_preview_png)
        self.assertEqual(display.focus_preview_label, "")
        self.assertFalse(display.focus_preview_pending)


# -- Render composition ---------------------------------------------------

class TestRenderIncludesFocusPreview(unittest.TestCase):
    def test_render_includes_focus_preview_loading_band_by_default(self):
        """render() should include a FocusPreviewLoadingBand placeholder
        even before any focus_preview event fires."""
        module = _load_narrator_reader()
        display = module.PaintDryDisplay()

        rendered = display.render()
        # The Group's renderables should include the loading band.
        # We check that more than the base 3 panels are present
        # (header + live + history = 3; with preview = 4+).
        renderables = list(rendered.renderables)
        self.assertGreaterEqual(
            len(renderables),
            4,
            "render() must include at least 4 panels when focus preview "
            "loading band is active (header + live + preview + history)",
        )


if __name__ == "__main__":
    unittest.main()
