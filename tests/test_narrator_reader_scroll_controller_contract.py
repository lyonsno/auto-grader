"""Scroll key controller contract (Crispy Drips slice 3).

The live Paint Dry reader installs a small key -> action controller
that maps single keystrokes to PaintDryDisplay scroll calls. The
controller itself is pure logic: given a display and a byte/char, it
invokes the corresponding method (or none).

Raw-terminal plumbing (termios / tty.setcbreak / side thread) is NOT
exercised here — it is untestable without a real TTY. What is pinned
here is:

  * The set of keys that MUST scroll the history pane.
  * That an unbound key is a no-op and does not reset scroll state.
  * That an explicit "return to live edge" key actually returns.
  * That the controller operates on a live PaintDryDisplay, not a
    stub, so the integration wiring of Crispy Drips' public surface
    stays honest.
"""

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


def _populate_many_items(module, display, n: int = 10) -> None:
    for i in range(n):
        display.history.append(("header", f"[item {i+1}/{n}] test-{i}", None))
        display.history.append(("line", f"line-{i}-a", 0))
        display.history.append(("line", f"line-{i}-b", 0))
        display.history.append(("topic", f"topic-{i}", None))


class ScrollControllerContract(unittest.TestCase):
    def _make(self, module):
        display = module.PaintDryDisplay()
        display._wrap_width_override = 80
        _populate_many_items(module, display)
        controller = module.HistoryScrollController(display)
        return display, controller

    def test_k_scrolls_up_one_row(self):
        module = _load_narrator_reader()
        display, controller = self._make(module)
        self.assertTrue(display.history_viewport().at_live_edge)
        controller.handle_key("k")
        self.assertFalse(display.history_viewport().at_live_edge)
        self.assertEqual(display.history_viewport().scroll_offset, 1)

    def test_j_scrolls_down_one_row(self):
        module = _load_narrator_reader()
        display, controller = self._make(module)
        controller.handle_key("k")
        controller.handle_key("k")
        self.assertEqual(display.history_viewport().scroll_offset, 2)
        controller.handle_key("j")
        self.assertEqual(display.history_viewport().scroll_offset, 1)

    def test_u_and_d_are_page_sized_scrolls(self):
        module = _load_narrator_reader()
        display, controller = self._make(module)
        controller.handle_key("u")
        page_offset = display.history_viewport().scroll_offset
        self.assertGreater(
            page_offset, 1,
            "u must scroll by more than one row (page up)",
        )
        controller.handle_key("d")
        self.assertLess(
            display.history_viewport().scroll_offset, page_offset,
            "d must scroll back down by more than zero",
        )

    def test_zero_returns_to_live_edge(self):
        module = _load_narrator_reader()
        display, controller = self._make(module)
        controller.handle_key("u")
        self.assertFalse(display.history_viewport().at_live_edge)
        controller.handle_key("0")
        self.assertTrue(display.history_viewport().at_live_edge)
        self.assertEqual(display.history_viewport().scroll_offset, 0)

    def test_unbound_key_is_a_no_op_and_preserves_scroll(self):
        module = _load_narrator_reader()
        display, controller = self._make(module)
        controller.handle_key("k")
        before = display.history_viewport().scroll_offset
        controller.handle_key("z")
        controller.handle_key("!")
        controller.handle_key("\x1b")  # ESC byte
        self.assertEqual(display.history_viewport().scroll_offset, before)

    def test_controller_reports_whether_a_key_was_handled(self):
        module = _load_narrator_reader()
        _display, controller = self._make(module)
        self.assertTrue(controller.handle_key("k"))
        self.assertTrue(controller.handle_key("j"))
        self.assertTrue(controller.handle_key("0"))
        self.assertFalse(controller.handle_key("z"))

    def test_handled_keys_are_discoverable_for_help_text(self):
        module = _load_narrator_reader()
        _display, controller = self._make(module)
        bindings = controller.bindings()
        self.assertIsInstance(bindings, dict)
        # The minimal smokeable surface: up, down, page up, page down,
        # live edge.
        for key in ("k", "j", "u", "d", "0"):
            self.assertIn(key, bindings)
            self.assertTrue(bindings[key])  # non-empty description


if __name__ == "__main__":
    unittest.main()
