from __future__ import annotations

import unittest

from scripts.narrator_reader import PaintDryDisplay


class NarratorReaderContract(unittest.TestCase):
    def test_session_end_disables_animation(self):
        display = PaintDryDisplay()

        self.assertTrue(display.should_animate())

        display.session_ended = True

        self.assertFalse(display.should_animate())


if __name__ == "__main__":
    unittest.main()
