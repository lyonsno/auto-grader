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


class NarratorReaderPaletteContract(unittest.TestCase):
    def test_match_verdict_base_color_reads_cooler_than_warm_celadon(self):
        module = _load_narrator_reader()
        red, green, blue = module._BASE_RGB["topic_match"]

        self.assertGreater(
            blue,
            red,
            "match verdict should stay on the cool side, not read amber/wheat",
        )
        self.assertLessEqual(
            abs(green - blue),
            12,
            "match verdict should read as a colder electric aqua, not mossy green",
        )

    def test_match_verdict_gets_its_own_shimmer_lift_and_cool_peak(self):
        module = _load_narrator_reader()
        self.assertGreater(
            module._SHIMMER_KIND_INTENSITY["topic_match"],
            module._SHIMMER_KIND_INTENSITY["topic"],
            "match verdict should shimmer slightly more than a generic topic",
        )

        red, green, blue = module._SHIMMER_KIND_PEAK_RGB["topic_match"]
        self.assertGreaterEqual(
            blue,
            green,
            "match shimmer peak should borrow the cooler sky accent, not warm toward ochre",
        )
        self.assertGreater(
            green,
            red,
            "match shimmer peak should stay in the cool family",
        )


if __name__ == "__main__":
    unittest.main()
