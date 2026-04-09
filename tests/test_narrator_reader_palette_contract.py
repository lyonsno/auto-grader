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
    def test_match_verdict_base_color_reads_as_indigo_steel_not_aqua(self):
        module = _load_narrator_reader()
        red, green, blue = module._BASE_RGB["topic_match"]

        self.assertGreater(
            blue,
            green,
            "match verdict should land in the blue-forward family, not aqua",
        )
        self.assertGreater(
            green,
            red,
            "match verdict should still stay steel-cool rather than pure violet",
        )
        self.assertLess(
            green - red,
            45,
            "match verdict should be restrained and inky, not bright turquoise",
        )

    def test_status_rail_uses_same_indigo_steel_family(self):
        module = _load_narrator_reader()
        red, green, blue = module._BASE_RGB["status"]

        self.assertGreater(blue, green)
        self.assertGreater(green, red)

    def test_match_verdict_and_status_get_cool_steel_shimmer(self):
        module = _load_narrator_reader()
        self.assertGreater(
            module._SHIMMER_KIND_INTENSITY["topic_match"],
            module._SHIMMER_KIND_INTENSITY["topic"],
            "match verdict should still shimmer a little more than a generic topic",
        )

        for kind in ("topic_match", "status"):
            red, green, blue = module._SHIMMER_KIND_PEAK_RGB[kind]
            self.assertGreater(blue, green)
            self.assertGreater(green, red)


if __name__ == "__main__":
    unittest.main()
