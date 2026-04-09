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
    def test_status_base_color_reads_as_indigo_steel_not_turquoise(self):
        module = _load_narrator_reader()
        red, green, blue = module._BASE_RGB["status"]

        self.assertGreater(
            blue,
            green,
            "status should stay firmly blue-led, not drift toward turquoise",
        )
        self.assertGreater(
            blue - green,
            20,
            "status should read as indigo-steel, not aqua/celadon",
        )
        self.assertLess(
            red,
            green,
            "status should keep the red channel clearly behind the cool steel mix",
        )

    def test_status_and_match_share_the_indigo_steel_family(self):
        module = _load_narrator_reader()
        self.assertGreater(
            module._SHIMMER_KIND_INTENSITY["status"],
            module._SHIMMER_KIND_INTENSITY["topic"],
            "status should shimmer a bit more than a generic topic line",
        )

        status_red, status_green, status_blue = module._SHIMMER_KIND_PEAK_RGB["status"]
        match_red, match_green, match_blue = module._SHIMMER_KIND_PEAK_RGB["topic_match"]

        self.assertGreater(
            status_blue - status_green,
            18,
            "status shimmer peak should stay steel-blue instead of washing aqua",
        )
        self.assertGreater(
            match_blue - match_green,
            18,
            "match shimmer peak should harmonize with the indigo status lane",
        )
        self.assertLess(
            abs(status_blue - match_blue),
            28,
            "status and match highlights should feel like the same blue family",
        )


if __name__ == "__main__":
    unittest.main()
