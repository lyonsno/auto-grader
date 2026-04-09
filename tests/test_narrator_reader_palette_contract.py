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
    def test_header_title_reads_as_lacquer_red_not_plain_orange(self):
        module = _load_narrator_reader()
        red, green, blue = module._BASE_RGB["header"]

        self.assertGreater(
            red,
            green,
            "header title should stay red-led rather than orange-led",
        )
        self.assertLess(
            green,
            100,
            "header title should sit darker and redder, closer to lacquer than pumpkin",
        )
        self.assertLess(
            blue,
            60,
            "header title should stay out of burgundy-purple territory",
        )

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

    def test_status_rail_is_darker_than_header_index_accent(self):
        module = _load_narrator_reader()
        status_red, status_green, status_blue = module._BASE_RGB["status"]
        index_red, index_green, index_blue = module._BASE_RGB["header_index"]

        self.assertLess(
            status_red + status_green + status_blue,
            index_red + index_green + index_blue,
            "status rail should sit a step darker than the crisp header-index blue",
        )

    def test_match_verdict_is_darker_and_more_indigo_than_header_index(self):
        module = _load_narrator_reader()
        match_red, match_green, match_blue = module._BASE_RGB["topic_match"]
        index_red, index_green, index_blue = module._BASE_RGB["header_index"]

        self.assertLess(
            match_red,
            index_red,
            "match verdict should sit deeper in the indigo family than the index blue",
        )
        self.assertLess(
            match_green,
            index_green,
            "match verdict should be darker than the header-index accent, not identical",
        )
        self.assertLess(
            match_blue,
            index_blue,
            "match verdict should complement the header-index blue rather than duplicate it",
        )

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

    def test_live_field_is_tempered_below_the_old_hot_pass(self):
        module = _load_narrator_reader()

        self.assertLessEqual(
            module._LIVE_BASE_SAT,
            0.74,
            "live field should be calmer than the earlier hotter saturation pass",
        )
        self.assertLessEqual(
            module._LIVE_BASE_VAL,
            0.92,
            "live field should be slightly dimmer than the earlier glare-prone pass",
        )


if __name__ == "__main__":
    unittest.main()
