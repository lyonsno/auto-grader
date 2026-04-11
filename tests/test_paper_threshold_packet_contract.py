from __future__ import annotations

import unittest


def _load_builder(test_case: unittest.TestCase):
    try:
        from auto_grader.paper_calibration_packet import build_mc_threshold_stress_packet
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.paper_calibration_packet.build_mc_threshold_stress_packet(...)` "
            "so we can print a second real-paper packet that pressures the "
            "current MC/OpenCV threshold boundary instead of relying only on the "
            "broader calibration packet."
        )
    except ImportError:
        test_case.fail(
            "Export `build_mc_threshold_stress_packet(...)` from "
            "`auto_grader.paper_calibration_packet` so the threshold-stress "
            "packet has a stable repo-local builder."
        )
    return build_mc_threshold_stress_packet


class PaperThresholdPacketContractTests(unittest.TestCase):
    def test_threshold_packet_is_two_pages_with_six_instruction_questions_per_page(self) -> None:
        build_mc_threshold_stress_packet = _load_builder(self)

        packet = build_mc_threshold_stress_packet(seed=17)

        artifact = packet["artifact"]
        self.assertEqual(len(artifact["mc_questions"]), 12)
        self.assertEqual(len(artifact["pages"]), 2)

        page_question_counts: list[int] = []
        for page in artifact["pages"]:
            question_ids = {region["question_id"] for region in page["bubble_regions"]}
            page_question_counts.append(len(question_ids))
        self.assertEqual(page_question_counts, [6, 6])

    def test_threshold_packet_embeds_instruction_in_each_question_prompt(self) -> None:
        build_mc_threshold_stress_packet = _load_builder(self)

        packet = build_mc_threshold_stress_packet(seed=17)

        instructions_by_question = {
            scenario["question_id"]: scenario["instruction"]
            for scenario in packet["scenario_manifest"]
        }
        for question in packet["artifact"]["mc_questions"]:
            self.assertEqual(question["prompt"], instructions_by_question[question["question_id"]])
            self.assertFalse(question["show_choice_legend"])

    def test_threshold_packet_covers_boundary_probe_mix(self) -> None:
        build_mc_threshold_stress_packet = _load_builder(self)

        packet = build_mc_threshold_stress_packet(seed=17)

        expected = {
            "thr-01": ("clear_fill", "correct"),
            "thr-02": ("incidental_stray_only", "blank"),
            "thr-03": ("incidental_stray_only", "blank"),
            "thr-04": ("incidental_stray_only", "blank"),
            "thr-05": ("compact_fill_attempt", "correct"),
            "thr-06": ("ugly_but_intended", "correct"),
            "thr-07": ("main_fill_plus_tiny_stray", "correct"),
            "thr-08": ("main_fill_plus_weak_secondary", "correct"),
            "thr-09": ("changed_answer_erasure", "correct"),
            "thr-10": ("genuine_double_mark", "multiple_marked"),
            "thr-11": ("ambiguous_patchy_fill", "ambiguous_mark"),
            "thr-12": ("illegible_scratchout", "illegible_mark"),
        }
        observed = {
            scenario["question_id"]: (scenario["probe_type"], scenario["expected_status"])
            for scenario in packet["scenario_manifest"]
        }
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
