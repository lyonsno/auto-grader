from __future__ import annotations

import unittest


def _load_builder(test_case: unittest.TestCase):
    try:
        from auto_grader.paper_calibration_packet import build_mc_paper_calibration_packet
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.paper_calibration_packet.build_mc_paper_calibration_packet(...)` "
            "so we can print a tiny real-paper calibration packet instead of relying only on synthetic probes."
        )
    return build_mc_paper_calibration_packet


class PaperCalibrationPacketContractTests(unittest.TestCase):
    def test_packet_is_two_pages_with_six_instruction_questions_per_page(self) -> None:
        build_mc_paper_calibration_packet = _load_builder(self)

        packet = build_mc_paper_calibration_packet(seed=17)

        artifact = packet["artifact"]
        self.assertEqual(len(artifact["mc_questions"]), 12)
        self.assertEqual(len(artifact["pages"]), 2)

        page_question_counts: list[int] = []
        for page in artifact["pages"]:
            question_ids = {region["question_id"] for region in page["bubble_regions"]}
            page_question_counts.append(len(question_ids))
        self.assertEqual(page_question_counts, [6, 6])

    def test_packet_embeds_marking_instruction_in_each_question_prompt(self) -> None:
        build_mc_paper_calibration_packet = _load_builder(self)

        packet = build_mc_paper_calibration_packet(seed=17)

        instructions_by_question = {
            scenario["question_id"]: scenario["instruction"]
            for scenario in packet["scenario_manifest"]
        }
        for question in packet["artifact"]["mc_questions"]:
            self.assertEqual(question["prompt"], instructions_by_question[question["question_id"]])
            self.assertFalse(question["show_choice_legend"])

    def test_packet_records_expected_probe_behaviors(self) -> None:
        build_mc_paper_calibration_packet = _load_builder(self)

        packet = build_mc_paper_calibration_packet(seed=17)

        expected = {
            "cal-01": ("clear_fill", "correct"),
            "cal-02": ("clear_fill", "correct"),
            "cal-03": ("ugly_but_intended", "correct"),
            "cal-04": ("ugly_but_intended", "correct"),
            "cal-05": ("incidental_stray_only", "blank"),
            "cal-06": ("incidental_stray_only", "blank"),
            "cal-07": ("main_fill_plus_tiny_stray", "correct"),
            "cal-08": ("main_fill_plus_tiny_stray", "correct"),
            "cal-09": ("main_fill_plus_weak_secondary", "correct"),
            "cal-10": ("main_fill_plus_weak_secondary", "correct"),
            "cal-11": ("changed_answer_erasure", "correct"),
            "cal-12": ("genuine_double_mark", "multiple_marked"),
        }
        observed = {
            scenario["question_id"]: (scenario["probe_type"], scenario["expected_status"])
            for scenario in packet["scenario_manifest"]
        }
        self.assertEqual(observed, expected)

    def test_packet_pages_match_public_compact_generation_layout(self) -> None:
        build_mc_paper_calibration_packet = _load_builder(self)
        try:
            from auto_grader.generation import McAnswerSheetLayout, build_mc_answer_sheet_pages
        except ImportError:
            self.fail(
                "Route paper-calibration packet pages through a public generation layout seam "
                "instead of a packet-local page builder."
            )

        packet = build_mc_paper_calibration_packet(seed=17)
        artifact = packet["artifact"]
        compact_layout = McAnswerSheetLayout(
            rows_per_page=6,
            layout_top=150,
            row_height=94,
            bubble_row_left=372,
        )

        expected_pages = build_mc_answer_sheet_pages(
            artifact["opaque_instance_code"],
            artifact["mc_questions"],
            layout=compact_layout,
        )

        self.assertEqual(artifact["pages"], expected_pages)
