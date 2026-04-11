"""Build small printable MC/OpenCV packets for real paper threshold testing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

from auto_grader.generation import (
    _BUBBLE_GAP,
    _BUBBLE_LABELS,
    _BUBBLE_SIZE,
    _LETTER_HEIGHT,
    _LETTER_WIDTH,
    _build_identity_qr_codes,
    _build_registration_markers,
    build_mc_answer_sheet,
)
from auto_grader.pdf_rendering import render_mc_answer_sheet_pdf


_CALIBRATION_PACKET_TEMPLATE_SLUG = "mc-paper-calibration"
_THRESHOLD_PACKET_TEMPLATE_SLUG = "mc-threshold-stress"
_PACKET_ROWS_PER_PAGE = 6
_PACKET_LAYOUT_TOP = 150
_PACKET_ROW_HEIGHT = 94
_PACKET_BUBBLE_ROW_LEFT = 372


def build_mc_paper_calibration_packet(*, seed: int | str = 17) -> dict[str, Any]:
    """Return a printable two-page calibration packet plus scenario manifest."""
    return _build_instruction_packet(
        packet_name=_CALIBRATION_PACKET_TEMPLATE_SLUG,
        packet_title="MC Paper Calibration Packet",
        scenarios=_scenario_manifest(),
        student_id="paper-calibration",
        student_name="Paper Calibration",
        seed=seed,
    )

def write_mc_paper_calibration_packet(
    *,
    output_dir: str | Path,
    seed: int | str = 17,
) -> dict[str, Any]:
    """Render the packet and write the PDF plus JSON artifacts to disk."""
    return _write_instruction_packet(
        output_dir=output_dir,
        packet=build_mc_paper_calibration_packet(seed=seed),
        packet_stem="mc-paper-calibration-packet",
        artifact_stem="mc-paper-calibration-artifact",
        scenario_stem="mc-paper-calibration-scenarios",
    )


def build_mc_threshold_stress_packet(*, seed: int | str = 17) -> dict[str, Any]:
    """Return a printable two-page packet aimed at the practical threshold boundary."""
    return _build_instruction_packet(
        packet_name=_THRESHOLD_PACKET_TEMPLATE_SLUG,
        packet_title="MC Threshold Stress Packet",
        scenarios=_threshold_scenario_manifest(),
        student_id="paper-threshold",
        student_name="Paper Threshold",
        seed=seed,
    )


def write_mc_threshold_stress_packet(
    *,
    output_dir: str | Path,
    seed: int | str = 17,
) -> dict[str, Any]:
    """Render the threshold packet and write the PDF plus JSON artifacts to disk."""
    return _write_instruction_packet(
        output_dir=output_dir,
        packet=build_mc_threshold_stress_packet(seed=seed),
        packet_stem="mc-threshold-stress-packet",
        artifact_stem="mc-threshold-stress-artifact",
        scenario_stem="mc-threshold-stress-scenarios",
    )


def _build_instruction_packet(
    *,
    packet_name: str,
    packet_title: str,
    scenarios: list[Mapping[str, Any]],
    student_id: str,
    student_name: str,
    seed: int | str,
) -> dict[str, Any]:
    template = _packet_template(packet_name, packet_title, scenarios)
    artifact = build_mc_answer_sheet(
        template,
        {"student_id": student_id, "student_name": student_name},
        attempt_number=1,
        seed=seed,
    )

    for question in artifact["mc_questions"]:
        question["show_choice_legend"] = False

    page_numbers_by_question = _page_numbers_for_questions(artifact["mc_questions"])
    artifact["pages"] = _build_compact_pages(artifact["opaque_instance_code"], artifact["mc_questions"])

    enriched_manifest: list[dict[str, Any]] = []
    for scenario in scenarios:
        manifest_entry = dict(scenario)
        manifest_entry["page_number"] = page_numbers_by_question[scenario["question_id"]]
        enriched_manifest.append(manifest_entry)

    return {
        "packet_name": packet_name,
        "artifact": artifact,
        "scenario_manifest": enriched_manifest,
    }


def _write_instruction_packet(
    *,
    output_dir: str | Path,
    packet: Mapping[str, Any],
    packet_stem: str,
    artifact_stem: str,
    scenario_stem: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_bytes = render_mc_answer_sheet_pdf(packet["artifact"])

    pdf_path = output_path / f"{packet_stem}.pdf"
    artifact_path = output_path / f"{artifact_stem}.json"
    scenario_path = output_path / f"{scenario_stem}.json"

    pdf_path.write_bytes(pdf_bytes)
    artifact_path.write_text(json.dumps(packet["artifact"], indent=2), encoding="utf-8")
    scenario_path.write_text(json.dumps(packet["scenario_manifest"], indent=2), encoding="utf-8")

    return {
        "pdf_path": str(pdf_path),
        "artifact_path": str(artifact_path),
        "scenario_path": str(scenario_path),
        "page_count": len(packet["artifact"]["pages"]),
        "question_count": len(packet["artifact"]["mc_questions"]),
    }


def _scenario_manifest() -> list[dict[str, Any]]:
    return [
        {"question_id": "cal-01", "instruction": "Fill B dark and solid.", "probe_type": "clear_fill", "expected_status": "correct", "correct_choice_key": "B"},
        {"question_id": "cal-02", "instruction": "Fill D dark and solid.", "probe_type": "clear_fill", "expected_status": "correct", "correct_choice_key": "D"},
        {"question_id": "cal-03", "instruction": "Make an ugly but clearly intended fill in A.", "probe_type": "ugly_but_intended", "expected_status": "correct", "correct_choice_key": "A"},
        {"question_id": "cal-04", "instruction": "Make an ugly but clearly intended fill in C.", "probe_type": "ugly_but_intended", "expected_status": "correct", "correct_choice_key": "C"},
        {"question_id": "cal-05", "instruction": "Put a tiny accidental-looking dot inside B only. Do not fill any bubble.", "probe_type": "incidental_stray_only", "expected_status": "blank", "correct_choice_key": "A"},
        {"question_id": "cal-06", "instruction": "Put a tiny accidental-looking slash inside D only. Do not fill any bubble.", "probe_type": "incidental_stray_only", "expected_status": "blank", "correct_choice_key": "A"},
        {"question_id": "cal-07", "instruction": "Fill A dark and solid. Also add a tiny accidental speck inside C.", "probe_type": "main_fill_plus_tiny_stray", "expected_status": "correct", "correct_choice_key": "A"},
        {"question_id": "cal-08", "instruction": "Fill C dark and solid. Also add a tiny accidental speck inside A.", "probe_type": "main_fill_plus_tiny_stray", "expected_status": "correct", "correct_choice_key": "C"},
        {"question_id": "cal-09", "instruction": "Fill B dark and solid. Also add a lighter weak trace inside D.", "probe_type": "main_fill_plus_weak_secondary", "expected_status": "correct", "correct_choice_key": "B"},
        {"question_id": "cal-10", "instruction": "Fill D dark and solid. Also add a lighter weak trace inside B.", "probe_type": "main_fill_plus_weak_secondary", "expected_status": "correct", "correct_choice_key": "D"},
        {"question_id": "cal-11", "instruction": "Fill A, erase it imperfectly, then fill C dark and solid.", "probe_type": "changed_answer_erasure", "expected_status": "correct", "correct_choice_key": "C"},
        {"question_id": "cal-12", "instruction": "Fill B and D as two real answers.", "probe_type": "genuine_double_mark", "expected_status": "multiple_marked", "correct_choice_key": "B"},
    ]


def _threshold_scenario_manifest() -> list[dict[str, Any]]:
    return [
        {"question_id": "thr-01", "instruction": "Fill B dark and solid.", "probe_type": "clear_fill", "expected_status": "correct", "correct_choice_key": "B"},
        {"question_id": "thr-02", "instruction": "Put a tiny accidental-looking dot inside C only. Do not fill any bubble.", "probe_type": "incidental_stray_only", "expected_status": "blank", "correct_choice_key": "A"},
        {"question_id": "thr-03", "instruction": "Draw a short diagonal slash inside D only. Do not fill any bubble.", "probe_type": "incidental_stray_only", "expected_status": "blank", "correct_choice_key": "A"},
        {"question_id": "thr-04", "instruction": "Draw a heavy check-like tick inside B only. Do not fill any bubble.", "probe_type": "incidental_stray_only", "expected_status": "blank", "correct_choice_key": "A"},
        {"question_id": "thr-05", "instruction": "Make a compact dark scribble fill in C.", "probe_type": "compact_fill_attempt", "expected_status": "correct", "correct_choice_key": "C"},
        {"question_id": "thr-06", "instruction": "Make an ugly but clearly intended compact fill in A.", "probe_type": "ugly_but_intended", "expected_status": "correct", "correct_choice_key": "A"},
        {"question_id": "thr-07", "instruction": "Fill D dark and solid. Also add a tiny accidental speck inside B.", "probe_type": "main_fill_plus_tiny_stray", "expected_status": "correct", "correct_choice_key": "D"},
        {"question_id": "thr-08", "instruction": "Fill B dark and solid. Also add a lighter weak trace inside D.", "probe_type": "main_fill_plus_weak_secondary", "expected_status": "correct", "correct_choice_key": "B"},
        {"question_id": "thr-09", "instruction": "Fill A, erase it imperfectly, then fill C dark and solid.", "probe_type": "changed_answer_erasure", "expected_status": "correct", "correct_choice_key": "C"},
        {"question_id": "thr-10", "instruction": "Fill B and D as two real answers.", "probe_type": "genuine_double_mark", "expected_status": "multiple_marked", "correct_choice_key": "B"},
        {"question_id": "thr-11", "instruction": "Make a faint patchy center fill in B only. Do not fill any other bubble.", "probe_type": "ambiguous_patchy_fill", "expected_status": "ambiguous_mark", "correct_choice_key": "A"},
        {"question_id": "thr-12", "instruction": "Scratch out C messily until it looks unreadable. Do not fill any other bubble.", "probe_type": "illegible_scratchout", "expected_status": "illegible_mark", "correct_choice_key": "A"},
    ]


def _packet_template(
    packet_name: str,
    packet_title: str,
    scenarios: list[Mapping[str, Any]],
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for scenario in scenarios:
        questions.append(
            {
                "id": str(scenario["question_id"]),
                "points": 1,
                "answer_type": "multiple_choice",
                "prompt": str(scenario["instruction"]),
                "choices": {label: label for label in _BUBBLE_LABELS[:4]},
                "correct": str(scenario["correct_choice_key"]),
                "shuffle": False,
            }
        )
    return {
        "slug": packet_name,
        "title": packet_title,
        "sections": [{"id": "mc", "title": "Calibration", "questions": questions}],
    }


def _page_numbers_for_questions(rendered_questions: list[Mapping[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, question in enumerate(rendered_questions):
        mapping[str(question["question_id"])] = (index // _PACKET_ROWS_PER_PAGE) + 1
    return mapping


def _build_compact_pages(
    opaque_instance_code: str,
    rendered_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    registration_markers = _build_registration_markers()
    pages: list[dict[str, Any]] = []

    for page_index, start in enumerate(range(0, len(rendered_questions), _PACKET_ROWS_PER_PAGE), start=1):
        page_questions = rendered_questions[start : start + _PACKET_ROWS_PER_PAGE]
        bubble_regions: list[dict[str, Any]] = []

        for row, question in enumerate(page_questions):
            row_y = _PACKET_LAYOUT_TOP + (row * _PACKET_ROW_HEIGHT)
            for choice_index, choice in enumerate(question["choices"]):
                bubble_regions.append(
                    {
                        "question_id": question["question_id"],
                        "bubble_label": choice["bubble_label"],
                        "shape": "circle",
                        "x": _PACKET_BUBBLE_ROW_LEFT + choice_index * (_BUBBLE_SIZE + _BUBBLE_GAP),
                        "y": row_y,
                        "width": _BUBBLE_SIZE,
                        "height": _BUBBLE_SIZE,
                    }
                )

        fallback_page_code = f"{opaque_instance_code}-p{page_index}"
        pages.append(
            {
                "page_number": page_index,
                "fallback_page_code": fallback_page_code,
                "layout_version": "mc_answer_sheet_v1",
                "units": "pt",
                "origin": "top_left",
                "y_axis": "down",
                "width": _LETTER_WIDTH,
                "height": _LETTER_HEIGHT,
                "identity_qr_codes": _build_identity_qr_codes(fallback_page_code),
                "registration_markers": registration_markers,
                "bubble_regions": bubble_regions,
            }
        )

    return pages
