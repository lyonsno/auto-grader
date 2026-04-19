"""Minimal PDF rendering for deterministic MC answer sheets.

This module intentionally keeps the paper surface small and explicit:

- consume the generation artifact as the sole layout truth
- render visible recovery codes onto the page
- draw the exact bubble circles recorded in page space
- render duplicated identity QR codes and registration markers from the same artifact
- avoid heavyweight PDF dependencies while the contract is still moving
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import fitz
import qrcode

from auto_grader.mc_layout import (
    CHOICE_LEGEND_LINE_SPACING as _CHOICE_LEGEND_LINE_SPACING,
    CHOICE_LEGEND_TOP_OFFSET as _CHOICE_LEGEND_TOP_OFFSET,
    PROMPT_LINE_GAP as _PROMPT_LINE_GAP,
    PROMPT_LINE_SPACING as _PROMPT_LINE_SPACING,
    display_prompt as _display_prompt,
    wrap_choice_text as _wrap_choice_text,
    wrap_prompt_text as _wrap_prompt_text,
)

_QUESTION_BLOCK_LEFT = 72
_HEADER_LEFT = 60
_HEADER_TITLE_TOP = 36
_HEADER_INSTANCE_TOP = 62
_HEADER_PAGE_CODE_TOP = 82
_HEADER_TITLE_FONT_SIZE = 18
_HEADER_META_FONT_SIZE = 10
_QUESTION_FONT_SIZE = 13
_CHOICE_LEGEND_FONT_SIZE = 10
_BUBBLE_LABEL_FONT_SIZE = 10
_BUBBLE_LABEL_TOP_OFFSET = 6
_CHOICE_LEGEND_LEFT_OFFSET = 12


def render_mc_answer_sheet_pdf(artifact: Mapping[str, Any]) -> bytes:
    """Render a generated MC answer-sheet artifact into PDF bytes."""
    if not isinstance(artifact, Mapping):
        raise TypeError("artifact must be a mapping")

    pages = artifact.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("artifact.pages must be a non-empty list")

    objects: list[bytes] = []
    page_object_numbers: list[int] = []
    next_object_number = 4

    for page in pages:
        page_object_numbers.append(next_object_number)
        next_object_number += 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{object_number} 0 R" for object_number in page_object_numbers)
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("utf-8"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, page in enumerate(pages):
        page_object_number = page_object_numbers[index]
        content_object_number = page_object_number + 1
        rendered_page = _render_page(artifact, _require_page(page))
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {_pdf_number(rendered_page['width'])} {_pdf_number(rendered_page['height'])}] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("utf-8")
        )
        stream = rendered_page["stream"].encode("utf-8")
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("utf-8")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    return _build_pdf(objects)


def render_quiz5_short_answer_pdf(artifact: Mapping[str, Any]) -> bytes:
    """Render a Quiz #5 short-answer artifact into PDF bytes."""
    if not isinstance(artifact, Mapping):
        raise TypeError("artifact must be a mapping")

    pages = artifact.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("artifact.pages must be a non-empty list")

    return _render_quiz5_short_answer_pdf_from_source_templates(pages)


def _render_quiz5_short_answer_pdf_from_source_templates(
    pages: list[Mapping[str, Any]],
) -> bytes:
    document = fitz.open()
    source_cache: dict[str, fitz.Document] = {}
    try:
        for raw_page in pages:
            page = _require_short_answer_page(raw_page)
            width = _require_number(page.get("width"), "page.width")
            height = _require_number(page.get("height"), "page.height")
            output_page = document.new_page(width=width, height=height)

            template_variant_id = str(page["reference_template_variant_id"])
            template_page_number = int(page["template_page_number"])
            source_doc = source_cache.get(template_variant_id)
            if source_doc is None:
                source_doc = fitz.open(_quiz5_reference_pdf_path(template_variant_id))
                source_cache[template_variant_id] = source_doc
            output_page.show_pdf_page(
                fitz.Rect(0, 0, width, height),
                source_doc,
                template_page_number - 1,
            )

            for raw_overlay in page.get("text_overlays", []):
                _apply_text_overlay(output_page, dict(raw_overlay))
            for raw_marker in page.get("registration_markers", []):
                _draw_fitz_rect(output_page, dict(raw_marker), fill=(0, 0, 0))
            for raw_qr_code in page.get("identity_qr_codes", []):
                _draw_qr_code_fitz(output_page, dict(raw_qr_code))

        return document.tobytes(garbage=4, deflate=True)
    finally:
        for source_doc in source_cache.values():
            source_doc.close()
        document.close()


def _require_page(page: Any) -> dict[str, Any]:
    if not isinstance(page, Mapping):
        raise TypeError("artifact.pages entries must be mappings")

    page_dict = dict(page)
    required_layout = {
        "layout_version": "mc_answer_sheet_v1",
        "units": "pt",
        "origin": "top_left",
        "y_axis": "down",
    }
    for key, expected in required_layout.items():
        if page_dict.get(key) != expected:
            raise ValueError(f"Unsupported page layout contract: {key} must be {expected!r}")
    return page_dict


def _require_short_answer_page(page: Any) -> dict[str, Any]:
    if not isinstance(page, Mapping):
        raise TypeError("artifact.pages entries must be mappings")

    page_dict = dict(page)
    required_layout = {
        "layout_version": "quiz5_short_answer_v1",
        "units": "pt",
        "origin": "top_left",
        "y_axis": "down",
    }
    for key, expected in required_layout.items():
        if page_dict.get(key) != expected:
            raise ValueError(f"Unsupported page layout contract: {key} must be {expected!r}")
    return page_dict


def _render_page(artifact: Mapping[str, Any], page: Mapping[str, Any]) -> dict[str, Any]:
    width = _require_number(page.get("width"), "page.width")
    height = _require_number(page.get("height"), "page.height")
    bubble_regions = page.get("bubble_regions")
    if not isinstance(bubble_regions, list):
        raise ValueError("page.bubble_regions must be a list")

    region_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    page_question_ids: set[str] = set()
    for raw_region in bubble_regions:
        if not isinstance(raw_region, Mapping):
            raise ValueError("page.bubble_regions entries must be mappings")
        region = dict(raw_region)
        key = (str(region["question_id"]), str(region["bubble_label"]))
        region_lookup[key] = region
        page_question_ids.add(str(region["question_id"]))

    content_lines = [
        "0 G",
        "1 w",
        *_text_block(_HEADER_LEFT, _pdf_text_y(height, _HEADER_TITLE_TOP), _HEADER_TITLE_FONT_SIZE, "MC Answer Sheet"),
        *_text_block(
            _HEADER_LEFT,
            _pdf_text_y(height, _HEADER_INSTANCE_TOP),
            _HEADER_META_FONT_SIZE,
            f"Instance: {artifact['opaque_instance_code']}",
        ),
        *_text_block(
            _HEADER_LEFT,
            _pdf_text_y(height, _HEADER_PAGE_CODE_TOP),
            _HEADER_META_FONT_SIZE,
            f"Page code: {page['fallback_page_code']}",
        ),
    ]

    registration_markers = page.get("registration_markers")
    if not isinstance(registration_markers, list):
        raise ValueError("page.registration_markers must be a list")

    for raw_marker in registration_markers:
        if not isinstance(raw_marker, Mapping):
            raise ValueError("page.registration_markers entries must be mappings")
        marker = dict(raw_marker)
        marker_x = _require_number(marker["x"], "registration_marker.x")
        marker_y = _require_number(marker["y"], "registration_marker.y")
        marker_width = _require_number(marker["width"], "registration_marker.width")
        marker_height = _require_number(marker["height"], "registration_marker.height")
        content_lines.append(
            (
                f"{_pdf_number(marker_x)} "
                f"{_pdf_number(_pdf_rect_y(height, marker_y, marker_height))} "
                f"{_pdf_number(marker_width)} "
                f"{_pdf_number(marker_height)} re f"
            )
        )

    identity_qr_codes = page.get("identity_qr_codes")
    if not isinstance(identity_qr_codes, list):
        raise ValueError("page.identity_qr_codes must be a list")

    for raw_qr_code in identity_qr_codes:
        if not isinstance(raw_qr_code, Mapping):
            raise ValueError("page.identity_qr_codes entries must be mappings")
        qr_code = dict(raw_qr_code)
        content_lines.extend(_render_qr_code(height, qr_code))

    questions = artifact.get("mc_questions")
    if not isinstance(questions, list):
        raise ValueError("artifact.mc_questions must be a list")

    for question_number, raw_question in enumerate(questions, start=1):
        if not isinstance(raw_question, Mapping):
            raise ValueError("artifact.mc_questions entries must be mappings")
        question = dict(raw_question)
        question_id = str(question["question_id"])
        if question_id not in page_question_ids:
            continue

        choices = question.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Question {question_id!r} must have rendered choices")

        question_regions = [
            region_lookup[(question_id, str(choice["bubble_label"]))]
            for choice in choices
        ]
        prompt_y_top = min(
            _require_number(region["y"], "bubble_region.y") for region in question_regions
        )
        prompt_left = _QUESTION_BLOCK_LEFT
        prompt_lines = _wrap_prompt_text(_display_prompt(question_number, str(question.get("prompt", ""))))
        prompt_block_top = prompt_y_top - _PROMPT_LINE_GAP - (
            _PROMPT_LINE_SPACING * (len(prompt_lines) - 1)
        )
        for line_index, prompt_line in enumerate(prompt_lines):
            content_lines.extend(
                _text_block(
                    prompt_left,
                    _pdf_text_y(height, prompt_block_top + (line_index * _PROMPT_LINE_SPACING)),
                    _QUESTION_FONT_SIZE,
                    prompt_line,
                )
            )

        first_bubble_x = min(
            _require_number(region["x"], "bubble_region.x") for region in question_regions
        )
        choice_legend_left = prompt_left + _CHOICE_LEGEND_LEFT_OFFSET
        show_choice_legend = bool(question.get("show_choice_legend", True))

        if show_choice_legend:
            choice_line_cursor = 0
            for choice in choices:
                wrapped_choice_lines = _wrap_choice_text(
                    choice["bubble_label"],
                    str(choice.get("text", "")),
                )
                for wrapped_line in wrapped_choice_lines:
                    content_lines.extend(
                        _text_block(
                            choice_legend_left,
                            _pdf_text_y(
                                height,
                                prompt_y_top
                                + _CHOICE_LEGEND_TOP_OFFSET
                                + (choice_line_cursor * _CHOICE_LEGEND_LINE_SPACING),
                            ),
                            _CHOICE_LEGEND_FONT_SIZE,
                            wrapped_line,
                        )
                    )
                    choice_line_cursor += 1

        for choice in choices:
            bubble_label = str(choice["bubble_label"])
            region = region_lookup[(question_id, bubble_label)]
            region_x = _require_number(region["x"], "bubble_region.x")
            region_y = _require_number(region["y"], "bubble_region.y")
            region_width = _require_number(region["width"], "bubble_region.width")
            region_height = _require_number(region["height"], "bubble_region.height")
            content_lines.append(
                _circle_path(
                    region_x,
                    _pdf_rect_y(height, region_y, region_height),
                    region_width,
                    region_height,
                )
            )
            center_x = region_x + (region_width / 2)
            content_lines.extend(
                _text_block(
                    center_x - 2,
                    _pdf_text_y(height, region_y + region_height + _BUBBLE_LABEL_TOP_OFFSET),
                    _BUBBLE_LABEL_FONT_SIZE,
                    bubble_label,
                )
            )
        if choice_legend_left >= first_bubble_x:
            raise ValueError("Choice legend must stay to the left of the bubble row")

    return {"width": width, "height": height, "stream": "\n".join(content_lines)}


def _render_quiz5_short_answer_page(artifact: Mapping[str, Any], page: Mapping[str, Any]) -> dict[str, Any]:
    width = _require_number(page.get("width"), "page.width")
    height = _require_number(page.get("height"), "page.height")

    content_lines = [
        "0 G",
        "1 w",
    ]
    content_lines.extend(_render_quiz5_source_header(height, page_number=int(page["page_number"])))
    if int(page["page_number"]) == 1:
        content_lines.extend(_render_quiz5_reference_band(height))

    registration_markers = page.get("registration_markers")
    if not isinstance(registration_markers, list):
        raise ValueError("page.registration_markers must be a list")
    for raw_marker in registration_markers:
        if not isinstance(raw_marker, Mapping):
            raise ValueError("page.registration_markers entries must be mappings")
        marker = dict(raw_marker)
        marker_x = _require_number(marker["x"], "registration_marker.x")
        marker_y = _require_number(marker["y"], "registration_marker.y")
        marker_width = _require_number(marker["width"], "registration_marker.width")
        marker_height = _require_number(marker["height"], "registration_marker.height")
        content_lines.append(
            (
                f"{_pdf_number(marker_x)} "
                f"{_pdf_number(_pdf_rect_y(height, marker_y, marker_height))} "
                f"{_pdf_number(marker_width)} "
                f"{_pdf_number(marker_height)} re f"
            )
        )

    identity_qr_codes = page.get("identity_qr_codes")
    if not isinstance(identity_qr_codes, list):
        raise ValueError("page.identity_qr_codes must be a list")
    for raw_qr_code in identity_qr_codes:
        if not isinstance(raw_qr_code, Mapping):
            raise ValueError("page.identity_qr_codes entries must be mappings")
        content_lines.extend(_render_qr_code(height, dict(raw_qr_code)))

    prompt_blocks = page.get("prompt_blocks")
    if not isinstance(prompt_blocks, list):
        raise ValueError("page.prompt_blocks must be a list")
    for raw_prompt_block in prompt_blocks:
        if not isinstance(raw_prompt_block, Mapping):
            raise ValueError("page.prompt_blocks entries must be mappings")
        prompt_block = dict(raw_prompt_block)
        wrapped_lines = prompt_block.get("wrapped_lines")
        if not isinstance(wrapped_lines, list):
            raise ValueError("prompt_block.wrapped_lines must be a list")
        x = _require_number(prompt_block["x"], "prompt_block.x")
        y = _require_number(prompt_block["y"], "prompt_block.y")
        font_size = _require_number(prompt_block["font_size"], "prompt_block.font_size")
        line_spacing = _require_number(prompt_block["line_spacing"], "prompt_block.line_spacing")
        continuation_x = _require_number(
            prompt_block.get("continuation_x", x),
            "prompt_block.continuation_x",
        )
        label_text = prompt_block.get("label_text")
        if label_text is not None:
            label_x = _require_number(prompt_block["label_x"], "prompt_block.label_x")
            content_lines.extend(
                _pdf_safe_text_block(
                    label_x,
                    _pdf_text_y(height, y),
                    font_size,
                    str(label_text),
                    font_resource="F4",
                )
            )
        for line_index, line in enumerate(wrapped_lines):
            line_x = x if line_index == 0 else continuation_x
            content_lines.extend(
                _pdf_safe_text_block(
                    line_x,
                    _pdf_text_y(height, y + (line_index * line_spacing)),
                    font_size,
                    str(line),
                    font_resource="F4",
                )
            )

    response_boxes = page.get("response_boxes")
    if not isinstance(response_boxes, list):
        raise ValueError("page.response_boxes must be a list")
    for raw_response_box in response_boxes:
        if not isinstance(raw_response_box, Mapping):
            raise ValueError("page.response_boxes entries must be mappings")
        response_box = dict(raw_response_box)
        box_x = _require_number(response_box["x"], "response_box.x")
        box_y = _require_number(response_box["y"], "response_box.y")
        box_width = _require_number(response_box["width"], "response_box.width")
        box_height = _require_number(response_box["height"], "response_box.height")
        content_lines.extend(
            _pdf_safe_text_block(
                box_x + 6,
                _pdf_text_y(height, box_y + 15),
                10.5,
                str(response_box["label"]),
                font_resource="F4",
            )
        )
        content_lines.append(
            (
                f"{_pdf_number(box_x)} "
                f"{_pdf_number(_pdf_rect_y(height, box_y, box_height))} "
                f"{_pdf_number(box_width)} "
                f"{_pdf_number(box_height)} re S"
            )
        )

    return {"width": width, "height": height, "stream": "\n".join(content_lines)}


def _render_quiz5_source_header(page_height: int | float, *, page_number: int) -> list[str]:
    lines: list[str] = [
        *_font_text_block("F4", 94, _pdf_text_y(page_height, 54), 14, "CHM 142"),
        *_font_text_block("F4", 224, _pdf_text_y(page_height, 54), 14, "Prof. Lyons"),
        *_font_text_block("F4", 360, _pdf_text_y(page_height, 54), 14, "Quiz #5"),
        *_font_text_block("F4", 438, _pdf_text_y(page_height, 54), 12, "26 March, 2026"),
        *_font_text_block("F2", 166, _pdf_text_y(page_height, 90), 12, "Name:"),
        *_draw_line(230, page_height, 90, 430, 90),
        *_draw_line(70, page_height, 110, 540, 110),
    ]

    if page_number == 1:
        lines.extend(
            [
                *_font_text_block("F4", 250, _pdf_text_y(page_height, 148), 10, "There are 5 questions -"),
                *_font_text_block(
                    "F2",
                    192,
                    _pdf_text_y(page_height, 170),
                    11,
                    "Be sure to enter your answers inside the boxes !",
                ),
            ]
        )
    return lines


def _render_quiz5_reference_band(page_height: int | float) -> list[str]:
    lines: list[str] = [
        *_draw_line(82, page_height, 216, 304, 216),
        *_font_text_block("F2", 112, _pdf_text_y(page_height, 230), 9, "Rate"),
        *_font_text_block("F2", 102, _pdf_text_y(page_height, 243), 9, "Order"),
        *_font_text_block("F2", 150, _pdf_text_y(page_height, 230), 9, "Rate"),
        *_font_text_block("F2", 141, _pdf_text_y(page_height, 243), 9, "Equation"),
        *_font_text_block("F2", 238, _pdf_text_y(page_height, 230), 9, "Integrated Rate"),
        *_font_text_block("F2", 247, _pdf_text_y(page_height, 243), 9, "Equation"),
        *_draw_line(82, page_height, 254, 304, 254),
        *_pdf_safe_text_block(82, _pdf_text_y(page_height, 268), 8.8, "0      -Δ[R]/ΔT = k[R]^0      [R]0 - [R]t = kt", font_resource="F4"),
        *_pdf_safe_text_block(82, _pdf_text_y(page_height, 288), 8.8, "1      -Δ[R]/ΔT = k[R]^1      ln ([R]t/[R]0) = -kt", font_resource="F4"),
        *_pdf_safe_text_block(82, _pdf_text_y(page_height, 306), 8.8, "2      -Δ[R]/ΔT = k[R]^2      (1/[R]t) - (1/[R]0) = kt", font_resource="F4"),
        *_draw_line(82, page_height, 326, 304, 326),
        *_font_text_block("F4", 320, _pdf_text_y(page_height, 228), 11, "R = 8.3145 J/(K*mol)"),
        *_font_text_block("F4", 320, _pdf_text_y(page_height, 248), 11, "R = 0.08206 L*atm/mol*K"),
        *_font_text_block("F4", 320, _pdf_text_y(page_height, 268), 11, "1atm = 760 mm Hg"),
        *_pdf_safe_text_block(320, _pdf_text_y(page_height, 294), 10.5, "ln(k2/k1) = (-Ea/R)(1/T2 - 1/T1)", font_resource="F4"),
        *_font_text_block("F4", 320, _pdf_text_y(page_height, 332), 11, "Kw = 1 x 10^-14 = [H3O+][OH-]"),
        *_font_text_block("F4", 94, _pdf_text_y(page_height, 360), 14, "1.   Net ionic equations:"),
    ]
    return lines


def _text_block(x: int | float, y: int | float, font_size: int | float, text: str) -> list[str]:
    return _font_text_block("F1", x, y, font_size, text)


def _font_text_block(
    font_resource: str,
    x: int | float,
    y: int | float,
    font_size: int | float,
    text: str,
) -> list[str]:
    return [
        "BT",
        f"/{font_resource} {_pdf_number(font_size)} Tf",
        f"{_pdf_number(x)} {_pdf_number(y)} Td",
        f"({_escape_text(text)}) Tj",
        "ET",
    ]


def _pdf_safe_text_block(
    x: int | float,
    y: int | float,
    font_size: int | float,
    text: str,
    *,
    font_resource: str = "F1",
) -> list[str]:
    return [
        "BT",
        f"/{font_resource} {_pdf_number(font_size)} Tf",
        f"{_pdf_number(x)} {_pdf_number(y)} Td",
        f"({_escape_text(_pdf_safe_text(text))}) Tj",
        "ET",
    ]


def _draw_line(
    x1: int | float,
    page_height: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
) -> list[str]:
    return [
        f"{_pdf_number(x1)} {_pdf_number(page_height - y1)} m",
        f"{_pdf_number(x2)} {_pdf_number(page_height - y2)} l S",
    ]


def _render_qr_code(page_height: int | float, qr_code: Mapping[str, Any]) -> list[str]:
    if qr_code.get("kind") != "page_identity_qr":
        raise ValueError("identity_qr_codes.kind must be 'page_identity_qr'")
    if qr_code.get("encoding") != "qr":
        raise ValueError("identity_qr_codes.encoding must be 'qr'")

    qr_x = _require_number(qr_code["x"], "identity_qr_code.x")
    qr_y = _require_number(qr_code["y"], "identity_qr_code.y")
    qr_width = _require_number(qr_code["width"], "identity_qr_code.width")
    qr_height = _require_number(qr_code["height"], "identity_qr_code.height")
    if qr_width != qr_height:
        raise ValueError("identity_qr_codes must be square")

    matrix = _qr_matrix(
        str(qr_code["payload"]),
        error_correction=str(qr_code["error_correction"]),
        border_modules=int(qr_code["border_modules"]),
    )
    module_width = qr_width / len(matrix)
    module_height = qr_height / len(matrix)
    commands: list[str] = []

    for row_index, row in enumerate(matrix):
        for col_index, is_black in enumerate(row):
            if not is_black:
                continue
            module_x = qr_x + (col_index * module_width)
            module_y = qr_y + (row_index * module_height)
            commands.append(
                (
                    f"{_pdf_number(module_x)} "
                    f"{_pdf_number(_pdf_rect_y(page_height, module_y, module_height))} "
                    f"{_pdf_number(module_width)} "
                    f"{_pdf_number(module_height)} re f"
                )
            )

    return commands


def _quiz5_asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


def _quiz5_reference_pdf_path(variant_id: str) -> Path:
    asset_root = _quiz5_asset_root()
    path = asset_root / f"260326_Quiz _5 {variant_id}.pdf"
    if not path.exists():
        raise FileNotFoundError(f"Missing Quiz #5 reference PDF for variant {variant_id}: {path}")
    return path


def _draw_fitz_rect(
    page: fitz.Page,
    rect_mapping: Mapping[str, Any],
    *,
    fill: tuple[float, float, float] | None = None,
) -> None:
    x = _require_number(rect_mapping["x"], "rect.x")
    y = _require_number(rect_mapping["y"], "rect.y")
    width = _require_number(rect_mapping["width"], "rect.width")
    height = _require_number(rect_mapping["height"], "rect.height")
    page.draw_rect(
        fitz.Rect(x, y, x + width, y + height),
        color=fill or (0, 0, 0),
        fill=fill,
        width=1,
        overlay=True,
    )


def _draw_qr_code_fitz(page: fitz.Page, qr_code: Mapping[str, Any]) -> None:
    if qr_code.get("kind") != "page_identity_qr":
        raise ValueError("identity_qr_codes.kind must be 'page_identity_qr'")
    if qr_code.get("encoding") != "qr":
        raise ValueError("identity_qr_codes.encoding must be 'qr'")

    qr_x = _require_number(qr_code["x"], "identity_qr_code.x")
    qr_y = _require_number(qr_code["y"], "identity_qr_code.y")
    qr_width = _require_number(qr_code["width"], "identity_qr_code.width")
    qr_height = _require_number(qr_code["height"], "identity_qr_code.height")
    if qr_width != qr_height:
        raise ValueError("identity_qr_codes must be square")

    matrix = _qr_matrix(
        str(qr_code["payload"]),
        error_correction=str(qr_code["error_correction"]),
        border_modules=int(qr_code["border_modules"]),
    )
    module_width = qr_width / len(matrix)
    module_height = qr_height / len(matrix)
    for row_index, row in enumerate(matrix):
        for col_index, is_black in enumerate(row):
            if not is_black:
                continue
            page.draw_rect(
                fitz.Rect(
                    qr_x + (col_index * module_width),
                    qr_y + (row_index * module_height),
                    qr_x + ((col_index + 1) * module_width),
                    qr_y + ((row_index + 1) * module_height),
                ),
                color=(0, 0, 0),
                fill=(0, 0, 0),
                width=0,
                overlay=True,
            )


def _apply_text_overlay(page: fitz.Page, overlay: Mapping[str, Any]) -> None:
    font_size = float(overlay["font_size"])
    if "search_text" in overlay:
        search_text = str(overlay["search_text"])
        replacement_text = str(overlay["replacement_text"])
        matches = page.search_for(search_text)
        if not matches:
            raise ValueError(f"Could not find source overlay text: {search_text!r}")
        rect = fitz.Rect(matches[0])
        for match in matches[1:]:
            rect |= fitz.Rect(match)
        replacement_width = fitz.get_text_length(
            replacement_text,
            fontname="Times-Roman",
            fontsize=font_size,
        )
        wipe_rect = fitz.Rect(
            rect.x0 - 2,
            rect.y0 - 2,
            max(rect.x1, rect.x0 + replacement_width) + 2,
            rect.y1 + 2,
        )
        page.draw_rect(wipe_rect, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
        page.insert_text(
            fitz.Point(rect.x0, rect.y1 - 1),
            replacement_text,
            fontname="Times-Roman",
            fontsize=font_size,
            color=(0, 0, 0),
            overlay=True,
        )
        return
    if "search_lines" in overlay:
        replacement_lines = list(overlay["replacement_lines"])
        for source_line, replacement_line in zip(overlay["search_lines"], replacement_lines, strict=False):
            matches = page.search_for(str(source_line))
            if not matches:
                raise ValueError(f"Could not find source overlay line: {source_line!r}")
            rect = fitz.Rect(matches[0])
            for match in matches[1:]:
                rect |= fitz.Rect(match)
            rect = fitz.Rect(rect.x0 - 4, rect.y0 - 2, rect.x1 + 4, rect.y1 + 2)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
            page.insert_text(
                fitz.Point(rect.x0, rect.y1 - 1),
                str(replacement_line),
                fontname="Times-Roman",
                fontsize=font_size,
                color=(0, 0, 0),
                overlay=True,
            )
        return

    rect = fitz.Rect(*overlay["rect"])
    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
    x = float(overlay["x"])
    y = float(overlay.get("y", rect.y0 + 12))
    line_spacing = float(overlay["line_spacing"])
    for line in overlay["lines"]:
        page.insert_text(
            fitz.Point(x, y),
            str(line),
            fontname="Times-Roman",
            fontsize=font_size,
            color=(0, 0, 0),
            overlay=True,
        )
        y += line_spacing


def _qr_matrix(payload: str, *, error_correction: str, border_modules: int) -> list[list[bool]]:
    correction_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    try:
        correction_level = correction_levels[error_correction]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported QR error correction level: {error_correction!r}"
        ) from exc

    qr = qrcode.QRCode(
        border=border_modules,
        error_correction=correction_level,
        box_size=1,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.get_matrix()


def _escape_text(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_safe_text(text: str) -> str:
    return (
        str(text)
        .replace("×", "x")
        .replace("⁻", "-")
        .replace("⁰", "0")
        .replace("¹", "1")
        .replace("²", "2")
        .replace("³", "3")
        .replace("⁴", "4")
        .replace("⁵", "5")
        .replace("⁶", "6")
        .replace("⁷", "7")
        .replace("⁸", "8")
        .replace("⁹", "9")
        .replace("⇄", "<=>")
    )


def _circle_path(x: int | float, y: int | float, width: int | float, height: int | float) -> str:
    radius_x = width / 2
    radius_y = height / 2
    center_x = x + radius_x
    center_y = y + radius_y
    control = 0.552284749831 * radius_x
    return " ".join(
        [
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y)} m",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y + control)}",
            f"{_pdf_number(center_x + control)} {_pdf_number(center_y + radius_y)}",
            f"{_pdf_number(center_x)} {_pdf_number(center_y + radius_y)} c",
            f"{_pdf_number(center_x - control)} {_pdf_number(center_y + radius_y)}",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y + control)}",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y)} c",
            f"{_pdf_number(center_x - radius_x)} {_pdf_number(center_y - control)}",
            f"{_pdf_number(center_x - control)} {_pdf_number(center_y - radius_y)}",
            f"{_pdf_number(center_x)} {_pdf_number(center_y - radius_y)} c",
            f"{_pdf_number(center_x + control)} {_pdf_number(center_y - radius_y)}",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y - control)}",
            f"{_pdf_number(center_x + radius_x)} {_pdf_number(center_y)} c S",
        ]
    )


def _pdf_rect_y(page_height: int | float, top_y: int | float, rect_height: int | float) -> int | float:
    return page_height - top_y - rect_height


def _pdf_text_y(page_height: int | float, top_y: int | float) -> int | float:
    return page_height - top_y - 10


def _require_number(value: Any, label: str) -> int | float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return value


def _pdf_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _build_pdf(objects: list[bytes]) -> bytes:
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{object_number} 0 obj\n".encode("utf-8"))
        document.extend(body)
        document.extend(b"\nendobj\n")

    startxref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{startxref}\n%%EOF\n"
        ).encode("utf-8")
    )
    return bytes(document)
