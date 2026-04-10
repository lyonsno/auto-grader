"""Minimal PDF rendering for deterministic MC answer sheets.

This module intentionally keeps the paper surface small and explicit:

- consume the generation artifact as the sole layout truth
- render visible recovery codes onto the page
- draw the exact bubble rectangles recorded in page space
- avoid heavyweight PDF dependencies while the contract is still moving
"""

from __future__ import annotations

from collections.abc import Mapping
import textwrap
from typing import Any


_PROMPT_OFFSET = 96
_PROMPT_LINE_GAP = 28
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
_CHOICE_LEGEND_TOP_OFFSET = 4
_CHOICE_LEGEND_LINE_SPACING = 14
_PROMPT_LINE_SPACING = 14
_PROMPT_WRAP_WIDTH = 52


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
        prompt_lines = _wrap_prompt_text(f"{question_number}. {question.get('prompt', '')}")
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

        for index, choice in enumerate(choices):
            content_lines.extend(
                _text_block(
                    choice_legend_left,
                    _pdf_text_y(
                        height,
                        prompt_y_top + _CHOICE_LEGEND_TOP_OFFSET + (index * _CHOICE_LEGEND_LINE_SPACING),
                    ),
                    _CHOICE_LEGEND_FONT_SIZE,
                    f"{choice['bubble_label']}. {choice.get('text', '')}",
                )
            )

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


def _text_block(x: int | float, y: int | float, font_size: int | float, text: str) -> list[str]:
    return [
        "BT",
        f"/F1 {_pdf_number(font_size)} Tf",
        f"{_pdf_number(x)} {_pdf_number(y)} Td",
        f"({_escape_text(text)}) Tj",
        "ET",
    ]


def _escape_text(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_prompt_text(text: str) -> list[str]:
    wrapped = textwrap.wrap(
        text,
        width=_PROMPT_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [text]


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
