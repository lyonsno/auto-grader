"""Deterministic MC answer-sheet generation primitives.

This module now owns the canonical paper artifact contract for the MC/OpenCV
lane:

- stable per-student exam-instance identity codes
- rendered MC questions with deterministic choice shuffling
- answer-key mappings from logical choice key to physical bubble label
- explicit page-space bubble regions for readback
- duplicated QR placements and registration markers for scan-side recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import hashlib
import random
import re
from typing import Any

from auto_grader.template_schema import validate_template


MC_BUBBLE_LABELS = tuple("ABCDE")
_BUBBLE_LABELS = MC_BUBBLE_LABELS
_LETTER_WIDTH = 612
_LETTER_HEIGHT = 792
_LAYOUT_LEFT = 72
_LAYOUT_TOP = 188
_ROW_HEIGHT = 150
_BUBBLE_SIZE = 22
_BUBBLE_GAP = 28
_BUBBLE_ROW_LEFT = 320
_MAX_ROWS_PER_PAGE = 4
_MAX_PAGES = 5
_REGISTRATION_MARKER_SIZE = 18
_REGISTRATION_MARKER_INSET = 24
_IDENTITY_QR_SIZE = 56
_IDENTITY_QR_TOP = 36
_IDENTITY_QR_LEFT = 420
_IDENTITY_QR_GAP = 10

_PAGE_CONTENT_BOTTOM = _LETTER_HEIGHT - _REGISTRATION_MARKER_INSET - _REGISTRATION_MARKER_SIZE - 10

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class McAnswerSheetLayout:
    """Public page-layout surface for MC answer-sheet artifacts."""

    rows_per_page: int = _MAX_ROWS_PER_PAGE
    layout_top: int = _LAYOUT_TOP
    row_height: int = _ROW_HEIGHT
    bubble_row_left: int = _BUBBLE_ROW_LEFT

    def __post_init__(self) -> None:
        if self.rows_per_page <= 0:
            raise ValueError("rows_per_page must be positive")
        if self.row_height <= 0:
            raise ValueError("row_height must be positive")
        if self.bubble_row_left <= 0:
            raise ValueError("bubble_row_left must be positive")
        if self.layout_top < 0:
            raise ValueError("layout_top must be non-negative")


DEFAULT_MC_ANSWER_SHEET_LAYOUT = McAnswerSheetLayout()


def build_mc_answer_sheet(
    template: Mapping[str, Any],
    student: Mapping[str, Any],
    *,
    attempt_number: int,
    seed: int | str,
) -> dict[str, Any]:
    """Build the canonical MC answer-sheet artifact for one student."""
    _validate_attempt_number(attempt_number)
    template_dict = _require_valid_template(template)
    student_id = _require_student_id(student)

    mc_questions = _collect_mc_questions(template_dict)
    if not mc_questions:
        raise ValueError("Template must contain at least one multiple-choice question")
    if len(mc_questions) > (_MAX_ROWS_PER_PAGE * _MAX_PAGES):
        raise ValueError(
            "MC answer-sheet v0 supports at most 20 multiple-choice questions total"
        )

    opaque_instance_code = _build_opaque_instance_code(
        template_slug=str(template_dict["slug"]),
        student_id=student_id,
        attempt_number=attempt_number,
        seed=seed,
    )

    rendered_questions: list[dict[str, Any]] = []
    answer_key: dict[str, dict[str, Any]] = {}

    for question in mc_questions:
        question_seed = _stable_seed(seed, student_id, question["id"])
        question_rng = random.Random(question_seed)
        variable_values = _sample_variables(question.get("variables", {}), question_rng)
        rendered_question = _render_mc_question(question, variable_values, question_rng)
        rendered_questions.append(rendered_question)
        answer_key[rendered_question["question_id"]] = {
            "question_id": rendered_question["question_id"],
            "correct_choice_key": rendered_question["correct_choice_key"],
            "correct_bubble_label": rendered_question["correct_bubble_label"],
            "choices": [
                {
                    "bubble_label": choice["bubble_label"],
                    "choice_key": choice["choice_key"],
                }
                for choice in rendered_question["choices"]
            ],
        }

    pages = build_mc_answer_sheet_pages(
        opaque_instance_code,
        rendered_questions,
        layout=DEFAULT_MC_ANSWER_SHEET_LAYOUT,
    )

    return {
        "template_slug": str(template_dict["slug"]),
        "student_id": student_id,
        "attempt_number": attempt_number,
        "opaque_instance_code": opaque_instance_code,
        "mc_questions": rendered_questions,
        "answer_key": answer_key,
        "pages": pages,
    }


def build_mc_answer_sheets(
    template: Mapping[str, Any],
    roster: Iterable[Mapping[str, Any]],
    *,
    attempt_number: int,
    seed: int | str,
) -> list[dict[str, Any]]:
    """Build MC answer-sheet artifacts for each student in a roster."""
    artifacts: list[dict[str, Any]] = []
    seen_student_ids: set[str] = set()

    for student in roster:
        student_id = _require_student_id(student)
        if student_id in seen_student_ids:
            raise ValueError(f"Duplicate student_id in roster: {student_id!r}")
        seen_student_ids.add(student_id)
        artifacts.append(
            build_mc_answer_sheet(
                template,
                student,
                attempt_number=attempt_number,
                seed=seed,
            )
        )

    return artifacts


def _require_valid_template(template: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(template, Mapping):
        raise TypeError("template must be a mapping")

    template_dict = dict(template)
    errors = validate_template(template_dict)
    if errors:
        raise ValueError(f"Template validation failed: {'; '.join(errors)}")
    return template_dict


def _require_student_id(student: Mapping[str, Any]) -> str:
    if not isinstance(student, Mapping):
        raise TypeError("student must be a mapping")
    student_id = student.get("student_id")
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student.student_id must be a non-blank string")
    return student_id.strip()


def _validate_attempt_number(attempt_number: int) -> None:
    if not isinstance(attempt_number, int) or attempt_number <= 0:
        raise ValueError("attempt_number must be a positive integer")


def _collect_mc_questions(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for section in template.get("sections", []):
        for question in section.get("questions", []):
            questions.extend(_collect_mc_questions_from_node(question))
    return questions


def _collect_mc_questions_from_node(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    if node.get("answer_type") == "multiple_choice":
        return [dict(node)]

    collected: list[dict[str, Any]] = []
    for part in node.get("parts", []):
        collected.extend(_collect_mc_questions_from_node(part))
    return collected


def _sample_variables(
    variable_specs: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in sorted(variable_specs):
        values[name] = _sample_variable(variable_specs[name], rng)
    return values


def _sample_variable(spec: Mapping[str, Any], rng: random.Random) -> Any:
    minimum = spec["min"]
    maximum = spec["max"]
    step = spec["step"]
    count = int(round((maximum - minimum) / step)) + 1
    index = rng.randrange(count)
    value = minimum + (index * step)
    if spec.get("type") == "int":
        return int(round(value))
    return float(value)


def _render_mc_question(
    question: Mapping[str, Any],
    variable_values: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    choice_items = list(question["choices"].items())
    if question.get("shuffle", False):
        rng.shuffle(choice_items)

    if len(choice_items) > len(_BUBBLE_LABELS):
        raise ValueError(
            f"Question {question['id']!r} has {len(choice_items)} choices; only up to 5 are supported"
        )

    rendered_choices: list[dict[str, Any]] = []
    correct_bubble_label: str | None = None

    for bubble_label, (choice_key, choice_text) in zip(_BUBBLE_LABELS, choice_items, strict=False):
        rendered_choices.append(
            {
                "bubble_label": bubble_label,
                "choice_key": choice_key,
                "text": _render_text(str(choice_text), variable_values),
            }
        )
        if choice_key == question["correct"]:
            correct_bubble_label = bubble_label

    if correct_bubble_label is None:
        raise ValueError(
            f"Question {question['id']!r} correct key {question['correct']!r} not found in rendered choices"
        )

    return {
        "question_id": question["id"],
        "prompt": _render_text(str(question.get("prompt", "")), variable_values),
        "points": question["points"],
        "choices": rendered_choices,
        "correct_choice_key": question["correct"],
        "correct_bubble_label": correct_bubble_label,
    }


def _render_text(text: str, variable_values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        value = variable_values.get(variable_name)
        if value is None:
            return match.group(0)
        return _stringify_value(value)

    return _PLACEHOLDER_RE.sub(replace, text)


def _stringify_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_mc_answer_sheet_pages(
    opaque_instance_code: str,
    rendered_questions: list[dict[str, Any]],
    *,
    layout: McAnswerSheetLayout = DEFAULT_MC_ANSWER_SHEET_LAYOUT,
) -> list[dict[str, Any]]:
    """Build page-space MC answer-sheet layouts on the canonical paper substrate."""
    if not isinstance(opaque_instance_code, str) or not opaque_instance_code:
        raise ValueError("opaque_instance_code must be a non-blank string")
    if not isinstance(layout, McAnswerSheetLayout):
        raise TypeError("layout must be a McAnswerSheetLayout")

    from auto_grader.pdf_rendering import (
        _CHOICE_LEGEND_LINE_SPACING,
        _CHOICE_LEGEND_TOP_OFFSET,
        _PROMPT_LINE_GAP,
        _PROMPT_LINE_SPACING,
        _wrap_choice_text,
        _wrap_prompt_text,
        _display_prompt,
    )

    # Compute per-question layout metrics.
    question_metrics: list[dict[str, int]] = []
    for question_number, question in enumerate(rendered_questions, start=1):
        prompt_text = _display_prompt(question_number, str(question.get("prompt", "")))
        prompt_line_count = len(_wrap_prompt_text(prompt_text))

        legend_line_count = 0
        for choice in question.get("choices", []):
            legend_line_count += len(
                _wrap_choice_text(str(choice["bubble_label"]), str(choice.get("text", "")))
            )

        # Above the bubble row: prompt lines + gap
        above_bubbles = _PROMPT_LINE_GAP + _PROMPT_LINE_SPACING * max(0, prompt_line_count - 1)
        # Below the bubble row: choice legend
        below_bubbles = _CHOICE_LEGEND_TOP_OFFSET + _CHOICE_LEGEND_LINE_SPACING * legend_line_count

        content_height = above_bubbles + below_bubbles
        question_metrics.append({
            "above": above_bubbles,
            "below": below_bubbles,
            "row_height": max(layout.row_height, content_height),
        })

    # Pack questions onto pages.  The bubble row sits at cursor_y; content
    # extends above (prompt) and below (legend).  A question fits when its
    # legend bottom (cursor_y + below) stays within the page content area.
    pages: list[dict[str, Any]] = []
    registration_markers = _build_registration_markers()
    question_index = 0

    while question_index < len(rendered_questions):
        page_index = len(pages) + 1
        bubble_regions: list[dict[str, Any]] = []
        fallback_page_code = f"{opaque_instance_code}-p{page_index}"
        cursor_y = layout.layout_top
        rows_on_page = 0

        while question_index < len(rendered_questions):
            if rows_on_page >= layout.rows_per_page:
                break
            metrics = question_metrics[question_index]
            # Check if this question's legend bottom fits on the page.
            if rows_on_page > 0 and cursor_y + metrics["below"] > _PAGE_CONTENT_BOTTOM:
                break

            question = rendered_questions[question_index]
            bubbles_x = layout.bubble_row_left
            for choice_index, choice in enumerate(question["choices"]):
                bubble_regions.append(
                    {
                        "question_id": question["question_id"],
                        "bubble_label": choice["bubble_label"],
                        "shape": "circle",
                        "x": bubbles_x + choice_index * (_BUBBLE_SIZE + _BUBBLE_GAP),
                        "y": cursor_y,
                        "width": _BUBBLE_SIZE,
                        "height": _BUBBLE_SIZE,
                    }
                )

            cursor_y += metrics["row_height"]
            rows_on_page += 1
            question_index += 1

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


def _build_registration_markers() -> list[dict[str, Any]]:
    size = _REGISTRATION_MARKER_SIZE
    inset = _REGISTRATION_MARKER_INSET
    return [
        {
            "marker_id": "top_left",
            "kind": "square",
            "x": inset,
            "y": inset,
            "width": size,
            "height": size,
        },
        {
            "marker_id": "top_right",
            "kind": "square",
            "x": _LETTER_WIDTH - inset - size,
            "y": inset,
            "width": size,
            "height": size,
        },
        {
            "marker_id": "bottom_left",
            "kind": "square",
            "x": inset,
            "y": _LETTER_HEIGHT - inset - size,
            "width": size,
            "height": size,
        },
        {
            "marker_id": "bottom_right",
            "kind": "square",
            "x": _LETTER_WIDTH - inset - size,
            "y": _LETTER_HEIGHT - inset - size,
            "width": size,
            "height": size,
        },
    ]


def _build_identity_qr_codes(payload: str) -> list[dict[str, Any]]:
    return [
        {
            "qr_id": "header_left",
            "kind": "page_identity_qr",
            "encoding": "qr",
            "payload": payload,
            "error_correction": "M",
            "border_modules": 4,
            "x": _IDENTITY_QR_LEFT,
            "y": _IDENTITY_QR_TOP,
            "width": _IDENTITY_QR_SIZE,
            "height": _IDENTITY_QR_SIZE,
        },
        {
            "qr_id": "header_right",
            "kind": "page_identity_qr",
            "encoding": "qr",
            "payload": payload,
            "error_correction": "M",
            "border_modules": 4,
            "x": _IDENTITY_QR_LEFT + _IDENTITY_QR_SIZE + _IDENTITY_QR_GAP,
            "y": _IDENTITY_QR_TOP,
            "width": _IDENTITY_QR_SIZE,
            "height": _IDENTITY_QR_SIZE,
        },
    ]


def _build_opaque_instance_code(
    *,
    template_slug: str,
    student_id: str,
    attempt_number: int,
    seed: int | str,
) -> str:
    digest = hashlib.sha256(
        f"opaque-instance|{template_slug}|{student_id}|{attempt_number}|{seed}".encode("utf-8")
    ).hexdigest()[:20]
    return f"inst_{digest}"


def _stable_seed(seed: int | str, student_id: str, question_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{student_id}|{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
