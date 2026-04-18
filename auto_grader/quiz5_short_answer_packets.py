"""Printable Quiz #5 short-answer packets built from the reconstructed family."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import textwrap
from typing import Any

import yaml

from auto_grader.quiz5_short_answer_reconstruction import (
    build_generated_short_answer_variant,
    reconstruct_short_answer_quiz_family,
)


_LETTER_WIDTH = 612
_LETTER_HEIGHT = 792
_REGISTRATION_MARKER_SIZE = 18
_REGISTRATION_MARKER_INSET = 24
_IDENTITY_QR_SIZE = 30
_IDENTITY_QR_TOP = 28
_IDENTITY_QR_LEFT = 490
_IDENTITY_QR_GAP = 6


def build_quiz5_short_answer_variant_packet(
    family: Mapping[str, Any],
    *,
    variant_id: str,
    opaque_instance_code: str,
) -> dict[str, Any]:
    if not isinstance(family, Mapping):
        raise TypeError("family must be a mapping")
    if not isinstance(opaque_instance_code, str) or not opaque_instance_code.strip():
        raise ValueError("opaque_instance_code must be a non-empty string")

    resolved_variant = _resolve_variant(family, variant_id=variant_id)
    pages = _build_pages(
        prompts=resolved_variant["printable_prompts"],
        opaque_instance_code=opaque_instance_code.strip(),
    )
    return {
        "layout_version": "quiz5_short_answer_packet_v1",
        "template_slug": family["slug"],
        "title": family["title"],
        "variant_id": variant_id,
        "opaque_instance_code": opaque_instance_code.strip(),
        "source_variant_ids": resolved_variant["source_variant_ids"],
        "variables": resolved_variant["variables"],
        "pages": pages,
    }


def write_quiz5_short_answer_variant_bundle(
    *,
    pdf_paths: Iterable[str | Path],
    output_dir: str | Path,
    generated_variant_ids: Iterable[str] = (),
    database_url: str | None = None,
) -> dict[str, Any]:
    from auto_grader.db import create_connection, initialize_schema
    from auto_grader.pdf_rendering import render_quiz5_short_answer_pdf

    family = reconstruct_short_answer_quiz_family(pdf_paths)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    family_json_path = output_path / "short-answer-quiz-family.json"
    family_json_path.write_text(
        json.dumps(family, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    requested_variant_ids = sorted(set(family["variants"]).union(set(generated_variant_ids)))
    artifacts: dict[str, dict[str, Any]] = {}
    packets: dict[str, dict[str, Any]] = {}
    for variant_id in requested_variant_ids:
        opaque_instance_code = _default_instance_code(variant_id)
        packet = build_quiz5_short_answer_variant_packet(
            family,
            variant_id=variant_id,
            opaque_instance_code=opaque_instance_code,
        )
        packets[variant_id] = packet

        artifact_json_path = output_path / f"quiz5-{variant_id.lower()}-artifact.json"
        artifact_json_path.write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        pdf_path = output_path / f"quiz5-{variant_id.lower()}.pdf"
        pdf_path.write_bytes(render_quiz5_short_answer_pdf(packet))
        artifacts[variant_id] = {
            "artifact_json_path": str(artifact_json_path),
            "pdf_path": str(pdf_path),
            "opaque_instance_code": opaque_instance_code,
        }

    result: dict[str, Any] = {
        "family_json_path": str(family_json_path),
        "variant_ids": requested_variant_ids,
        "artifacts": artifacts,
    }

    if database_url:
        connection = create_connection(database_url)
        try:
            initialize_schema(connection)
            result["db"] = register_quiz5_short_answer_variants(
                family=family,
                packets=packets,
                connection=connection,
            )
        finally:
            connection.close()

    return result


def register_quiz5_short_answer_variants(
    *,
    family: Mapping[str, Any],
    packets: Mapping[str, Mapping[str, Any]],
    connection: object,
) -> dict[str, Any]:
    template = dict(family["template"])
    source_yaml = yaml.safe_dump(template, sort_keys=False, allow_unicode=True)

    with connection.transaction():
        template_version_id = _get_or_create_template_version(
            slug=str(template["slug"]),
            source_yaml=source_yaml,
            connection=connection,
        )
        exam_definition_id = _get_or_create_exam_definition(
            slug=str(template["slug"]),
            title=str(family["title"]),
            template_version_id=template_version_id,
            connection=connection,
        )

        variants: dict[str, Any] = {}
        for variant_id in sorted(packets):
            packet = packets[variant_id]
            student_id = _get_or_create_student(
                student_key=f"quiz5-variant::{variant_id}",
                connection=connection,
            )
            exam_instance = _get_or_create_exam_instance(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                opaque_instance_code=str(packet["opaque_instance_code"]),
                connection=connection,
            )
            page_rows = _get_or_create_exam_pages(
                exam_instance_id=exam_instance["id"],
                pages=packet["pages"],
                connection=connection,
            )
            variants[variant_id] = {
                "exam_instance_id": exam_instance["id"],
                "opaque_instance_code": exam_instance["opaque_instance_code"],
                "page_codes": [row["fallback_page_code"] for row in page_rows],
            }

    return {
        "template_version_id": template_version_id,
        "exam_definition_id": exam_definition_id,
        "variants": variants,
    }


def _resolve_variant(family: Mapping[str, Any], *, variant_id: str) -> dict[str, Any]:
    if variant_id in family["variants"]:
        variables = dict(family["variants"][variant_id]["variables"])
        return {
            "source_variant_ids": [variant_id],
            "variables": variables,
            "printable_prompts": _build_printable_prompts(variables),
        }

    generated = build_generated_short_answer_variant(dict(family), variant_id=variant_id)
    variables = dict(generated["variables"])
    return {
        "source_variant_ids": generated["source_variant_ids"],
        "variables": variables,
        "printable_prompts": _build_printable_prompts(variables),
    }


def _build_printable_prompts(variables: Mapping[str, Any]) -> list[dict[str, str]]:
    acid_molarity = _ascii_scientific_notation(float(variables["acid_molarity"]), sig_figs=3)
    return [
        {
            "id": "q1a",
            "response_box_label": "1a.",
            "prompt": (
                f"Write a net ionic equation to show how {variables['bronsted_base']} "
                "behaves as a Bronsted base in water."
            ),
        },
        {
            "id": "q1b",
            "response_box_label": "1b.",
            "prompt": (
                f"Write a net ionic equation to show how {variables['bronsted_acid']} "
                "behaves as a Bronsted acid in water."
            ),
        },
        {
            "id": "q2",
            "response_box_label": "2.",
            "prompt": (
                "What is the pH of an aqueous solution of "
                f"{acid_molarity} M {variables['acid_species']}?"
            ),
        },
        {
            "id": "q3",
            "response_box_label": "3.",
            "prompt": (
                f"What is the pH of a {variables['base_molarity']:.4f}".rstrip("0").rstrip(".")
                + " M aqueous solution of sodium hydroxide?"
            ),
        },
        {
            "id": "q4",
            "response_box_label": "4.",
            "prompt": (
                "What concentration of nitric acid is needed to make an aqueous "
                f"solution with a pH of {variables['target_ph']:.2f}?"
            ),
        },
        {
            "id": "q5",
            "response_box_label": "5.",
            "prompt": (
                "For 2SO3(g) <=> 2SO2(g) + O2(g), the equilibrium constant Kc is "
                f"{variables['kc_q5']:.4f} at 1200 K. If a 1.00 L equilibrium mixture "
                "contains 0.200 mol SO3 and 0.387 mol SO2, calculate the equilibrium "
                "concentration of O2."
            ),
        },
        {
            "id": "q6-ch4",
            "response_box_label": "6.[CH4]=",
            "prompt": (
                "For CH4(g) + CCl4(g) <=> 2CH2Cl2(g), the equilibrium constant Kc is "
                f"{variables['kc_q6']:.4f} at 548 K. Calculate the equilibrium "
                "concentrations when 0.375 mol of CH4 and 0.375 mol of CCl4 are "
                "introduced into a 1.00 L vessel."
            ),
        },
        {
            "id": "q6-ccl4",
            "response_box_label": "6. [CCl4]=",
            "prompt": "Same equilibrium problem as 6: report the equilibrium concentration of CCl4.",
        },
        {
            "id": "q6-ch2cl2",
            "response_box_label": "6. [CH2Cl2]=",
            "prompt": "Same equilibrium problem as 6: report the equilibrium concentration of CH2Cl2.",
        },
    ]


def _build_pages(
    *,
    prompts: list[dict[str, str]],
    opaque_instance_code: str,
) -> list[dict[str, Any]]:
    prompt_index = {entry["id"]: entry for entry in prompts}
    return [
        {
            "layout_version": "quiz5_short_answer_v1",
            "units": "pt",
            "origin": "top_left",
            "y_axis": "down",
            "width": _LETTER_WIDTH,
            "height": _LETTER_HEIGHT,
            "page_number": 1,
            "fallback_page_code": f"{opaque_instance_code}-p1",
            "registration_markers": _build_registration_markers(),
            "identity_qr_codes": _build_identity_qr_codes(f"{opaque_instance_code}-p1"),
            "prompt_blocks": [
                _prompt_block(
                    prompt_index["q1a"],
                    x=152,
                    y=378,
                    width=392,
                    wrap_width=64,
                    display_prefix="a.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
                _prompt_block(
                    prompt_index["q1b"],
                    x=152,
                    y=466,
                    width=392,
                    wrap_width=64,
                    display_prefix="b.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
                _prompt_block(
                    prompt_index["q2"],
                    x=114,
                    y=584,
                    width=432,
                    wrap_width=78,
                    display_prefix="2.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
                _prompt_block(
                    prompt_index["q3"],
                    x=114,
                    y=664,
                    width=432,
                    wrap_width=78,
                    display_prefix="3.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
                _prompt_block(
                    prompt_index["q4"],
                    x=114,
                    y=730,
                    width=432,
                    wrap_width=76,
                    display_prefix="4.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
            ],
            "response_boxes": [
                _response_box(prompt_index["q1a"], x=86, y=430, width=436, height=34),
                _response_box(prompt_index["q1b"], x=86, y=518, width=436, height=34),
                _response_box(prompt_index["q2"], x=86, y=624, width=92, height=28),
                _response_box(prompt_index["q3"], x=86, y=704, width=92, height=28),
                _response_box(prompt_index["q4"], x=86, y=760, width=92, height=24),
            ],
        },
        {
            "layout_version": "quiz5_short_answer_v1",
            "units": "pt",
            "origin": "top_left",
            "y_axis": "down",
            "width": _LETTER_WIDTH,
            "height": _LETTER_HEIGHT,
            "page_number": 2,
            "fallback_page_code": f"{opaque_instance_code}-p2",
            "registration_markers": _build_registration_markers(),
            "identity_qr_codes": _build_identity_qr_codes(f"{opaque_instance_code}-p2"),
            "prompt_blocks": [
                _prompt_block(
                    prompt_index["q5"],
                    x=114,
                    y=190,
                    width=432,
                    wrap_width=74,
                    display_prefix="5.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
                _prompt_block(
                    prompt_index["q6-ch4"],
                    x=114,
                    y=336,
                    width=432,
                    wrap_width=74,
                    display_prefix="6.  ",
                    font_size=10.5,
                    line_spacing=13,
                ),
            ],
            "response_boxes": [
                _response_box(prompt_index["q5"], x=86, y=254, width=92, height=34),
                _response_box(prompt_index["q6-ch4"], x=86, y=430, width=110, height=34),
                _response_box(prompt_index["q6-ccl4"], x=86, y=482, width=110, height=34),
                _response_box(prompt_index["q6-ch2cl2"], x=86, y=534, width=126, height=34),
            ],
        },
    ]


def _prompt_block(
    entry: Mapping[str, str],
    *,
    x: int,
    y: int,
    width: int,
    wrap_width: int = 64,
    display_prefix: str = "",
    font_size: float = 12,
    line_spacing: float = 15,
) -> dict[str, Any]:
    wrapped_lines = textwrap.wrap(f"{display_prefix}{entry['prompt']}", width=wrap_width)
    return {
        "question_id": entry["id"],
        "text": entry["prompt"],
        "wrapped_lines": wrapped_lines,
        "x": x,
        "y": y,
        "width": width,
        "font_size": font_size,
        "line_spacing": line_spacing,
    }


def _response_box(
    entry: Mapping[str, str],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "question_id": entry["id"],
        "label": entry["response_box_label"],
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _build_registration_markers() -> list[dict[str, Any]]:
    return [
        _registration_marker("top_left", _REGISTRATION_MARKER_INSET, _REGISTRATION_MARKER_INSET),
        _registration_marker(
            "top_right",
            _LETTER_WIDTH - _REGISTRATION_MARKER_INSET - _REGISTRATION_MARKER_SIZE,
            _REGISTRATION_MARKER_INSET,
        ),
        _registration_marker(
            "bottom_left",
            _REGISTRATION_MARKER_INSET,
            _LETTER_HEIGHT - _REGISTRATION_MARKER_INSET - _REGISTRATION_MARKER_SIZE,
        ),
        _registration_marker(
            "bottom_right",
            _LETTER_WIDTH - _REGISTRATION_MARKER_INSET - _REGISTRATION_MARKER_SIZE,
            _LETTER_HEIGHT - _REGISTRATION_MARKER_INSET - _REGISTRATION_MARKER_SIZE,
        ),
    ]


def _registration_marker(marker_id: str, x: int, y: int) -> dict[str, Any]:
    return {
        "marker_id": marker_id,
        "x": x,
        "y": y,
        "width": _REGISTRATION_MARKER_SIZE,
        "height": _REGISTRATION_MARKER_SIZE,
    }


def _build_identity_qr_codes(payload: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": "page_identity_qr",
            "encoding": "qr",
            "payload": payload,
            "x": _IDENTITY_QR_LEFT,
            "y": _IDENTITY_QR_TOP,
            "width": _IDENTITY_QR_SIZE,
            "height": _IDENTITY_QR_SIZE,
            "error_correction": "M",
            "border_modules": 1,
        },
        {
            "kind": "page_identity_qr",
            "encoding": "qr",
            "payload": payload,
            "x": _IDENTITY_QR_LEFT + _IDENTITY_QR_SIZE + _IDENTITY_QR_GAP,
            "y": _IDENTITY_QR_TOP,
            "width": _IDENTITY_QR_SIZE,
            "height": _IDENTITY_QR_SIZE,
            "error_correction": "M",
            "border_modules": 1,
        },
    ]


def _ascii_scientific_notation(value: float, *, sig_figs: int) -> str:
    mantissa, exponent = f"{value:.{sig_figs - 1}e}".split("e")
    return f"{mantissa} x 10^{int(exponent)}"


def _default_instance_code(variant_id: str) -> str:
    return f"QUIZ5-{variant_id}"


def _get_or_create_template_version(*, slug: str, source_yaml: str, connection: object) -> int:
    existing = connection.execute(
        "SELECT id FROM template_versions WHERE slug = %s AND version = 1",
        (slug,),
    ).fetchone()
    if existing is not None:
        return existing["id"]

    return connection.execute(
        "INSERT INTO template_versions (slug, version, source_yaml) VALUES (%s, 1, %s) RETURNING id",
        (slug, source_yaml),
    ).fetchone()["id"]


def _get_or_create_exam_definition(
    *,
    slug: str,
    title: str,
    template_version_id: int,
    connection: object,
) -> int:
    existing = connection.execute(
        "SELECT id FROM exam_definitions WHERE slug = %s AND version = 1",
        (slug,),
    ).fetchone()
    if existing is not None:
        return existing["id"]

    return connection.execute(
        "INSERT INTO exam_definitions (slug, version, title, template_version_id) VALUES (%s, 1, %s, %s) RETURNING id",
        (slug, title, template_version_id),
    ).fetchone()["id"]


def _get_or_create_student(*, student_key: str, connection: object) -> int:
    existing = connection.execute(
        "SELECT id FROM students WHERE student_key = %s",
        (student_key,),
    ).fetchone()
    if existing is not None:
        return existing["id"]
    return connection.execute(
        "INSERT INTO students (student_key) VALUES (%s) RETURNING id",
        (student_key,),
    ).fetchone()["id"]


def _get_or_create_exam_instance(
    *,
    exam_definition_id: int,
    student_id: int,
    opaque_instance_code: str,
    connection: object,
) -> dict[str, Any]:
    existing = connection.execute(
        """
        SELECT id, opaque_instance_code
        FROM exam_instances
        WHERE exam_definition_id = %s AND student_id = %s AND attempt_number = 1
        """,
        (exam_definition_id, student_id),
    ).fetchone()
    if existing is not None:
        return dict(existing)

    return dict(
        connection.execute(
            """
            INSERT INTO exam_instances
            (exam_definition_id, student_id, attempt_number, opaque_instance_code)
            VALUES (%s, %s, 1, %s)
            RETURNING id, opaque_instance_code
            """,
            (exam_definition_id, student_id, opaque_instance_code),
        ).fetchone()
    )


def _get_or_create_exam_pages(
    *,
    exam_instance_id: int,
    pages: Iterable[Mapping[str, Any]],
    connection: object,
) -> list[dict[str, Any]]:
    created_or_existing: list[dict[str, Any]] = []
    for page in pages:
        existing = connection.execute(
            """
            SELECT id, page_number, fallback_page_code
            FROM exam_pages
            WHERE exam_instance_id = %s AND page_number = %s
            """,
            (exam_instance_id, page["page_number"]),
        ).fetchone()
        if existing is not None:
            created_or_existing.append(dict(existing))
            continue
        created_or_existing.append(
            dict(
                connection.execute(
                    """
                    INSERT INTO exam_pages (exam_instance_id, page_number, fallback_page_code)
                    VALUES (%s, %s, %s)
                    RETURNING id, page_number, fallback_page_code
                    """,
                    (exam_instance_id, page["page_number"], page["fallback_page_code"]),
                ).fetchone()
            )
        )
    return created_or_existing
