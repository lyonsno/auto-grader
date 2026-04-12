"""Build and write a printable MC demo packet from a real exam template.

This surface exists to prove the landed MC/OpenCV pipeline on an actual
generated exam, not just on synthetic calibration packets. It deliberately
reuses the canonical template-loading, generation, and PDF-rendering paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

from auto_grader.generation import build_mc_answer_sheet
from auto_grader.template_schema import load_template

_REPO_ROOT = Path(__file__).resolve().parent.parent


def build_generated_mc_exam_demo_packet(
    *,
    template_path: str | Path,
    student_id: str,
    student_name: str,
    attempt_number: int,
    seed: int | str = 17,
) -> dict[str, Any]:
    """Return one printable generated-MC packet from a real template file."""
    resolved_template_path = _resolve_template_path(template_path)
    template = _load_template_from_path(resolved_template_path)
    artifact = build_mc_answer_sheet(
        template,
        {"student_id": student_id, "student_name": student_name},
        attempt_number=attempt_number,
        seed=seed,
    )

    return {
        "template_path": str(resolved_template_path),
        "artifact": artifact,
        "metadata": _packet_metadata(
            artifact=artifact,
            template_path=resolved_template_path,
            student_id=student_id,
            student_name=student_name,
            attempt_number=attempt_number,
            seed=seed,
        ),
    }


def write_generated_mc_exam_demo_packet(
    *,
    output_dir: str | Path,
    template_path: str | Path,
    student_id: str,
    student_name: str,
    attempt_number: int,
    seed: int | str = 17,
) -> dict[str, Any]:
    """Write a printable generated-MC exam packet and its artifact metadata."""
    from auto_grader.pdf_rendering import render_mc_answer_sheet_pdf

    packet = build_generated_mc_exam_demo_packet(
        template_path=template_path,
        student_id=student_id,
        student_name=student_name,
        attempt_number=attempt_number,
        seed=seed,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / "mc-generated-exam-demo.pdf"
    artifact_path = output_path / "mc-generated-exam-demo-artifact.json"
    metadata_path = output_path / "mc-generated-exam-demo-metadata.json"

    pdf_path.write_bytes(render_mc_answer_sheet_pdf(packet["artifact"]))
    artifact_path.write_text(json.dumps(packet["artifact"], indent=2), encoding="utf-8")

    metadata = dict(packet["metadata"])
    metadata["artifact_path"] = str(artifact_path)
    metadata["pdf_path"] = str(pdf_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "pdf_path": str(pdf_path),
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "page_count": len(packet["artifact"]["pages"]),
        "question_count": len(packet["artifact"]["mc_questions"]),
        "template_slug": packet["artifact"]["template_slug"],
    }


def _load_template_from_path(template_path: Path) -> Mapping[str, Any]:
    yaml_text = template_path.read_text(encoding="utf-8")
    return load_template(yaml_text)


def _resolve_template_path(template_path: str | Path) -> Path:
    candidate = Path(template_path)
    if candidate.is_absolute():
        return candidate
    return _REPO_ROOT / candidate


def _packet_metadata(
    *,
    artifact: Mapping[str, Any],
    template_path: Path,
    student_id: str,
    student_name: str,
    attempt_number: int,
    seed: int | str,
) -> dict[str, Any]:
    return {
        "template_path": str(template_path),
        "template_slug": artifact["template_slug"],
        "student_id": student_id,
        "student_name": student_name,
        "attempt_number": attempt_number,
        "seed": seed,
        "opaque_instance_code": artifact["opaque_instance_code"],
        "page_count": len(artifact["pages"]),
        "question_count": len(artifact["mc_questions"]),
        "question_ids": [question["question_id"] for question in artifact["mc_questions"]],
    }
