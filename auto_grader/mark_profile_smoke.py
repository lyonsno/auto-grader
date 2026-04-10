"""Synthetic student-like bubble mark smoke for the MC/OpenCV lane."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import qrcode

from auto_grader.generation import build_mc_answer_sheet
from auto_grader.mc_page_extraction import extract_scored_mc_page
from auto_grader.scan_readback import read_page_identity_qr_payload


def run_mark_profile_smoke(
    *,
    output_dir: Path,
    seed: int = 17,
) -> dict[str, Any]:
    """Render a deterministic matrix of student-like bubble marks and read them back."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(seed=seed)
    page = artifact["pages"][0]
    answer_key = artifact["answer_key"]
    correct_bubble_label = answer_key["mc-1"]["correct_bubble_label"]
    alternate_bubble_label = next(
        choice["bubble_label"]
        for choice in answer_key["mc-1"]["choices"]
        if choice["bubble_label"] != correct_bubble_label
    )

    base_page = _render_base_page(page)
    Image.fromarray(base_page).save(output_dir / "base_page.png")

    profiles = [
        {
            "profile_id": "solid_center",
            "description": "Dark centered fill",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "light_center",
            "description": "Lighter centered pencil fill",
            "mark_fn": lambda draw, bubble, scale: _draw_light_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "scribble_center",
            "description": "Scribbly centered student fill",
            "mark_fn": lambda draw, bubble, scale: _draw_scribble_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "off_center_patch",
            "description": "Off-center but still intentional fill inside the bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_off_center_patch(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "edge_smudge",
            "description": "Edge-only smudge with no center fill",
            "mark_fn": lambda draw, bubble, scale: _draw_edge_smudge(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "faint_center",
            "description": "Very faint centered fill near the current blank boundary",
            "mark_fn": lambda draw, bubble, scale: _draw_faint_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "double_mark",
            "description": "Two clearly filled bubbles",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label, alternate_bubble_label],
        },
        {
            "profile_id": "hostile_correct_plus_glance",
            "description": "Correct fill plus a glancing stray on a neighboring wrong bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "stray_marks": [
                {
                    "bubble_label": alternate_bubble_label,
                    "draw_fn": _draw_glancing_stray,
                }
            ],
        },
        {
            "profile_id": "ambiguous_patchy_center",
            "description": "Patchy center fill near the current ambiguous boundary",
            "mark_fn": lambda draw, bubble, scale: _draw_patchy_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
        {
            "profile_id": "illegible_scratchout",
            "description": "Scratchy unreadable fill that drags across the bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_illegible_scratchout(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
        },
    ]

    report_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        rendered = _render_profile_page(
            base_page,
            page,
            bubble_labels=profile["bubble_labels"],
            mark_fn=profile["mark_fn"],
            stray_marks=profile.get("stray_marks"),
        )
        degraded = _degrade_scan(rendered)
        image_path = output_dir / f"{profile['profile_id']}.png"
        Image.fromarray(degraded).save(image_path)

        qr_payload = read_page_identity_qr_payload(degraded)
        extracted = extract_scored_mc_page(degraded, page, answer_key)
        normalized_path = output_dir / f"{profile['profile_id']}_normalized.png"
        Image.fromarray(extracted["normalized_image"]).save(normalized_path)

        question_result = extracted["scored_questions"]["mc-1"]
        question_observation = extracted["bubble_observations"]["mc-1"]
        report_profiles.append(
            {
                "profile_id": profile["profile_id"],
                "description": profile["description"],
                "image_path": str(image_path),
                "normalized_image_path": str(normalized_path),
                "decoded_payload": qr_payload,
                "observed_marked_bubble_labels": extracted["marked_bubble_labels"]["mc-1"],
                "observed_ambiguous_bubble_labels": question_observation["ambiguous_bubble_labels"],
                "observed_illegible_bubble_labels": question_observation["illegible_bubble_labels"],
                "observed_status": question_result["status"],
                "review_required": question_result["review_required"],
            }
        )

    report = {
        "page_code": page["fallback_page_code"],
        "profiles": report_profiles,
    }
    (output_dir / "summary.json").write_text(_json_dump(report))
    return report


def _build_artifact(seed: int) -> dict[str, Any]:
    template = {
        "slug": "mc-mark-profile-smoke",
        "title": "MC Mark Profile Smoke",
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": [
                    {
                        "id": "mc-1",
                        "points": 2,
                        "answer_type": "multiple_choice",
                        "prompt": "Which species is elemental oxygen?",
                        "choices": {
                            "A": "CO2",
                            "B": "O2",
                            "C": "H2O",
                            "D": "NaCl",
                        },
                        "correct": "B",
                        "shuffle": True,
                    }
                ],
            }
        ],
    }
    return build_mc_answer_sheet(
        template,
        {"student_id": "s-001", "student_name": "Ada Lovelace"},
        attempt_number=1,
        seed=seed,
    )


def _render_base_page(page: Mapping[str, Any], *, scale: int = 4) -> np.ndarray:
    canvas = Image.new("RGB", (int(page["width"] * scale), int(page["height"] * scale)), "white")
    draw = ImageDraw.Draw(canvas)

    for marker in page["registration_markers"]:
        draw.rectangle(
            [
                marker["x"] * scale,
                marker["y"] * scale,
                (marker["x"] + marker["width"]) * scale,
                (marker["y"] + marker["height"]) * scale,
            ],
            fill="black",
        )

    correction_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    for qr_code in page["identity_qr_codes"]:
        qr = qrcode.QRCode(
            border=qr_code["border_modules"],
            error_correction=correction_levels[qr_code["error_correction"]],
            box_size=1,
        )
        qr.add_data(qr_code["payload"])
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_image = qr_image.resize(
            (int(qr_code["width"] * scale), int(qr_code["height"] * scale)),
            Image.Resampling.NEAREST,
        )
        canvas.paste(qr_image, (int(qr_code["x"] * scale), int(qr_code["y"] * scale)))

    for bubble in page["bubble_regions"]:
        left = bubble["x"] * scale
        top = bubble["y"] * scale
        right = (bubble["x"] + bubble["width"]) * scale
        bottom = (bubble["y"] + bubble["height"]) * scale
        draw.ellipse([left, top, right, bottom], outline="black", width=max(2, scale))

    return np.array(canvas)


def _render_profile_page(
    base_page: np.ndarray,
    page: Mapping[str, Any],
    *,
    bubble_labels: list[str],
    mark_fn: Callable[[ImageDraw.ImageDraw, Mapping[str, Any], int], None],
    stray_marks: list[Mapping[str, Any]] | None = None,
    scale: int = 4,
) -> np.ndarray:
    canvas = Image.fromarray(base_page.copy())
    draw = ImageDraw.Draw(canvas)
    for bubble in page["bubble_regions"]:
        if bubble["question_id"] != "mc-1" or bubble["bubble_label"] not in bubble_labels:
            continue
        mark_fn(draw, bubble, scale=scale)
    for stray_mark in stray_marks or []:
        for bubble in page["bubble_regions"]:
            if bubble["question_id"] != "mc-1" or bubble["bubble_label"] != stray_mark["bubble_label"]:
                continue
            stray_mark["draw_fn"](draw, bubble, scale=scale)
            break
    return np.array(canvas)


def _draw_dark_center(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    inset = 0.22 * bubble["width"] * scale
    draw.ellipse([left + inset, top + inset, right - inset, bottom - inset], fill="black")


def _draw_light_center(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    inset = 0.33 * bubble["width"] * scale
    draw.ellipse(
        [left + inset, top + inset, right - inset, bottom - inset],
        fill=(188, 188, 188),
    )


def _draw_faint_center(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    inset = 0.34 * bubble["width"] * scale
    draw.ellipse(
        [left + inset, top + inset, right - inset, bottom - inset],
        fill=(214, 214, 214),
    )


def _draw_scribble_center(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    width = max(2, scale)
    draw.line(
        [
            (left + 12, top + 16),
            (left + 34, top + 28),
            (left + 18, top + 42),
            (left + 40, top + 52),
        ],
        fill="black",
        width=width,
    )
    draw.line(
        [
            (left + 16, top + 48),
            (left + 30, top + 20),
            (left + 42, top + 46),
        ],
        fill="black",
        width=width,
    )
    draw.line(
        [
            (left + 20, top + 14),
            (left + 26, top + 36),
            (left + 46, top + 32),
        ],
        fill="black",
        width=width,
    )
    draw.ellipse(
        [left + 20, top + 22, right - 18, bottom - 18],
        fill=(90, 90, 90),
    )


def _draw_off_center_patch(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    draw.ellipse(
        [left + 22, top + 16, right - 16, bottom - 20],
        fill=(160, 160, 160),
    )


def _draw_edge_smudge(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, _, _ = _bubble_box(bubble, scale)
    width = bubble["width"] * scale
    height = bubble["height"] * scale
    draw.ellipse(
        [
            left + 1,
            top + (0.35 * height),
            left + (0.28 * width),
            top + (0.70 * height),
        ],
        fill="black",
    )


def _draw_patchy_center(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    width = right - left
    height = bottom - top
    draw.ellipse(
        [
            left + (0.37 * width),
            top + (0.35 * height),
            left + (0.53 * width),
            top + (0.51 * height),
        ],
        fill=(198, 198, 198),
    )
    draw.ellipse(
        [
            left + (0.49 * width),
            top + (0.47 * height),
            left + (0.62 * width),
            top + (0.60 * height),
        ],
        fill=(206, 206, 206),
    )


def _draw_illegible_scratchout(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    width = max(3, scale + 1)
    span_x = right - left
    span_y = bottom - top
    draw.line(
        [
            left - (0.08 * span_x),
            top + (0.18 * span_y),
            right + (0.08 * span_x),
            bottom - (0.18 * span_y),
        ],
        fill="black",
        width=width,
    )
    draw.line(
        [
            left + (0.12 * span_x),
            bottom + (0.04 * span_y),
            right - (0.08 * span_x),
            top - (0.05 * span_y),
        ],
        fill="black",
        width=width,
    )
    draw.line(
        [
            left - (0.04 * span_x),
            top + (0.58 * span_y),
            right + (0.06 * span_x),
            top + (0.38 * span_y),
        ],
        fill="black",
        width=width,
    )


def _draw_glancing_stray(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    span_x = right - left
    span_y = bottom - top
    draw.line(
        [
            left + (0.03 * span_x),
            top + (0.56 * span_y),
            left + (0.31 * span_x),
            top + (0.28 * span_y),
        ],
        fill="black",
        width=max(2, scale),
    )


def _bubble_box(bubble: Mapping[str, Any], scale: int) -> tuple[float, float, float, float]:
    left = bubble["x"] * scale
    top = bubble["y"] * scale
    right = (bubble["x"] + bubble["width"]) * scale
    bottom = (bubble["y"] + bubble["height"]) * scale
    return left, top, right, bottom


def _degrade_scan(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]
    )
    destination = np.float32(
        [
            [90, 45],
            [width - 120, 35],
            [55, height - 95],
            [width - 25, height - 55],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    distorted = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    blurred = cv2.GaussianBlur(distorted, (7, 7), 0)
    noisy = blurred.astype(np.int16)
    noisy += np.random.default_rng(7).normal(0, 16, size=noisy.shape).astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _json_dump(report: Mapping[str, Any]) -> str:
    import json

    return json.dumps(report, indent=2)
