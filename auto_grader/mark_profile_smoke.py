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
            "expected_behavior_band": "grade",
            "scan_profile_ids": ["clean_scan", "office_scan", "stressed_scan"],
        },
        {
            "profile_id": "light_center",
            "description": "Lighter centered pencil fill",
            "mark_fn": lambda draw, bubble, scale: _draw_light_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "grade",
        },
        {
            "profile_id": "scribble_center",
            "description": "Scribbly centered student fill",
            "mark_fn": lambda draw, bubble, scale: _draw_scribble_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "grade",
        },
        {
            "profile_id": "off_center_patch",
            "description": "Off-center but still intentional fill inside the bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_off_center_patch(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "grade",
        },
        {
            "profile_id": "edge_smudge",
            "description": "Edge-only smudge with no center fill",
            "mark_fn": lambda draw, bubble, scale: _draw_edge_smudge(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "ignore",
            "scan_profile_ids": ["clean_scan", "office_scan", "stressed_scan"],
        },
        {
            "profile_id": "faint_center",
            "description": "Very faint centered fill near the current blank boundary",
            "mark_fn": lambda draw, bubble, scale: _draw_faint_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "review",
        },
        {
            "profile_id": "double_mark",
            "description": "Two clearly filled bubbles",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label, alternate_bubble_label],
            "expected_behavior_band": "review",
            "scan_profile_ids": ["clean_scan", "office_scan", "stressed_scan"],
        },
        {
            "profile_id": "hostile_correct_plus_glance",
            "description": "Correct fill plus a glancing stray on a neighboring wrong bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "grade",
            "stray_marks": [
                {
                    "bubble_label": alternate_bubble_label,
                    "draw_fn": _draw_glancing_stray,
                }
            ],
        },
        {
            "profile_id": "glancing_stray_only",
            "description": "A lone glancing stray that clips the bubble edge without a fill attempt",
            "mark_fn": lambda draw, bubble, scale: None,
            "bubble_labels": [],
            "expected_behavior_band": "ignore",
            "stray_marks": [
                {
                    "bubble_label": correct_bubble_label,
                    "draw_fn": _draw_glancing_stray,
                }
            ],
        },
        {
            "profile_id": "tiny_center_dot",
            "description": "A tiny incidental graphite dot near the center that is not a fill attempt",
            "mark_fn": lambda draw, bubble, scale: None,
            "bubble_labels": [],
            "expected_behavior_band": "ignore",
            "stray_marks": [
                {
                    "bubble_label": correct_bubble_label,
                    "draw_fn": _draw_tiny_center_dot,
                }
            ],
        },
        {
            "profile_id": "correct_plus_wrong_dot",
            "description": "A correct fill plus a tiny accidental dot on a neighboring wrong bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_dark_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "grade",
            "stray_marks": [
                {
                    "bubble_label": alternate_bubble_label,
                    "draw_fn": _draw_tiny_center_dot,
                }
            ],
        },
        {
            "profile_id": "ambiguous_patchy_center",
            "description": "Patchy center fill near the current ambiguous boundary",
            "mark_fn": lambda draw, bubble, scale: _draw_patchy_center(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "review",
        },
        {
            "profile_id": "illegible_scratchout",
            "description": "Scratchy unreadable fill that drags across the bubble",
            "mark_fn": lambda draw, bubble, scale: _draw_illegible_scratchout(draw, bubble, scale=scale),
            "bubble_labels": [correct_bubble_label],
            "expected_behavior_band": "review",
        },
    ]

    scan_profiles = [
        {
            "scan_profile_id": "clean_scan",
            "description": "Near-flat clean scan with mild paper and sensor noise",
            "severity_rank": 0,
        },
        {
            "scan_profile_id": "office_scan",
            "description": "Ordinary institutional scanner stress with mild skew, blur, and noise",
            "severity_rank": 1,
        },
        {
            "scan_profile_id": "stressed_scan",
            "description": "Harsher but still plausible copier stress with stronger blur, warp, and contrast loss",
            "severity_rank": 2,
        },
    ]

    report_profiles: list[dict[str, Any]] = []
    for scan_profile in scan_profiles:
        for profile in profiles:
            enabled_scan_profile_ids = profile.get("scan_profile_ids", ["office_scan"])
            if scan_profile["scan_profile_id"] not in enabled_scan_profile_ids:
                continue
            rendered = _render_profile_page(
                base_page,
                page,
                bubble_labels=profile["bubble_labels"],
                mark_fn=profile["mark_fn"],
                stray_marks=profile.get("stray_marks"),
            )
            degraded = _degrade_scan(
                rendered,
                scan_profile_id=scan_profile["scan_profile_id"],
                rng=np.random.default_rng(seed + (100 * scan_profile["severity_rank"]) + len(profile["profile_id"])),
            )
            stem = f"{profile['profile_id']}__{scan_profile['scan_profile_id']}"
            image_path = output_dir / f"{stem}.png"
            Image.fromarray(degraded).save(image_path)

            normalized_path = output_dir / f"{stem}_normalized.png"
            qr_payload, extracted, pipeline_error = _extract_with_failure_capture(
                degraded,
                page=page,
                answer_key=answer_key,
            )

            observed_marked_bubble_labels: list[str] = []
            observed_ambiguous_bubble_labels: list[str] = []
            observed_illegible_bubble_labels: list[str] = []
            observed_status: str | None = None
            review_required = False
            if extracted is not None:
                Image.fromarray(extracted["normalized_image"]).save(normalized_path)
                question_result = extracted["scored_questions"]["mc-1"]
                question_observation = extracted["bubble_observations"]["mc-1"]
                observed_marked_bubble_labels = extracted["marked_bubble_labels"]["mc-1"]
                observed_ambiguous_bubble_labels = question_observation["ambiguous_bubble_labels"]
                observed_illegible_bubble_labels = question_observation["illegible_bubble_labels"]
                observed_status = question_result["status"]
                review_required = question_result["review_required"]

            observed_behavior_band = _behavior_band_from_status(observed_status)
            expected_behavior_band = _require_behavior_band(
                profile["expected_behavior_band"],
                label=f"profile[{profile['profile_id']}].expected_behavior_band",
            )

            report_profiles.append(
                {
                    "profile_id": profile["profile_id"],
                    "description": profile["description"],
                    "scan_profile_id": scan_profile["scan_profile_id"],
                    "scan_description": scan_profile["description"],
                    "severity_rank": scan_profile["severity_rank"],
                    "image_path": str(image_path),
                    "normalized_image_path": str(normalized_path) if extracted is not None else None,
                    "decoded_payload": qr_payload,
                    "expected_behavior_band": expected_behavior_band,
                    "observed_behavior_band": observed_behavior_band,
                    "behavior_matches_expectation": observed_behavior_band == expected_behavior_band,
                    "pipeline_error": pipeline_error,
                    "observed_marked_bubble_labels": observed_marked_bubble_labels,
                    "observed_ambiguous_bubble_labels": observed_ambiguous_bubble_labels,
                    "observed_illegible_bubble_labels": observed_illegible_bubble_labels,
                    "observed_status": observed_status,
                    "review_required": review_required,
                }
            )

    scan_profile_summaries = _summarize_scan_profiles(report_profiles, scan_profiles)
    report = {
        "page_code": page["fallback_page_code"],
        "profiles": report_profiles,
        "scan_profile_summaries": scan_profile_summaries,
        "practical_boundary": _practical_boundary(scan_profile_summaries),
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


def _draw_tiny_center_dot(
    draw: ImageDraw.ImageDraw,
    bubble: Mapping[str, Any],
    *,
    scale: int,
) -> None:
    left, top, right, bottom = _bubble_box(bubble, scale)
    width = right - left
    height = bottom - top
    dot_radius = max(2.0, 0.055 * min(width, height))
    center_x = left + (0.50 * width)
    center_y = top + (0.50 * height)
    draw.ellipse(
        [
            center_x - dot_radius,
            center_y - dot_radius,
            center_x + dot_radius,
            center_y + dot_radius,
        ],
        fill=(120, 120, 120),
    )


def _bubble_box(bubble: Mapping[str, Any], scale: int) -> tuple[float, float, float, float]:
    left = bubble["x"] * scale
    top = bubble["y"] * scale
    right = (bubble["x"] + bubble["width"]) * scale
    bottom = (bubble["y"] + bubble["height"]) * scale
    return left, top, right, bottom


def _extract_with_failure_capture(
    degraded: np.ndarray,
    *,
    page: Mapping[str, Any],
    answer_key: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    qr_payload: str | None = None
    extracted: dict[str, Any] | None = None
    try:
        qr_payload = read_page_identity_qr_payload(degraded)
        extracted = extract_scored_mc_page(degraded, page, answer_key)
        return qr_payload, extracted, None
    except Exception as exc:
        return qr_payload, extracted, f"{type(exc).__name__}: {exc}"


def _degrade_scan(
    image: np.ndarray,
    *,
    scan_profile_id: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if scan_profile_id == "clean_scan":
        config = {
            "destination": np.float32(
                [
                    [28, 20],
                    [image.shape[1] - 34, 16],
                    [18, image.shape[0] - 28],
                    [image.shape[1] - 20, image.shape[0] - 24],
                ]
            ),
            "blur_kernel": (3, 3),
            "noise_sigma": 5.0,
            "contrast": 0.99,
            "brightness_shift": 1.0,
            "paper_sigma": 2.5,
            "streak_alpha": 0.025,
            "resize_scale": 0.70,
        }
    elif scan_profile_id == "office_scan":
        config = {
            "destination": np.float32(
                [
                    [90, 45],
                    [image.shape[1] - 120, 35],
                    [55, image.shape[0] - 95],
                    [image.shape[1] - 25, image.shape[0] - 55],
                ]
            ),
            "blur_kernel": (7, 7),
            "noise_sigma": 16.0,
            "contrast": 0.965,
            "brightness_shift": 4.0,
            "paper_sigma": 5.5,
            "streak_alpha": 0.055,
            "resize_scale": 0.58,
        }
    elif scan_profile_id == "stressed_scan":
        config = {
            "destination": np.float32(
                [
                    [106, 62],
                    [image.shape[1] - 138, 52],
                    [68, image.shape[0] - 124],
                    [image.shape[1] - 34, image.shape[0] - 72],
                ]
            ),
            "blur_kernel": (9, 9),
            "noise_sigma": 18.0,
            "contrast": 0.945,
            "brightness_shift": 7.0,
            "paper_sigma": 7.0,
            "streak_alpha": 0.075,
            "resize_scale": 0.52,
        }
    else:
        raise ValueError(f"Unknown scan_profile_id {scan_profile_id!r}")

    height, width = image.shape[:2]
    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, config["destination"])
    distorted = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    blurred = cv2.GaussianBlur(distorted, config["blur_kernel"], 0)
    working = blurred.astype(np.float32) * float(config["contrast"]) + float(config["brightness_shift"])
    working += _paper_texture(height, width, sigma=float(config["paper_sigma"]), rng=rng)
    working += _scanner_streaks(height, width, alpha=float(config["streak_alpha"]), rng=rng)
    working += rng.normal(0, float(config["noise_sigma"]), size=working.shape)
    clipped = np.clip(working, 0, 255).astype(np.uint8)
    resized = cv2.resize(
        clipped,
        dsize=None,
        fx=float(config["resize_scale"]),
        fy=float(config["resize_scale"]),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _paper_texture(
    height: int,
    width: int,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    texture = rng.normal(0, sigma, size=(height, width, 1))
    return np.repeat(texture, 3, axis=2).astype(np.float32)


def _scanner_streaks(
    height: int,
    width: int,
    *,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    horizontal = rng.normal(0, 255 * alpha, size=(height, 1, 1))
    vertical = rng.normal(0, 255 * (alpha * 0.45), size=(1, width, 1))
    field = horizontal + vertical
    return np.repeat(field, 3, axis=2).astype(np.float32)


def _behavior_band_from_status(status: str | None) -> str:
    if status in {"correct", "incorrect"}:
        return "grade"
    if status in {"multiple_marked", "ambiguous_mark", "illegible_mark"}:
        return "review"
    if status == "blank":
        return "ignore"
    return "failure"


def _summarize_scan_profiles(
    report_profiles: list[dict[str, Any]],
    scan_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scan_profile in scan_profiles:
        profile_rows = [
            row for row in report_profiles if row["scan_profile_id"] == scan_profile["scan_profile_id"]
        ]
        unexpected_cases = [
            {
                "profile_id": row["profile_id"],
                "observed_behavior_band": row["observed_behavior_band"],
                "expected_behavior_band": row["expected_behavior_band"],
                "pipeline_error": row["pipeline_error"],
            }
            for row in profile_rows
            if not row["behavior_matches_expectation"]
        ]
        summaries.append(
            {
                "scan_profile_id": scan_profile["scan_profile_id"],
                "description": scan_profile["description"],
                "severity_rank": scan_profile["severity_rank"],
                "unexpected_case_count": len(unexpected_cases),
                "unexpected_cases": unexpected_cases,
                "all_expected_behavior_held": len(unexpected_cases) == 0,
            }
        )
    return summaries


def _practical_boundary(scan_profile_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    strongest_all_expected_behavior_scan_profile_id: str | None = None
    next_scan_profile_with_unexpected_cases: str | None = None
    for summary in sorted(scan_profile_summaries, key=lambda item: item["severity_rank"]):
        if summary["all_expected_behavior_held"]:
            strongest_all_expected_behavior_scan_profile_id = summary["scan_profile_id"]
            continue
        next_scan_profile_with_unexpected_cases = summary["scan_profile_id"]
        break
    return {
        "strongest_all_expected_behavior_scan_profile_id": strongest_all_expected_behavior_scan_profile_id,
        "next_scan_profile_with_unexpected_cases": next_scan_profile_with_unexpected_cases,
    }


def _require_behavior_band(value: Any, *, label: str) -> str:
    if value not in {"grade", "review", "ignore"}:
        raise ValueError(f"{label} must be one of 'grade', 'review', or 'ignore'")
    return value


def _json_dump(report: Mapping[str, Any]) -> str:
    import json

    return json.dumps(report, indent=2)
