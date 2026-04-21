"""Prepare per-response-box Quiz #5 crops for short-answer VLM trialing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def prepare_quiz5_short_answer_trial_crops(
    *,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    normalized_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    normalized_path = Path(normalized_dir)
    output_path = Path(output_dir)
    crops_dir = output_path / "response_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    page_index = _page_index(artifact)
    responses: list[dict[str, Any]] = []
    matched_pages = 0

    for scan_result in manifest["scan_results"]:
        if scan_result.get("status") != "matched":
            continue

        matched_pages += 1
        page_code = str(scan_result["fallback_page_code"])
        page = page_index[page_code]
        image_path = normalized_path / str(scan_result["scan_id"])
        if not image_path.exists():
            raise ValueError(f"Missing normalized image for matched page: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable normalized image for matched page: {image_path}")

        prompt_index = {
            str(block["question_id"]): str(block["text"])
            for block in page["prompt_blocks"]
        }
        response_regions = _response_regions(page)
        for region in response_regions:
            crop_box = _image_crop_box(
                region=region["crop_region"],
                page=page,
                image=image,
            )
            crop = _crop_image(image, crop_box)
            crop_path = crops_dir / f"{page_code}--{region['question_id']}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Failed to write crop image to {crop_path}")
            responses.append(
                {
                    "question_id": region["question_id"],
                    "label": region["label"],
                    "page_number": page["page_number"],
                    "fallback_page_code": page_code,
                    "scan_id": scan_result["scan_id"],
                    "crop_path": str(crop_path),
                    "crop_box": crop_box,
                    "prompt_text": prompt_index.get(region["question_id"], ""),
                    "crop_kind": region["crop_kind"],
                }
            )

    response_manifest = {
        "opaque_instance_code": artifact["opaque_instance_code"],
        "variant_id": artifact["variant_id"],
        "responses": responses,
        "summary": {
            "matched_pages": matched_pages,
            "response_box_crops": len(responses),
        },
    }
    manifest_path = output_path / "trial_prep_manifest.json"
    manifest_path.write_text(json.dumps(response_manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "responses_dir": str(crops_dir),
    }


def _page_index(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pages = artifact.get("pages")
    if not isinstance(pages, list):
        raise TypeError("artifact.pages must be a list")
    return {str(page["fallback_page_code"]): dict(page) for page in pages}


def _response_regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    response_boxes = page.get("response_boxes")
    if not isinstance(response_boxes, list):
        raise TypeError("page.response_boxes must be a list")

    regions: list[dict[str, Any]] = []
    index = 0
    while index < len(response_boxes):
        current = dict(response_boxes[index])
        workspace = current.get("workspace")
        if isinstance(workspace, dict):
            group: list[dict[str, Any]] = [current]
            group_end = index + 1
            while group_end < len(response_boxes) and not isinstance(response_boxes[group_end].get("workspace"), dict):
                group.append(dict(response_boxes[group_end]))
                group_end += 1
            regions.extend(_workspace_group_regions(group, workspace))
            index = group_end
            continue

        regions.append(
            {
                "question_id": str(current["question_id"]),
                "label": str(current["label"]),
                "crop_region": _box_region(current),
                "crop_kind": "label_box",
            }
        )
        index += 1

    return regions


def _workspace_group_regions(
    group: list[dict[str, Any]],
    workspace: dict[str, Any],
) -> list[dict[str, Any]]:
    workspace_key = _shared_workspace_key(str(group[0]["question_id"]))
    if any(_shared_workspace_key(str(entry["question_id"])) != workspace_key for entry in group[1:]):
        raise ValueError(
            "Malformed response_boxes: shared-workspace followers must stay within the same question family"
        )
    if len(group) == 1:
        only = group[0]
        return [
            {
                "question_id": str(only["question_id"]),
                "label": str(only["label"]),
                "crop_region": dict(workspace),
                "crop_kind": "workspace",
            }
        ]

    ws_x = int(workspace["x"])
    ws_y = int(workspace["y"])
    ws_width = int(workspace["width"])
    ws_height = int(workspace["height"])
    band_height = ws_height // len(group)

    regions: list[dict[str, Any]] = []
    for idx, response_box in enumerate(group):
        band_y = ws_y + (idx * band_height)
        next_y = ws_y + ws_height if idx == len(group) - 1 else ws_y + ((idx + 1) * band_height)
        regions.append(
            {
                "question_id": str(response_box["question_id"]),
                "label": str(response_box["label"]),
                "crop_region": {
                    "x": ws_x,
                    "y": band_y,
                    "width": ws_width,
                    "height": next_y - band_y,
                },
                "crop_kind": "workspace_band" if idx == 0 else "derived_shared_workspace_band",
            }
        )
    return regions


def _box_region(response_box: dict[str, Any]) -> dict[str, int]:
    return {
        "x": int(response_box["x"]),
        "y": int(response_box["y"]),
        "width": int(response_box["width"]),
        "height": int(response_box["height"]),
    }


def _shared_workspace_key(question_id: str) -> str:
    return question_id.split("-", 1)[0]


def _image_crop_box(
    *,
    region: dict[str, int],
    page: dict[str, Any],
    image: np.ndarray,
) -> dict[str, int]:
    page_width = float(page["width"])
    page_height = float(page["height"])
    image_height, image_width = image.shape[:2]
    scale_x = image_width / page_width
    scale_y = image_height / page_height

    x = max(0, min(int(round(region["x"] * scale_x)), image_width - 1))
    y = max(0, min(int(round(region["y"] * scale_y)), image_height - 1))
    width = max(1, int(round(region["width"] * scale_x)))
    height = max(1, int(round(region["height"] * scale_y)))

    if x + width > image_width:
        width = image_width - x
    if y + height > image_height:
        height = image_height - y

    return {"x": x, "y": y, "width": width, "height": height}


def _crop_image(image: np.ndarray, crop_box: dict[str, int]) -> np.ndarray:
    x = crop_box["x"]
    y = crop_box["y"]
    width = crop_box["width"]
    height = crop_box["height"]
    return image[y : y + height, x : x + width].copy()
