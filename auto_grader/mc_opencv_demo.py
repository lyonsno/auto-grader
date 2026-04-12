"""Run the landed MC/OpenCV pipeline over a directory of scans and persist a demo bundle."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np

from auto_grader.mc_scan_ingest import ingest_mc_scans


_SCAN_SUFFIXES = {".png", ".jpg", ".jpeg"}


def run_mc_opencv_demo(
    *,
    artifact_json_path: str | Path,
    scan_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    artifact_path = _require_existing_file(artifact_json_path, "artifact_json_path")
    scan_directory = _require_existing_dir(scan_dir, "scan_dir")
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scan_images = _load_scan_images(scan_directory)
    ingest_result = ingest_mc_scans(scan_images, artifact)

    matched_pages_dir = output_directory / "matched_pages"
    matched_pages_dir.mkdir(parents=True, exist_ok=True)
    persisted_ingest_result = _persist_demo_images(ingest_result, matched_pages_dir)
    summary = _summarize_ingest_result(persisted_ingest_result)

    summary_path = output_directory / "summary.json"
    ingest_result_path = output_directory / "ingest_result.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ingest_result_path.write_text(
        json.dumps(persisted_ingest_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "artifact_json_path": str(artifact_path),
        "scan_dir": str(scan_directory),
        "output_dir": str(output_directory),
        "summary_json_path": str(summary_path),
        "ingest_result_json_path": str(ingest_result_path),
        "summary": summary,
    }


def _persist_demo_images(
    ingest_result: Mapping[str, Any],
    matched_pages_dir: Path,
) -> dict[str, Any]:
    normalized_image_paths_by_scan_id: dict[str, str] = {}

    for matched_page in _require_list(ingest_result.get("matched_pages"), "ingest_result.matched_pages"):
        matched_mapping = _require_mapping(matched_page, "ingest_result.matched_pages[]")
        scan_id = _require_string(matched_mapping.get("scan_id"), "matched_page.scan_id")
        normalized_image = matched_mapping.get("normalized_image")
        if not isinstance(normalized_image, np.ndarray):
            raise TypeError(f"{scan_id}.normalized_image must be a numpy.ndarray")

        normalized_path = matched_pages_dir / f"{Path(scan_id).stem}__normalized.png"
        if not cv2.imwrite(str(normalized_path), normalized_image):
            raise RuntimeError(f"Failed to write normalized image for {scan_id!r}")
        normalized_image_paths_by_scan_id[scan_id] = str(normalized_path)

    return _sanitize_for_json(ingest_result, normalized_image_paths_by_scan_id)


def _sanitize_for_json(
    value: Any,
    normalized_image_paths_by_scan_id: Mapping[str, str],
    *,
    current_scan_id: str | None = None,
) -> Any:
    if isinstance(value, np.ndarray):
        raise TypeError("Raw numpy arrays must be replaced with persisted image paths before JSON export")
    if isinstance(value, Mapping):
        next_scan_id = current_scan_id
        if "scan_id" in value:
            next_scan_id = _require_string(value.get("scan_id"), "scan_result.scan_id")

        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "normalized_image":
                if next_scan_id is None:
                    raise ValueError("normalized_image cannot be persisted without an associated scan_id")
                sanitized["normalized_image_path"] = _require_string(
                    normalized_image_paths_by_scan_id.get(next_scan_id),
                    f"{next_scan_id}.normalized_image_path",
                )
                continue
            sanitized[_require_string(key, "mapping.key")] = _sanitize_for_json(
                nested_value,
                normalized_image_paths_by_scan_id,
                current_scan_id=next_scan_id,
            )
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_for_json(
                item,
                normalized_image_paths_by_scan_id,
                current_scan_id=current_scan_id,
            )
            for item in value
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported demo JSON value {type(value).__name__}")


def _summarize_ingest_result(ingest_result: Mapping[str, Any]) -> dict[str, Any]:
    scan_results = _require_list(ingest_result.get("scan_results"), "ingest_result.scan_results")
    review_required_pages = _require_list(
        ingest_result.get("review_required_pages"),
        "ingest_result.review_required_pages",
    )

    status_counts = Counter()
    scored_question_status_counts = Counter()
    for scan_result in scan_results:
        scan_mapping = _require_mapping(scan_result, "ingest_result.scan_results[]")
        status = _require_string(scan_mapping.get("status"), "scan_result.status")
        status_counts[status] += 1

        if status != "matched":
            continue
        for question_result in _require_mapping(
            scan_mapping.get("scored_questions"),
            "matched_page.scored_questions",
        ).values():
            question_mapping = _require_mapping(question_result, "scored_question")
            scored_question_status_counts[
                _require_string(question_mapping.get("status"), "scored_question.status")
            ] += 1

    return {
        "matched": status_counts["matched"],
        "unmatched": status_counts["unmatched"],
        "ambiguous": status_counts["ambiguous"],
        "review_required_pages": len(review_required_pages),
        "scored_question_status_counts": dict(sorted(scored_question_status_counts.items())),
    }


def _load_scan_images(scan_dir: Path) -> dict[str, np.ndarray]:
    scans: dict[str, np.ndarray] = {}
    for path in sorted(scan_dir.iterdir()):
        if path.suffix.lower() not in _SCAN_SUFFIXES or not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load scan image {path}")
        scans[path.name] = image

    if not scans:
        raise ValueError(f"No scan images found in {scan_dir}")
    return scans


def _require_existing_dir(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must point to an existing directory")
    return path


def _require_existing_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} must point to an existing file")
    return path


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
