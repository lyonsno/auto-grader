"""Persist one MC scan-ingest session as a durable artifact set on disk.

Wraps the in-memory ``ingest_mc_scans`` surface and writes:
- ``session_manifest.json``: machine-readable manifest with checksums,
  per-scan outcomes, expected page codes, and summary counts.
- ``normalized_images/<scan_id>.png``: normalized page images for matched scans.

Re-running with the same inputs into the same directory is idempotent at the
artifact-identity level: the manifest is overwritten atomically and normalized
images are replaced only if content changed.
"""

from __future__ import annotations

import json
import os
from pathlib import PurePath
import tempfile
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from auto_grader.mc_scan_ingest import ingest_mc_scans


def persist_scan_session(
    *,
    scan_images: Mapping[str, np.ndarray],
    artifact: Mapping[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Ingest scans and persist the session as a durable artifact set.

    Returns a dict with ``output_dir`` and ``manifest_path`` keys.
    """
    ingest_result = ingest_mc_scans(scan_images, artifact)

    os.makedirs(output_dir, exist_ok=True)

    normalized_dir = os.path.join(output_dir, "normalized_images")

    serializable_results: list[dict[str, Any]] = []

    for scan_result in ingest_result["scan_results"]:
        safe_scan_id = _require_safe_scan_id(scan_result["scan_id"])
        entry: dict[str, Any] = {
            "scan_id": safe_scan_id,
            "checksum": scan_result["checksum"],
            "status": scan_result["status"],
            "failure_reason": scan_result["failure_reason"],
        }

        if scan_result["status"] == "matched":
            entry["page_number"] = scan_result["page_number"]
            entry["fallback_page_code"] = scan_result["fallback_page_code"]
            entry["scored_questions"] = _serialize_scored_questions(
                scan_result["scored_questions"]
            )

            # Write normalized image.
            os.makedirs(normalized_dir, exist_ok=True)
            image_path = os.path.join(normalized_dir, safe_scan_id)
            _write_image_atomic(image_path, scan_result["normalized_image"])

        serializable_results.append(entry)

    summary = {
        "total_scans": len(ingest_result["scan_results"]),
        "matched": len(ingest_result["matched_pages"]),
        "unmatched": len(ingest_result["unmatched_scans"]),
        "ambiguous": len(ingest_result["ambiguous_scans"]),
        "review_required": len(ingest_result["review_required_pages"]),
    }

    manifest = {
        "opaque_instance_code": ingest_result["opaque_instance_code"],
        "expected_page_codes": ingest_result["expected_page_codes"],
        "scan_results": serializable_results,
        "summary": summary,
    }

    manifest_path = os.path.join(output_dir, "session_manifest.json")
    _write_json_atomic(manifest_path, manifest)

    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
    }


def _serialize_scored_questions(
    scored_questions: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the JSON-serializable subset of scored_questions."""
    serialized: dict[str, Any] = {}
    for qid, question in scored_questions.items():
        serialized[qid] = {
            k: v
            for k, v in question.items()
            if not isinstance(v, (np.ndarray, bytes, memoryview))
        }
    return serialized


def _require_safe_scan_id(scan_id: Any) -> str:
    if not isinstance(scan_id, str) or scan_id == "":
        raise ValueError("scan_id must be a non-empty safe filename")
    if scan_id != PurePath(scan_id).name or scan_id in {".", ".."}:
        raise ValueError("scan_id must be a non-empty safe filename")
    return scan_id


def _write_json_atomic(path: str, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_image_atomic(path: str, image: np.ndarray) -> None:
    """Write a numpy image as PNG atomically via tmp + rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".png")
    os.close(fd)
    try:
        if not cv2.imwrite(tmp_path, image):
            raise OSError(f"Failed to write normalized image to temporary path: {tmp_path}")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
