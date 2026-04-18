"""Persist one Quiz #5 short-answer scan-ingest session as durable artifacts on disk."""

from __future__ import annotations

import json
import os
from pathlib import PurePath
import tempfile
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from auto_grader.scan_readback import read_page_identity_qr_payload
from auto_grader.scan_registration import normalize_page_image


def persist_quiz5_short_answer_scan_session(
    *,
    scan_images: Mapping[str, np.ndarray],
    artifact: Mapping[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    pages = _page_index(artifact.get("pages"))
    serializable_results: list[dict[str, Any]] = []
    matched_page_codes: set[str] = set()

    os.makedirs(output_dir, exist_ok=True)
    normalized_dir = os.path.join(output_dir, "normalized_images")

    for scan_id, image in scan_images.items():
        safe_scan_id = _require_safe_scan_id(scan_id)
        checksum = _image_checksum(image)
        try:
            page_code = read_page_identity_qr_payload(image)
        except ValueError as exc:
            error_text = str(exc)
            lowered = error_text.lower()
            serializable_results.append(
                {
                    "scan_id": safe_scan_id,
                    "checksum": checksum,
                    "status": "ambiguous" if ("ambiguous" in lowered or "conflict" in lowered) else "unmatched",
                    "failure_reason": error_text,
                }
            )
            continue

        page = pages.get(page_code)
        if page is None:
            serializable_results.append(
                {
                    "scan_id": safe_scan_id,
                    "checksum": checksum,
                    "status": "unmatched",
                    "fallback_page_code": page_code,
                    "failure_reason": f"Unknown page code {page_code!r}",
                }
            )
            continue

        if page_code in matched_page_codes:
            serializable_results.append(
                {
                    "scan_id": safe_scan_id,
                    "checksum": checksum,
                    "status": "ambiguous",
                    "fallback_page_code": page_code,
                    "failure_reason": f"Multiple scans matched same page code {page_code!r}; refusing to guess a canonical page.",
                }
            )
            continue

        normalized = normalize_page_image(image, page)
        os.makedirs(normalized_dir, exist_ok=True)
        _write_image_atomic(os.path.join(normalized_dir, safe_scan_id), normalized)
        matched_page_codes.add(page_code)
        serializable_results.append(
            {
                "scan_id": safe_scan_id,
                "checksum": checksum,
                "status": "matched",
                "failure_reason": None,
                "page_number": page["page_number"],
                "fallback_page_code": page_code,
            }
        )

    summary = {
        "total_scans": len(serializable_results),
        "matched": sum(1 for entry in serializable_results if entry["status"] == "matched"),
        "unmatched": sum(1 for entry in serializable_results if entry["status"] == "unmatched"),
        "ambiguous": sum(1 for entry in serializable_results if entry["status"] == "ambiguous"),
    }

    manifest = {
        "opaque_instance_code": artifact["opaque_instance_code"],
        "expected_page_codes": [page["fallback_page_code"] for page in sorted(pages.values(), key=lambda p: p["page_number"])],
        "scan_results": serializable_results,
        "summary": summary,
    }
    manifest_path = os.path.join(output_dir, "session_manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
    }


def _page_index(raw_pages: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw_pages, list):
        raise TypeError("artifact.pages must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, Mapping):
            raise TypeError("artifact.pages entries must be mappings")
        indexed[str(page["fallback_page_code"])] = page
    return indexed


def _require_safe_scan_id(scan_id: Any) -> str:
    if not isinstance(scan_id, str) or scan_id == "":
        raise ValueError("scan_id must be a non-empty safe filename")
    if scan_id != PurePath(scan_id).name or scan_id in {".", ".."}:
        raise ValueError("scan_id must be a non-empty safe filename")
    return scan_id


def _image_checksum(image: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: str, data: Any) -> None:
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_image_atomic(path: str, image: np.ndarray) -> None:
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
