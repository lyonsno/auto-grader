"""Package one batch of scanned MC pages into explicit matched/unmatched outcomes."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

import numpy as np

from auto_grader.mc_page_extraction import extract_scored_mc_page
from auto_grader.scan_readback import read_page_identity_qr_payload


def ingest_mc_scans(
    scan_images: Mapping[str, np.ndarray],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(scan_images, Mapping):
        raise TypeError("scan_images must be a mapping")
    if not isinstance(artifact, Mapping):
        raise TypeError("artifact must be a mapping")

    pages = _page_index(_require_list(artifact.get("pages"), "artifact.pages"))
    answer_key = _require_mapping(artifact.get("answer_key"), "artifact.answer_key")
    opaque_instance_code = _require_string(
        artifact.get("opaque_instance_code"),
        "artifact.opaque_instance_code",
    )

    provisional_results: list[dict[str, Any]] = []
    matched_groups: dict[str, list[int]] = {}

    for scan_id, image in scan_images.items():
        normalized_scan_id = _require_string(scan_id, "scan_id")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{normalized_scan_id}.image must be a numpy.ndarray")

        checksum = _image_checksum(image)
        try:
            payload = read_page_identity_qr_payload(image)
        except ValueError as exc:
            error_text = str(exc)
            lowered_error_text = error_text.lower()
            status = (
                "ambiguous"
                if ("ambiguous" in lowered_error_text or "conflict" in lowered_error_text)
                else "unmatched"
            )
            provisional_results.append(
                {
                    "scan_id": normalized_scan_id,
                    "checksum": checksum,
                    "status": status,
                    "failure_reason": error_text,
                }
            )
            continue

        page = pages.get(payload)
        if page is None:
            provisional_results.append(
                {
                    "scan_id": normalized_scan_id,
                    "checksum": checksum,
                    "status": "unmatched",
                    "fallback_page_code": payload,
                    "failure_reason": f"Unknown page code {payload!r}",
                }
            )
            continue

        provisional_results.append(
            {
                "scan_id": normalized_scan_id,
                "checksum": checksum,
                "status": "_matched_candidate",
                "failure_reason": None,
                "fallback_page_code": payload,
                "_page": page,
                "_image": image,
            }
        )
        matched_groups.setdefault(payload, []).append(len(provisional_results) - 1)

    for page_code, indexes in matched_groups.items():
        if len(indexes) < 2:
            continue
        for index in indexes:
            result = provisional_results[index]
            result.pop("_page", None)
            result.pop("_image", None)
            result["status"] = "ambiguous"
            result["failure_reason"] = (
                f"Multiple scans matched same page code {page_code!r}; refusing to guess a canonical page."
            )

    scan_results: list[dict[str, Any]] = []
    matched_pages: list[dict[str, Any]] = []
    unmatched_scans: list[dict[str, Any]] = []
    ambiguous_scans: list[dict[str, Any]] = []
    review_required_pages: list[dict[str, Any]] = []

    for result in provisional_results:
        status = result["status"]
        if status == "_matched_candidate":
            page = result.pop("_page")
            image = result.pop("_image")
            extracted = extract_scored_mc_page(image, page, answer_key)
            _raise_on_reserved_key_collisions(extracted)
            matched_result = {
                "scan_id": result["scan_id"],
                "checksum": result["checksum"],
                "status": "matched",
                "failure_reason": None,
                **extracted,
            }
            scan_results.append(matched_result)
            matched_pages.append(matched_result)
            if any(
                question_result["review_required"]
                for question_result in matched_result["scored_questions"].values()
            ):
                review_required_pages.append(matched_result)
            continue

        finalized = dict(result)
        scan_results.append(finalized)
        if status == "unmatched":
            unmatched_scans.append(finalized)
        elif status == "ambiguous":
            ambiguous_scans.append(finalized)
        else:
            raise ValueError(f"Unsupported ingest status {status!r}")

    ordered_page_codes = [
        _require_string(page.get("fallback_page_code"), "page.fallback_page_code")
        for page in sorted(
            pages.values(),
            key=lambda page: _require_int(page.get("page_number"), "page.page_number"),
        )
    ]

    return {
        "opaque_instance_code": opaque_instance_code,
        "expected_page_codes": ordered_page_codes,
        "scan_results": scan_results,
        "matched_pages": matched_pages,
        "unmatched_scans": unmatched_scans,
        "ambiguous_scans": ambiguous_scans,
        "review_required_pages": review_required_pages,
    }


def _page_index(pages: list[Any]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for page in pages:
        page_mapping = _require_mapping(page, "artifact.pages[]")
        fallback_page_code = _require_string(
            page_mapping.get("fallback_page_code"),
            "page.fallback_page_code",
        )
        indexed[fallback_page_code] = page_mapping
    return indexed


def _image_checksum(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _raise_on_reserved_key_collisions(extracted: Mapping[str, Any]) -> None:
    reserved_keys = {"scan_id", "checksum", "status", "failure_reason"}
    colliding_keys = sorted(reserved_keys.intersection(extracted))
    if colliding_keys:
        raise ValueError(
            "extract_scored_mc_page returned reserved ingest keys: "
            + ", ".join(colliding_keys)
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
