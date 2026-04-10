"""Package matched-page registration, bubble readback, and MC scoring into one result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from auto_grader.bubble_interpretation import read_marked_bubble_labels
from auto_grader.mc_scoring import score_marked_mc_bubbles
from auto_grader.scan_registration import normalize_page_image


def extract_scored_mc_page(
    image: np.ndarray,
    page: Mapping[str, Any],
    answer_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(page, Mapping):
        raise TypeError("page must be a mapping")
    if not isinstance(answer_key, Mapping):
        raise TypeError("answer_key must be a mapping")

    normalized_image = normalize_page_image(image, page)
    marked_bubble_labels = read_marked_bubble_labels(normalized_image, page)
    scored_questions = score_marked_mc_bubbles(marked_bubble_labels, answer_key)

    return {
        "page_number": _require_int(page.get("page_number"), "page.page_number"),
        "fallback_page_code": _require_string(
            page.get("fallback_page_code"),
            "page.fallback_page_code",
        ),
        "normalized_image": normalized_image,
        "marked_bubble_labels": marked_bubble_labels,
        "scored_questions": scored_questions,
    }


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
