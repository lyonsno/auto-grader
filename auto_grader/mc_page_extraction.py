"""Package matched-page registration, bubble readback, evidence, and MC scoring into one result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from auto_grader.bubble_interpretation import read_bubble_evidence, read_bubble_observations
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
    bubble_observations = read_bubble_observations(normalized_image, page)
    bubble_evidence = read_bubble_evidence(normalized_image, page)
    page_answer_key = {
        question_id: _require_mapping(
            answer_key.get(question_id),
            f"answer_key.{question_id}",
        )
        for question_id in bubble_observations
    }
    marked_bubble_labels = {
        question_id: observation["marked_bubble_labels"]
        for question_id, observation in bubble_observations.items()
    }
    scored_questions = score_marked_mc_bubbles(
        bubble_observations,
        page_answer_key,
        bubble_evidence,
    )

    return {
        "page_number": _require_int(page.get("page_number"), "page.page_number"),
        "fallback_page_code": _require_string(
            page.get("fallback_page_code"),
            "page.fallback_page_code",
        ),
        "normalized_image": normalized_image,
        "bubble_observations": bubble_observations,
        "bubble_evidence": bubble_evidence,
        "marked_bubble_labels": marked_bubble_labels,
        "scored_questions": scored_questions,
    }


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{label} must be a non-empty string")
    return value
