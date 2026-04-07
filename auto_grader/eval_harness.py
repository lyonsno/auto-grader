"""Eval harness for comparing model grading predictions against professor scores.

Loads ground truth from YAML, accepts model predictions, and produces
accuracy/calibration reports.  Model-agnostic — swapping models doesn't
change the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalItem:
    """One ground truth item from the professor-annotated eval dataset."""

    exam_id: str
    question_id: str
    answer_type: str
    page: int
    professor_score: float
    max_points: float
    professor_mark: str  # check | x | partial | unclear
    student_answer: str
    notes: str


@dataclass(frozen=True)
class Prediction:
    """Model output for one eval item."""

    exam_id: str
    question_id: str
    model_score: float
    model_confidence: float  # 0-1
    model_reasoning: str
    model_read: str  # what model thinks student wrote


@dataclass(frozen=True)
class CalibrationBin:
    """One bin in a confidence calibration histogram."""

    bin_start: float
    bin_end: float
    count: int
    avg_confidence: float
    accuracy: float  # fraction of exact score matches in this bin


@dataclass
class EvalReport:
    """Results of comparing model predictions against professor scores."""

    overall_exact_accuracy: float  # fraction of exact score matches
    overall_tolerance_accuracy: float  # fraction within ±1 point
    false_positives: int  # model gives credit, professor didn't
    false_negatives: int  # model docks, professor gave credit
    per_answer_type_exact: dict[str, float] = field(default_factory=dict)
    per_answer_type_tolerance: dict[str, float] = field(default_factory=dict)
    total_scored: int = 0
    unclear_excluded: int = 0
    total_points_possible: float = 0.0
    total_points_professor: float = 0.0
    calibration_bins: list[CalibrationBin] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_ground_truth(yaml_path: Path) -> list[EvalItem]:
    """Parse the ground truth YAML into a flat list of EvalItem."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    items: list[EvalItem] = []
    for exam in data["exams"]:
        exam_id = exam["exam_id"]
        for raw in exam["items"]:
            items.append(
                EvalItem(
                    exam_id=exam_id,
                    question_id=raw["question_id"],
                    answer_type=raw["answer_type"],
                    page=raw["page"],
                    professor_score=float(raw["professor_score"]),
                    max_points=float(raw["max_points"]),
                    professor_mark=raw["professor_mark"],
                    student_answer=raw["student_answer"],
                    notes=raw["notes"],
                )
            )
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_predictions(
    ground_truth: list[EvalItem],
    predictions: list[Prediction],
) -> EvalReport:
    """Compare model predictions against professor scores.

    Predictions are matched to ground truth by (exam_id, question_id).
    Items with professor_mark='unclear' are excluded from accuracy metrics.
    Raises ValueError if any scored ground truth item lacks a prediction.
    """
    pred_map: dict[tuple[str, str], Prediction] = {
        (p.exam_id, p.question_id): p for p in predictions
    }

    # Separate unclear items
    scored_items = [i for i in ground_truth if i.professor_mark != "unclear"]
    unclear_count = len(ground_truth) - len(scored_items)

    # Check all scored items have predictions
    missing = [
        (i.exam_id, i.question_id)
        for i in scored_items
        if (i.exam_id, i.question_id) not in pred_map
    ]
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} scored items: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Compute per-item results
    exact_matches = 0
    tolerance_matches = 0
    false_positives = 0
    false_negatives = 0
    total_points_possible = 0.0
    total_points_professor = 0.0

    # Per-answer-type tracking
    type_exact: dict[str, list[bool]] = {}
    type_tolerance: dict[str, list[bool]] = {}

    # Calibration tracking
    calibration_data: list[tuple[float, bool]] = []  # (confidence, exact_match)

    for item in scored_items:
        pred = pred_map[(item.exam_id, item.question_id)]

        exact = pred.model_score == item.professor_score
        within_tolerance = abs(pred.model_score - item.professor_score) <= 1.0

        if exact:
            exact_matches += 1
        if within_tolerance:
            tolerance_matches += 1

        # False positive: model gives more credit than professor
        if pred.model_score > item.professor_score:
            false_positives += 1
        # False negative: model gives less credit than professor
        if pred.model_score < item.professor_score:
            false_negatives += 1

        total_points_possible += item.max_points
        total_points_professor += item.professor_score

        # Per-type tracking
        atype = item.answer_type
        type_exact.setdefault(atype, []).append(exact)
        type_tolerance.setdefault(atype, []).append(within_tolerance)

        # Calibration
        calibration_data.append((pred.model_confidence, exact))

    n = len(scored_items)
    overall_exact = exact_matches / n if n > 0 else 0.0
    overall_tolerance = tolerance_matches / n if n > 0 else 0.0

    per_type_exact = {
        atype: sum(matches) / len(matches)
        for atype, matches in type_exact.items()
    }
    per_type_tolerance = {
        atype: sum(matches) / len(matches)
        for atype, matches in type_tolerance.items()
    }

    calibration_bins = _compute_calibration_bins(calibration_data)

    return EvalReport(
        overall_exact_accuracy=overall_exact,
        overall_tolerance_accuracy=overall_tolerance,
        false_positives=false_positives,
        false_negatives=false_negatives,
        per_answer_type_exact=per_type_exact,
        per_answer_type_tolerance=per_type_tolerance,
        total_scored=n,
        unclear_excluded=unclear_count,
        total_points_possible=total_points_possible,
        total_points_professor=total_points_professor,
        calibration_bins=calibration_bins,
    )


def _compute_calibration_bins(
    data: list[tuple[float, bool]],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Bin predictions by confidence and compute accuracy per bin."""
    if not data:
        return []

    bins: list[CalibrationBin] = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width

        in_bin = [
            (conf, match)
            for conf, match in data
            if bin_start <= conf < bin_end or (i == n_bins - 1 and conf == 1.0)
        ]

        if not in_bin:
            continue

        avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(1 for _, m in in_bin if m) / len(in_bin)

        bins.append(
            CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=len(in_bin),
                avg_confidence=avg_conf,
                accuracy=acc,
            )
        )

    return bins
