"""Eval harness for comparing model grading predictions against professor scores.

Loads ground truth from YAML, accepts model predictions, and produces
accuracy/calibration reports.  Model-agnostic — swapping models doesn't
change the harness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FocusRegion:
    """Optional page-local focus box for display/cropping affordances.

    Coordinates are normalized to the rendered page: x/y are the top-left
    corner, width/height are box dimensions, all in the closed interval
    [0, 1] with positive width/height.
    """

    page: int
    x: float
    y: float
    width: float
    height: float
    source: str = "ground_truth"


@dataclass(frozen=True)
class EvalItem:
    """One ground truth item from the professor-annotated eval dataset.

    `professor_score` is the historical record — what the prof actually
    wrote on the page. `corrected_score` is the post-investigation
    truth, populated only when human review has determined the prof
    made a grading error (in either direction). When `corrected_score`
    is not None, the eval harness uses it instead of `professor_score`
    when computing accuracy. The professor_score field is preserved
    for traceability — we never silently rewrite history.
    """

    exam_id: str
    question_id: str
    answer_type: str
    page: int
    professor_score: float
    max_points: float
    professor_mark: str  # check | x | partial | unclear
    student_answer: str
    notes: str
    focus_region: FocusRegion | None = None
    corrected_score: float | None = None
    correction_reason: str = ""

    @property
    def truth_score(self) -> float:
        """The score we believe is correct after any human investigation.

        Falls back to professor_score when no correction has been
        recorded. Use this everywhere accuracy is being measured.
        """
        return (
            self.corrected_score
            if self.corrected_score is not None
            else self.professor_score
        )


@dataclass(frozen=True)
class Prediction:
    """Model output for one eval item.

    raw_assistant and raw_reasoning preserve the unparsed grader output so
    a downstream critic pass can read the verbatim <think> trace where
    consistency-rule violations actually live (the curated model_reasoning
    field is too short to capture them). Both default to empty string for
    backward compatibility with predictions constructed in tests.
    """

    exam_id: str
    question_id: str
    model_score: float
    model_confidence: float  # 0-1
    model_reasoning: str
    model_read: str  # what model thinks student wrote
    raw_assistant: str = ""  # full assistant content string before JSON parse
    raw_reasoning: str = ""  # full reasoning_content stream (verbatim <think>)
    # Upstream-dependency forcing fields. The grader is required to fill
    # these in BEFORE committing to a score; their presence in the schema
    # is the structural lever for the consistency-rule failure mode that
    # prompt-only nudges did not move on Qwen3p5-35B-A3B.
    upstream_dependency: str = "none"  # e.g. "5(a)" or "none"
    if_dependent_then_consistent: bool | None = None  # null when no dependency


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
            corrected_raw = raw.get("corrected_score")
            corrected = (
                float(corrected_raw) if corrected_raw is not None else None
            )
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
                    focus_region=_parse_focus_region(
                        raw.get("focus_region"),
                        page=int(raw["page"]),
                        default_source="ground_truth",
                    ),
                    corrected_score=corrected,
                    correction_reason=raw.get("correction_reason", ""),
                )
            )
    return items


def resolve_focus_region(
    item: EvalItem,
    template: Mapping[str, Any] | None = None,
) -> FocusRegion | None:
    """Return the best available focus region for an eval item.

    Explicit item-level metadata wins. If absent, the resolver will look up a
    question- or part-level ``focus_region`` block in the provided template and
    attach it to the item's page.
    """
    if item.focus_region is not None:
        return item.focus_region
    if template is None:
        return None

    raw_focus = _find_template_focus_region(
        template.get("sections", []),
        question_id=item.question_id,
    )
    return _parse_focus_region(
        raw_focus,
        page=item.page,
        default_source="template",
    )


def _parse_focus_region(
    raw_focus: Any,
    *,
    page: int,
    default_source: str,
) -> FocusRegion | None:
    if raw_focus is None:
        return None
    if not isinstance(raw_focus, dict):
        raise ValueError("focus_region must be a mapping when provided")

    raw_page = raw_focus.get("page", page)
    x = float(raw_focus["x"])
    y = float(raw_focus["y"])
    width = float(raw_focus["width"])
    height = float(raw_focus["height"])
    source = str(raw_focus.get("source", default_source))
    return FocusRegion(
        page=int(raw_page),
        x=x,
        y=y,
        width=width,
        height=height,
        source=source,
    )


def _find_template_focus_region(
    nodes: list[dict[str, Any]],
    *,
    question_id: str,
) -> dict[str, Any] | None:
    for node in nodes:
        if node.get("id") == question_id and "focus_region" in node:
            raw_focus = node["focus_region"]
            if isinstance(raw_focus, dict):
                return raw_focus
            return None
        parts = node.get("parts")
        if isinstance(parts, list):
            nested = _find_template_focus_region(parts, question_id=question_id)
            if nested is not None:
                return nested
        questions = node.get("questions")
        if isinstance(questions, list):
            nested = _find_template_focus_region(questions, question_id=question_id)
            if nested is not None:
                return nested
    return None


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

        # Use truth_score (corrected if present, professor_score otherwise)
        # so that human-investigated grading errors don't penalize the
        # grader. The professor_score field is preserved as historical
        # record but not used for accuracy.
        truth = item.truth_score
        exact = pred.model_score == truth
        within_tolerance = abs(pred.model_score - truth) <= 1.0

        if exact:
            exact_matches += 1
        if within_tolerance:
            tolerance_matches += 1

        # False positive: model gives more credit than truth
        if pred.model_score > truth:
            false_positives += 1
        # False negative: model gives less credit than truth
        if pred.model_score < truth:
            false_negatives += 1

        total_points_possible += item.max_points
        total_points_professor += truth

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
