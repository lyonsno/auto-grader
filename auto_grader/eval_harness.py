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

    score_basis is the terse literal basis for the awarded score: what
    earned credit and what lost it. model_reasoning is the broader
    interpretive reasoning around ambiguity, OCR, rescue logic, or human
    review. raw_assistant and raw_reasoning preserve the unparsed grader
    output so a downstream critic pass can read the verbatim <think>
    trace where consistency-rule violations actually live (the curated
    fields are intentionally shorter). Optional fields default to empty
    string / None for backward compatibility with predictions
    constructed in tests.

    **Truncation sentinel contract** (Operation Zilch Reaper, forward
    lane): when the grader did not commit to a score — either because
    the VLM ran out of its token budget before finishing the JSON, or
    because the emitted output was otherwise unparseable — the resulting
    Prediction must carry ``model_score=None``, ``model_confidence=None``,
    and ``truncated=True``. Complete rows carry numeric score /
    confidence and ``truncated=False``. Downstream consumers that want
    to tell "model said 0" apart from "model said nothing" check
    ``truncated`` (or equivalently ``model_score is None``) rather than
    string-matching on ``model_reasoning``. See
    ``attractors/auto-grader_zilch-reaper-forward_stop-recording-truncated-
    grader-output-as-model-score-zero_2026-04-11.md``.
    """

    exam_id: str
    question_id: str
    # None on truncated / unparseable rows — see truncation sentinel
    # contract in the class docstring.
    model_score: float | None
    model_confidence: float | None  # 0-1, None on truncated rows
    model_reasoning: str
    model_read: str  # what model thinks student wrote
    score_basis: str = ""
    raw_assistant: str = ""  # full assistant content string before JSON parse
    raw_reasoning: str = ""  # full reasoning_content stream (verbatim <think>)
    # Upstream-dependency forcing fields. The grader is required to fill
    # these in BEFORE committing to a score; their presence in the schema
    # is the structural lever for the consistency-rule failure mode that
    # prompt-only nudges did not move on Qwen3p5-35B-A3B.
    upstream_dependency: str = "none"  # e.g. "5(a)" or "none"
    if_dependent_then_consistent: bool | None = None  # null when no dependency
    # Truncation sentinel flag. True when the grader did not commit to
    # a score (length-truncated output, unparseable content, etc.).
    # Complete rows carry False.
    truncated: bool = False


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
    # Truncation sentinel: rows where the grader did not commit to a
    # score (VLM ran out of token budget, output was unparseable, etc.).
    # Mirrors unclear_excluded — items the scorer deliberately kept out
    # of the accuracy denominator. Surfaced so operators can see
    # completion rate as a first-class number.
    truncated_excluded: int = 0
    total_points_possible: float = 0.0
    # total_points_truth is the sum of `EvalItem.truth_score` across all
    # scored items — this is the corrected-truth baseline the grader is
    # being measured against, NOT the historical prof mark total. When
    # no corrections are recorded, this equals the prof-mark total;
    # when corrections are present, it reflects them. The field was
    # formerly named `total_points_professor`, which became misleading
    # after eval_harness started using `truth_score` for scoring.
    total_points_truth: float = 0.0
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
                    corrected_score=corrected,
                    correction_reason=raw.get("correction_reason", ""),
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

    # Truncation sentinel: drop rows where the grader did not commit
    # to a score. These are non-predictions and must not contaminate
    # the accuracy numerator/denominator or calibration — see Operation
    # Zilch Reaper (forward lane) attractor for the full framing.
    truncated_count = sum(
        1
        for item in scored_items
        if pred_map[(item.exam_id, item.question_id)].truncated
    )
    scored_items = [
        item
        for item in scored_items
        if not pred_map[(item.exam_id, item.question_id)].truncated
    ]

    # Compute per-item results
    exact_matches = 0
    tolerance_matches = 0
    false_positives = 0
    false_negatives = 0
    total_points_possible = 0.0
    total_points_truth = 0.0

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
        total_points_truth += truth

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
        truncated_excluded=truncated_count,
        total_points_possible=total_points_possible,
        total_points_truth=total_points_truth,
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
