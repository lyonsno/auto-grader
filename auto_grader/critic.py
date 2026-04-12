"""Post-hoc critic pass over a grader run.

Reads predictions.jsonl from a run directory and produces critic.jsonl
with override deltas for items where the grader contradicted its own
schema commitments.

v1 — DETERMINISTIC ENFORCEMENT
==============================

The grader's JSON output schema (per the upstream-dependency-schema
slice, commit 1169739) requires it to declare two fields BEFORE it
commits to a score:

  - upstream_dependency: the earlier part this question depends on, or "none"
  - if_dependent_then_consistent: true / false / null when no dependency

The consistency rule says: if a question depends on an earlier part AND
the student's work in this part is internally consistent with their
(possibly wrong) earlier answer, the student gets FULL CREDIT on this
part. The X on the earlier part already captured the upstream error.

This module enforces the rule deterministically. If the grader's own
schema fields say "yes, depends on 5(a)" and "yes, the work is
consistent" but the score is less than max_points, the grader has
contradicted its own declaration. We override the score to max_points
and record the rule invocation.

No LLM calls in v1. The grader has already done the judgment work in
the schema fields; v1 just enforces that the score matches.

v2 — LLM AUDIT (NOT YET IMPLEMENTED)
====================================

v1 is blind to cases where the grader's SCHEMA FIELDS THEMSELVES are
wrong — e.g., the grader says "no upstream dependency" on fr-5b when
there clearly is one. For those, you want a second opinion that
re-reads the question and decides whether the grader correctly
identified the dependency in the first place.

`critique_run_llm` is the placeholder for that. It will be wired up
when we have data from the schema-forced smoke run telling us how
often the schema fields are themselves wrong. Pluggable so the same
interface can host Qwen self-critique or step-3.5-flash escalation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

CriticAction = Literal["unchanged", "override"]
CriticRule = Literal["consistency", "answered-form", "none"]


@dataclass(frozen=True)
class CriticDelta:
    """One critic verdict on one prediction.

    `action == "unchanged"` means the critic agrees with the grader and
    `new_score` equals the original. `action == "override"` means the
    critic is replacing the grader's score and `new_score` is the
    corrected value. `rule_invoked` says which rule fired (or "none" for
    unchanged items).
    """

    exam_id: str
    question_id: str
    action: CriticAction
    original_score: float
    new_score: float  # equals original_score when action == "unchanged"
    rule_invoked: CriticRule
    reason: str
    critic_confidence: float  # 1.0 for deterministic v1; reserved for v2


# ---------------------------------------------------------------------------
# v1 — deterministic consistency-rule enforcement
# ---------------------------------------------------------------------------


def apply_consistency_rule(record: dict) -> CriticDelta:
    """Apply the deterministic consistency rule to one prediction record.

    Returns a CriticDelta describing the verdict. Pure function — no I/O,
    no model calls. Deterministic given the input record.

    The rule fires when ALL of the following are true:

      1. The grader declared an upstream dependency (`upstream_dependency`
         is something other than "none" / empty).
      2. The grader declared the student's work is consistent with that
         upstream answer (`if_dependent_then_consistent` is True).
      3. The grader's score is less than max_points.

    When all three are true, the grader contradicted its own schema
    commitment by withholding credit on internally-consistent work. We
    override the score to max_points.

    In every other case the critic returns `action="unchanged"`.
    """
    exam_id = record.get("exam_id", "")
    question_id = record.get("question_id", "")

    # Truncation sentinel: the grader did not commit to a score on this
    # row. There is nothing to critique and no numeric original_score to
    # carry. Operation Zilch Reaper (forward lane) — a non-prediction
    # is not a judgment, so the critic abstains cleanly.
    # CriticDelta.original_score / new_score are float (not float | None)
    # because score_delta sums in critique_run / format_report would crash
    # on None. 0.0 with action="unchanged" contributes zero to the delta.
    if record.get("truncated") or record.get("model_score") is None:
        return CriticDelta(
            exam_id=exam_id,
            question_id=question_id,
            action="unchanged",
            original_score=0.0,
            new_score=0.0,
            rule_invoked="none",
            reason="Grader did not commit to a score (truncated); critic abstains.",
            critic_confidence=1.0,
        )

    original_score = float(record.get("model_score", 0.0))
    max_points = float(record.get("max_points", 0.0))
    upstream_dep = (record.get("upstream_dependency") or "none").strip()
    if_consistent = record.get("if_dependent_then_consistent")

    has_dependency = upstream_dep.lower() not in ("", "none")
    is_consistent = if_consistent is True  # strict — None and False both miss
    score_below_max = original_score < max_points

    if has_dependency and is_consistent and score_below_max:
        return CriticDelta(
            exam_id=exam_id,
            question_id=question_id,
            action="override",
            original_score=original_score,
            new_score=max_points,
            rule_invoked="consistency",
            reason=(
                f"Grader declared upstream_dependency={upstream_dep!r} and "
                f"if_dependent_then_consistent=True, but withheld credit "
                f"({original_score} of {max_points}). The consistency rule "
                f"requires full credit when the student's work is internally "
                f"consistent with their own (wrong) upstream answer — the "
                f"upstream error is already captured by the X on that part. "
                f"Overriding to {max_points}."
            ),
            critic_confidence=1.0,
        )

    return CriticDelta(
        exam_id=exam_id,
        question_id=question_id,
        action="unchanged",
        original_score=original_score,
        new_score=original_score,
        rule_invoked="none",
        reason="No consistency-rule violation detected.",
        critic_confidence=1.0,
    )


def critique_run(predictions_path: Path) -> list[CriticDelta]:
    """Read predictions.jsonl and apply v1 deterministic critic to each
    prediction record. Skips header / footer lines.

    Returns the full list of deltas in input order — both `unchanged`
    and `override` — so callers can compute aggregate stats and write a
    complete critic.jsonl.
    """
    deltas: list[CriticDelta] = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "prediction":
                continue
            deltas.append(apply_consistency_rule(record))
    return deltas


def write_critic_jsonl(deltas: list[CriticDelta], path: Path) -> None:
    """Write deltas to a JSONL file. Header line first, then one record
    per delta, then a footer with aggregate stats.
    """
    overrides = [d for d in deltas if d.action == "override"]
    unchanged = [d for d in deltas if d.action == "unchanged"]
    score_delta = sum(d.new_score - d.original_score for d in deltas)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "header",
                    "critic_version": "v1-deterministic",
                    "rule_invoked": "consistency",
                    "predictions_count": len(deltas),
                }
            )
            + "\n"
        )
        for d in deltas:
            f.write(json.dumps({"type": "delta", **asdict(d)}) + "\n")
        f.write(
            json.dumps(
                {
                    "type": "footer",
                    "overrides": len(overrides),
                    "unchanged": len(unchanged),
                    "total_score_delta": score_delta,
                }
            )
            + "\n"
        )


def summarize(deltas: list[CriticDelta]) -> str:
    """Human-readable summary of a critic run for stdout."""
    overrides = [d for d in deltas if d.action == "override"]
    score_delta = sum(d.new_score - d.original_score for d in deltas)
    lines = [
        f"Critic v1 (deterministic consistency-rule enforcement)",
        f"  predictions seen:  {len(deltas)}",
        f"  overrides:         {len(overrides)}",
        f"  unchanged:         {len(deltas) - len(overrides)}",
        f"  total score delta: +{score_delta:g} points",
    ]
    if overrides:
        lines.append("")
        lines.append("Overrides:")
        for d in overrides:
            lines.append(
                f"  {d.exam_id}/{d.question_id}: "
                f"{d.original_score} -> {d.new_score} "
                f"({d.rule_invoked})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v2 — LLM audit (placeholder)
# ---------------------------------------------------------------------------


def critique_run_llm(
    predictions_path: Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "1234",
    timeout: float = 120.0,
) -> list[CriticDelta]:
    """v2 placeholder — LLM-based audit of cases v1 cannot catch.

    The deterministic v1 only catches grader self-contradictions: the
    schema says "consistent" but the score says "no credit". It cannot
    catch cases where the grader's schema fields themselves are wrong
    (e.g., grader says "no upstream dependency" on fr-5b when there
    clearly is one).

    v2 is for those cases. It will:

      1. For each prediction where v1 returned `unchanged`, send the
         (question prompt, student answer, grader's reasoning span,
         grader's schema fields, grader's score) to an LLM critic.
      2. Ask: did the grader correctly identify whether this question
         depends on an earlier part? Did it correctly judge whether
         the work is consistent with that earlier answer?
      3. If the LLM disagrees with the grader's schema fields AND the
         disagreement would change the score, override.
      4. Return the merged list of v1 + v2 deltas.

    The model parameter is pluggable so the same interface can host
    Qwen self-critique (cheap, same endpoint) or step-3.5-flash
    escalation (expensive but stronger reasoning).

    Not yet implemented. Wire when the schema-forced smoke run gives
    us data on how often the schema fields are themselves wrong.
    """
    raise NotImplementedError(
        "v2 LLM audit not yet wired. Use v1 critique_run() for now. "
        "v2 is blocked on data from the schema-forced smoke run — we "
        "need to know how often the grader's schema declarations are "
        "themselves wrong before we can scope the LLM audit usefully."
    )


# ---------------------------------------------------------------------------
# Helpers used by tests and the run_critic CLI
# ---------------------------------------------------------------------------


def iter_predictions(path: Path) -> Iterator[dict]:
    """Yield prediction records from a predictions.jsonl, skipping header
    and footer lines. Useful for callers that want to inspect raw
    records alongside critic deltas.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "prediction":
                yield record
