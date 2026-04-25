# Project Paint Dry

Operator-facing semantics for the live narrator surface.

This doc is intentionally about what the surface means and how to read it. It is
not a full aesthetic history of the UI, and it is not a Bonsai bring-up guide.
For narrator server launch and troubleshooting, see
[Bonsai narrator server setup](bonsai_server_setup.md).

## What it is

Project Paint Dry is the live sidecar narrator for smoke and eval runs. It is
the operator surface for watching the grader think item by item while preserving
enough structure to spot drift, repetition, and scoring mistakes in real time.

It is not the source of truth for final grades. The actual grader outputs still
land in `predictions.jsonl`, and final scoring remains the eval pipeline's job.

## Major surfaces

### Scorebug

The scorebug is the top broadcast-style strip. It is meant to be readable in the
operator's peripheral vision.

Top row:
- `CURRENT MODEL`: the active grader model for this run
- `SET`: the current item subset, for example `TRICKY` or `TRICKY+`
- `ITEM`: current item index within the run

Lower tally row:
- `TOTAL`: total run elapsed time
- `TURN`: current item elapsed time
- `EXACT`: points where the grader exactly matched the corrected truth target
- `FLOOR MET`: lawful minimum credit the grader met under the acceptable band
- `PARTIAL`: lawful partial credit awarded above zero and below full credit
- `BELOW FLOOR`: points below the lawful minimum that should have been awarded

These are running tallies across the current run, not predictions about the next
item.

### Project Header

The `PROJECT PAINT DRY` header is the show identity line. It also carries:
- elapsed run time (`total=...`)
- current-item time (`turn=...`)
- narrator counters (`emitted=...`, `dedup=...`, `empty=...`)

Those counters are operational diagnostics, not grading metrics.

### Status + Live

This is the active cognition band.

- `status` is the durable lane-level summary of what the narrator thinks it is
  currently checking.
- `live` is the current first-person play-by-play line.

Status and live are intentionally distinct:
- status should read like the current investigative lane
- live should read like the current thought inside that lane

When the narrator has not yet produced new content for an item:
- status falls back to `AWAITING STATUS`
- live shows a rotating placeholder instead of stale previous-item text

### Focus Preview

When available, the focus preview shows the image crop for the region currently
being discussed. This is a provisional operator aid, not yet a fully settled
artifact contract. It exists to help evaluate whether the narrator and the
visible region are aligned.

### History

The history stack is the running durable trace for the current run.

It currently contains, in descending importance:
- item headers
- per-item scoring/topic lines
- acceptable score-band lines when the item has a lawful range
- structured dossier rows (`Basis`, `Read`, `What survives`, `Deciding issue`)
- rolling `Context` checkpoint lines
- targeted dossier-progress placeholders while background sidecars run

The history view uses stronger value and shimmer on the first visual row of an
entry and dims wrapped continuations faster. This is deliberate: the top row of
each item block should anchor the eye, while deeper reasoning should recede.

## Checkpoints

Checkpoints and dossiers are the durable synthesis layers.

They are not:
- a live thought
- a status line
- a score line

`Context` checkpoints are compact rolling summaries of the durable issue that
repeated live/status lines have established. New `Context` rows replace the
previous context for that item instead of piling up redundant siblings.

Dossiers are trailing per-item artifacts for long, unstable, interpretive, or
score-disagreement cases. They are targeted to the originating problem even when
the next item has already started. While they are in flight, history shows a
`Dossier:` placeholder; when the rows land, the placeholder is replaced by the
structured dossier rows.

Current transition rule:
- live/status lines are primarily active instrumentation
- `Context` checkpoints and dossier rows are the intended durable persisted
  history surface
- full-credit exact hits should normally avoid trailing dossiers unless another
  explicit trigger makes the case interesting

Checkpoint and dossier styling is intentionally calmer and more anchored than
live/body rows:
- warm structural mark
- cooler anchored text
- readable fade down a block so deeper rows recede without disappearing

## Persistence and artifacts

Typical narrator-related run artifacts include:
- `narrator.jsonl`: structured event log for the live narrator surface
- `narrator.txt`: plain-text companion log when emitted
- `predictions.jsonl`: grader outputs and reasoning, independent of narrator UI

The narrator UI can fail or be absent without losing grader predictions. A broken
Paint Dry surface is a quality-of-life problem, not a grading-data problem.

## Current contracts worth preserving

These are the durable semantics another maintainer should assume unless changed
deliberately alongside tests:

- The scorebug is always present from run start, even before any scored topics
  arrive.
- `status` and `live` are separate surfaces with separate fallback/idle behavior.
- `Context` checkpoints and dossier rows are the intended durable history layer.
- Dossier progress belongs under the originating problem, not in the current
  live status lane.
- History hierarchy is value-led: top rows matter more than wrapped continuation
  rows.
- The scorebug tallies are cumulative run state, not item-local state.

## What does not belong in this doc

The following should stay in Epistaxis or code review history unless they harden
into contract:
- screenshot-by-screenshot palette tuning
- failed shimmer experiments
- micro-arguments about exact glyph shapes unless they become a stable invariant
- provisional layout experiments such as a width-gated sidecar preview mode
