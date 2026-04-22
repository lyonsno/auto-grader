# Auto-Grader

Local-first exam generation and grading for chemistry courses. It handles the
full paper loop: generate per-student variants from reusable templates, ingest
scanned submissions, score multiple-choice deterministically with OpenCV, and
run model-based grading and evaluation on handwritten short-answer work.

The project is designed to run on a professor's own machine. There is no cloud
dependency in the normal workflow. Postgres is the persistence spine so grading
state, scan sessions, review decisions, and re-runs stay auditable instead of
turning into loose files and shell history.

## Requirements

- Python 3.12+
- PostgreSQL (local, native — no Docker required)
- Recent Apple Silicon Mac (the only tested platform)
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

Set `DATABASE_URL` to point at your local Postgres instance:

```bash
export DATABASE_URL="postgresql://postgres@localhost/auto_grader"
```

## Main surfaces

### Local web GUI

The normal front door for production multiple-choice grading. An instructor or
TA can:

- create or select a grading target
- ingest a batch of scans
- review flagged MC cases
- persist manual resolutions
- export final results

The GUI is intentionally thin over the same database-backed workflow the CLI
and contracts use.

### Project Paint Dry

The live observability surface for short-answer grading. Paint Dry is the tool
you use when the question is not just "what score came out?" but "what is the
model seeing, what is it defending, and where is the real uncertainty?"

It is not just a narrator and not just a dashboard. It is a weird but powerful
development instrument for watching model judgment form in real time on messy
handwritten work. In practice it is where prompt tuning, ambiguity diagnosis,
focus-region annotation, and model-comparison work become tractable instead of
staying buried in static logs.

- **Scoreboard** with tall-digit dials: total elapsed, turn elapsed, on-target
  fraction, left-on-table, and bad calls against the professor's ground truth
- **Live pane** streaming what the narrator thinks is happening while the
  grader reasons through each item
- **Focus preview** showing the exact handwriting crop the grader is looking at
- **History pane** that keeps structured per-item artifacts such as basis,
  evidence, lean, core issue, and the newer `Read / Salvage / Hinge` dossier
  rows for interesting items
- **Rejected pane** for dedup-filtered and empty-filtered narrator output

The history pane is scrollable (`k`/`j` row, `u`/`d` page, `0` live edge,
`a` reopens focus-region annotation for the current item). Each run persists
to `runs/<ts>-<model>/narrator.jsonl` (machine-replayable) and `narrator.txt`
(human-readable transcript).

## What the repo does

### Multiple-choice workflow

This side of the repo is the production workflow today:

- template-driven exam generation with per-student answer-key shuffling
- PDF rendering with QR identity markers, registration corners, and bubble grids
- OpenCV scan ingestion: QR readback → page registration → bubble interpretation → scoring
- dominance arbitration for borderline marks
- DB-backed scan sessions with idempotent re-ingestion and divergence detection
- human review workflow for flagged questions
- local web GUI and CLI for ingest → review → resolve → export

### Short-answer grading and evaluation

This side of the repo is where the more experimental and more interesting model
work lives:

- VLM-based grading pipeline with ground-truth scoring and run comparison
- Project Paint Dry for live inspection of model behavior during smoke and eval
- focus-region authoring and in-window annotation for the exact evidence crop
- split-model and single-model narrator/grader configurations
- truth correction and backfill tooling so historical professor marks and
  corrected ground truth can coexist honestly
- canonical reconstruction and variant generation work for fixed-layout
  short-answer quiz families such as Quiz 5

The short-answer lane is partly a grading system and partly an observability
program. The point is not only to get a score, but to learn where the score is
coming from, where the model is charitable or strict, and which disagreements
are model failures versus genuinely hard handwritten cases.

## Running it

### Professor-facing

```bash
# Launch the local web GUI (MC workflow)
python scripts/launch_mc_workflow_gui.py --open-browser \
  --database-url postgresql:///postgres \
  --artifact-json /path/to/artifact.json \
  --scan-dir /path/to/scans \
  --output-dir /tmp/mc-gui-output

# CLI workflow: ingest → review → resolve → export
python scripts/mc_workflow.py ingest \
  --exam-instance-id 123 \
  --artifact-json /path/to/artifact.json \
  --scan-dir /path/to/scans \
  --output-dir /tmp/mc-ingest

python scripts/mc_workflow.py review --exam-instance-id 123 --output-dir /tmp/mc-review
python scripts/mc_workflow.py resolve \
  --exam-instance-id 123 \
  --resolutions-json /tmp/resolutions.json \
  --output-dir /tmp/mc-resolve
python scripts/mc_workflow.py export \
  --exam-instance-id 123 --format csv --output-dir /tmp/mc-export
```

### Development and evaluation

```bash
# Render a generated MC exam from a real template
python scripts/render_generated_mc_exam_demo.py

# Run the OpenCV demo pipeline against a scan directory
python scripts/run_mc_opencv_demo.py

# Reconstruct Quiz 5 family from legacy PDFs
python scripts/reconstruct_quiz5_short_answer.py

# Run VLM smoke test
python scripts/smoke_vlm.py --model qwen3.5 --items 5
```

For Paint Dry and short-answer smoke work, see the eval-harness notes below
plus [`docs/project_paint_dry.md`](docs/project_paint_dry.md).

## Eval harness and Paint Dry

The eval harness (`scripts/smoke_vlm.py`) runs short-answer items through a
VLM grading pipeline against ground truth. It talks to local
OpenAI-compatible servers in two supported configurations:

- **Split mode**: dedicated narrator surface plus separate grader backend
- **Single-model mode**: one strong Qwen backend handles both grading and
  narration in the same smoke run

Both are real supported working modes. Single-model narration is no longer an
experiment; on strong backends it produces materially better live readouts than
the old tiny-sidecar assumption.

Focus regions live in `eval/focus_regions.yaml` and can be adjusted from the
command line or reopened directly from Paint Dry with `a`. The combination of
focus preview, structured live history, and durable run artifacts is what makes
short-answer smoke useful here: you can see the evidence crop, the evolving
judgment, and the post-hoc run record all in one workflow.

For deeper semantics and operator vocabulary, see:

- [`docs/project_paint_dry.md`](docs/project_paint_dry.md)
- [`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md)

## Implementation surfaces

These are the core modules maintainers usually touch first:

| Module | What it owns |
|--------|--------------|
| `auto_grader.template_schema` | YAML template validation, variables, expressions |
| `auto_grader.generation` | Per-student exam instance generation |
| `auto_grader.pdf_rendering` | PDF rendering for generated paper artifacts |
| `auto_grader.scan_readback` | QR identity decoding from scanned pages |
| `auto_grader.scan_registration` | Page registration and skew correction |
| `auto_grader.bubble_interpretation` | Filled-bubble detection from normalized scans |
| `auto_grader.mc_workflow` | Professor-facing MC ingest/review/resolve/export workflow |
| `auto_grader.mc_workflow_gui` | Local web GUI over the MC workflow |
| `auto_grader.vlm_inference` | Short-answer VLM inference and scan-to-page mapping |
| `auto_grader.eval_harness` | Ground truth loading, scoring, and report logic |
| `auto_grader.thinking_narrator` | Paint Dry narrator dispatch and structured artifacts |
| `auto_grader.narrator_sink` | Durable narrator event/log emission |

## Testing

Contract tests enforce schema, DB persistence, generation, rendering, scan
processing, and scoring invariants. 61 test files covering both pipelines.

```bash
# Run all non-DB contract tests
python -m auto_grader.contract_test_runner

# Run with Postgres-backed tests included
TEST_DATABASE_URL=... python -m auto_grader.contract_test_runner --require-postgres

# One-command disposable Postgres (bootstraps a temp cluster, runs tests, tears down)
./scripts/run_postgres_contracts.sh
```

Tests follow fail-first discipline: the failing test is written before the
implementation. Contract tests assert concrete invariants (shapes, specific
errors, deterministic values), not "it runs" signals.

## Database

Postgres-first from day one. No SQLite fallback. The database is the spine of
truth — it maps paper artifacts back to canonical exam instances, enforces
uniqueness invariants, supports idempotent re-ingestion, and preserves audit
history.

**Connection:** `psycopg` v3 via `create_connection(database_url=None)`.
Explicit URL wins over `DATABASE_URL` env var. Accepts `postgres://` and
`postgresql://` schemes.

**Schema (v0 tables):**

| Table | Purpose |
|-------|---------|
| `students` | Student identity |
| `template_versions` | Immutable versioned templates |
| `exam_definitions` | Exam composition and config |
| `exam_instances` | Per-student instances with `opaque_instance_code` |
| `exam_pages` | Page-level artifacts |
| `scan_artifacts` | Scan files, checksums, decoded IDs, status |
| `grade_records` | Computed and finalized grades |
| `audit_events` | Lightweight event trail (entity, type, JSONB payload, TIMESTAMPTZ) |
| `mc_scan_sessions` | MC scan session metadata |
| `mc_scan_pages` | Per-page match status within a session |
| `mc_question_outcomes` | Per-question scored outcomes |
| `mc_review_resolutions` | Human override decisions with original-status provenance |

Versioned tables (`template_versions`, `exam_definitions`) are immutable once
written, enforced by DB triggers.

## Design invariants

- A student never ends up with two finalized grades for the same exam attempt.
- An exam instance is uniquely traceable from paper artifacts via QR codes and
  fallback codes.
- Template and exam versions used for generation are immutable and
  reconstructable.
- Ingestion never silently guesses identity. Unmatched or ambiguous scans are
  quarantined with explicit failure reasons.
- Duplicate ingestion does not duplicate or corrupt results (idempotent via
  checksum key).
- Ambiguous marks are flagged for review, never confidently scored.
- Failed scans remain auditable artifacts with clear status and traceable provenance.

## Project structure

```
auto_grader/          # Core grading, workflow, narrator, and eval code
scripts/              # CLI entrypoints and dev tools
tests/                # Contract suites for grading, DB, narrator, and eval surfaces
templates/            # Exam templates (YAML)
eval/                 # Ground truth and focus regions for VLM eval
docs/                 # Decision records, setup guides, backlogs
```

## Further reading

- [`docs/postgres_contract_decision_worksheet.md`](docs/postgres_contract_decision_worksheet.md) — rationale behind DB contract decisions
- [`docs/postgres_migration_checklist.md`](docs/postgres_migration_checklist.md) — migration sequencing
- [`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md) — narrator server setup
- [`docs/project_paint_dry.md`](docs/project_paint_dry.md) — Project Paint Dry surface semantics and contracts
- [`docs/professor-backlog.md`](docs/professor-backlog.md) — items pending professor input
