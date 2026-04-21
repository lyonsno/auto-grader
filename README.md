# Auto-Grader

Local-first exam generation and grading for chemistry courses. Generates
per-student exam variants from reusable templates, then grades scanned paper
submissions using OpenCV (multiple-choice) and VLM inference (short-answer).

Runs on a professor's laptop. No cloud dependency. Postgres-backed for
auditability and idempotent re-runs.

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

## What's here

### Two grading pipelines

**Multiple-choice (MC)** — fully implemented, end-to-end:
- Template-driven exam generation with per-student answer-key shuffling
- PDF rendering with QR identity markers, registration corners, and bubble grids
- OpenCV scan pipeline: QR readback → page registration → bubble interpretation → scoring
- Dominance arbitration for borderline marks (one strong fill beats faint traces)
- DB-backed scan sessions with idempotent re-ingestion and divergence detection
- Human review workflow for flagged questions (multiple marks, ambiguous fills)
- Professor-facing local web GUI and CLI for the full ingest → review → export cycle

**Short-answer (Quiz 5)** — in progress:
- Canonical reconstruction of legacy quiz variants from PDF
- Variant generation (A/B observed → C generated)
- DB-backed scan session persistence
- VLM-based grading pipeline (eval harness with ground truth scoring)

### Key modules

| Module | What it does |
|--------|-------------|
| `auto_grader.template_schema` | YAML template validation, variable types, expression evaluation |
| `auto_grader.generation` | Per-student exam instance generation with stable answer keys |
| `auto_grader.pdf_rendering` | Answer-sheet PDF rendering from generation artifacts |
| `auto_grader.scan_readback` | QR identity decoding from scanned pages |
| `auto_grader.scan_registration` | Skew correction via corner registration markers |
| `auto_grader.bubble_interpretation` | Filled-bubble detection from normalized page images |
| `auto_grader.mc_scoring` | MC scoring with status vocabulary: correct, incorrect, blank, multiple_marked, ambiguous_mark, illegible_mark |
| `auto_grader.mc_scan_ingest` | Batch scan ingestion producing matched/unmatched/ambiguous outcomes |
| `auto_grader.mc_scan_db` | DB persistence for scan sessions and scored outcomes |
| `auto_grader.mc_review_db` | DB persistence for human review resolutions |
| `auto_grader.mc_results_db` | Authoritative current-final truth surface per exam instance |
| `auto_grader.mc_workflow` | Professor-facing workflow: ingest, review, resolve, export |
| `auto_grader.mc_workflow_gui` | Local web GUI over the MC workflow |
| `auto_grader.vlm_inference` | VLM inference wrapper for short-answer grading |
| `auto_grader.eval_harness` | Evaluation harness for VLM grading accuracy |
| `auto_grader.quiz5_short_answer_reconstruction` | Legacy quiz family reconstruction from PDF |

### Scripts

**Professor-facing:**

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

**Development / demo:**

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
- Failed scans remain auditable artifacts with clear status and traceable
  provenance.

## Eval harness and Project Paint Dry (dev)

The eval harness (`scripts/smoke_vlm.py`) runs short-answer items through a
VLM grading pipeline against ground truth. It talks to two local
OpenAI-compatible servers:

- **Grader model** (Qwen3.5 / Gemma-4 on a second machine): the model being
  evaluated
- **Narrator** (Bonsai 8B 1-bit, local): a small model that watches the
  grader's reasoning token stream and produces structured running commentary

**Project Paint Dry** is the live narrator terminal UI. When the eval harness
runs with `--narrate`, it opens a second terminal window that renders the
grading process in real time:

- **Scoreboard** with tall-digit dials: total elapsed, turn elapsed, on-target
  fraction, left-on-table, and bad calls against the professor's ground truth
- **Live pane** streaming the narrator's character-by-character dispatch as the
  grader reasons through each item
- **Focus preview** showing the actual student handwriting region being graded,
  rendered inline via Kitty graphics protocol
- **History pane** with structured narrator output per item: basis annotations
  (what the grader credited), evidence lines, lean summaries, and verdict
  comparisons (grader score vs. professor score with elapsed time)
- **Rejected pane** for dedup-filtered and empty-filtered narrator output

The history pane is scrollable (`k`/`j` row, `u`/`d` page, `0` live edge).
Each run persists to `runs/<ts>-<model>/narrator.jsonl` (machine-replayable)
and `narrator.txt` (human-readable transcript). Focus regions are defined in
`eval/focus_regions.yaml` and adjustable via `scripts/annotate_focus_regions.py`.

The narrator requires the PRISM MLX fork for 1-bit quantization support.
See [`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md) for server
setup and [`docs/project_paint_dry.md`](docs/project_paint_dry.md) for the
surface semantics, scorebug vocabulary, and checkpoint/history contract.

## Project structure

```
auto_grader/          # Core Python package
scripts/              # CLI entrypoints and dev tools
tests/                # Contract test suites (61 files)
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
