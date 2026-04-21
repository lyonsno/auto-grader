# Auto-Grader

Local-first exam generation and grading for chemistry courses. It generates
per-student exam variants from reusable templates, then grades scanned paper
submissions with deterministic OpenCV for multiple-choice work and VLM
inference for short-answer work.

The system runs on a professor's laptop, keeps the database local, and is built
around auditable reruns instead of cloud custody or hidden state.

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

This is the normal operator entry point. It is where an instructor or TA can:

- create or select a grading target
- ingest a batch of scans
- review flagged MC cases
- persist manual resolutions
- export final results

The GUI is intentionally thin over the same database-backed workflow the CLI
and contracts use. It is the front door, not a second application brain.

### Project Paint Dry

Paint Dry is the live grading view for short-answer smoke runs. It is the
developer and operator surface for watching a run happen in real time,
including what the model is looking at, what it is defending, and why it is
hesitating on a hard case.

Paint Dry combines:

- live narration of the grading run
- the current handwritten focus preview
- score accounting against ground truth
- retained scrollback for older narrator history
- in-window focus-region annotation for the active item

It is the surface you use when the interesting question is not just "what score
came out?" but "why did the model make that call?"

## What's here

### Two grading pipelines

**Multiple-choice (MC)** — fully implemented, end-to-end:
- Template-driven exam generation with per-student answer-key shuffling
- PDF rendering with QR identity markers, registration corners, and bubble grids
- OpenCV scan pipeline: QR readback → page registration → bubble interpretation → scoring
- Dominance arbitration for borderline marks (one strong fill beats faint traces)
- DB-backed scan sessions with idempotent re-ingestion and divergence detection
- Human review workflow for flagged questions (multiple marks, ambiguous fills)
- Local web GUI and CLI for the full ingest → review → export cycle

**Short-answer (Quiz 5)** — in progress:
- Canonical reconstruction of legacy quiz variants from PDF
- Variant generation (A/B observed → C generated)
- DB-backed scan session persistence
- VLM-based grading pipeline (eval harness with ground truth scoring)
- Project Paint Dry live grading view with focus preview, retained history,
  ambiguity diagnosis, and in-window focus-region annotation
- Split-model and single-model narrator/grader configurations for smoke and
  prompt work
- Run comparison and truth/backfill tooling for evaluating model behavior over time

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

## Eval harness and narrator (dev)

The eval harness (`scripts/smoke_vlm.py`) runs short-answer items through a
VLM grading pipeline against ground truth. It talks to two local
OpenAI-compatible servers:

- **Grader model** (Qwen3.5 / Gemma-4 on a second machine): the model being
  evaluated
- **Narrator** (Bonsai 8B 1-bit, local): produces live play-by-play commentary
  of the grading process in a second terminal window ("Project Paint Dry")

The standard split configuration is:

- grader on `http://macbook-pro-2.local:8001`
- narrator on `http://nlm2pr.local:8002`

The harness also supports a single-model configuration, where the same strong
model handles both grading and narration by pointing `--narrator-url` and
`--narrator-model` at the same backend as the grader.

In practice that means there are two real working modes:

- split mode, with a dedicated narrator surface and a separate grader backend
- single-model mode, where a strong Qwen backend grades and narrates in the
  same smoke run

Both are supported development surfaces. The second one is no longer just a
side experiment; it is a real path for live smoke and prompt work.

The narrator requires the PRISM MLX fork for 1-bit quantization support.
See [`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md) for setup.

## Project Paint Dry (dev)

Paint Dry is the live grading view used during short-answer smoke runs. It is
not the professor-facing front door; it is the operator and developer surface
for watching a run happen while it is happening.

The reader shows:

- live narration of the grading run
- the active focus preview
- score accounting
- retained history with scrollback
- a rejected-summary/debug surface

It is useful for:

- smoke visibility
- ambiguity diagnosis
- prompt tuning
- operator correction during a live run
- understanding why a hard case is slow or disputed

The history pane is scrollable throughout the run:

- `k` / `j` move one row
- `u` / `d` page up/down
- `0` returns to the live edge
- `a` reopens focus-region annotation for the current item

While scrolled up, new history continues to accumulate without yanking the
viewport back to newest, and rows that fall out of the default live-edge view
remain recoverable through scrollback.

The focus preview uses canonical focus-region data from
`eval/focus_regions.yaml`. Those regions can be reviewed and adjusted with
`scripts/annotate_focus_regions.py`, and the reader can reopen that annotator
directly for the active item via `a`, then refresh the preview in place.

Each run persists narrator output to `runs/<ts>-<model>/narrator.jsonl`
(machine-replayable) and `runs/<ts>-<model>/narrator.txt` (human-readable
transcript), alongside the ordinary grading artifacts.

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
- [`docs/professor-backlog.md`](docs/professor-backlog.md) — items pending professor input
