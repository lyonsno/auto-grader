# Auto-Grader

Local-first exam generation and grading for chemistry courses.

This project is built for a very specific workflow: a professor authors an
assessment, prints individualized paper artifacts, ingests batch scans, reviews
the few things the machine is not entitled to guess, and exports results
without handing custody to a cloud service.

The system is optimized for operational sanity rather than theatrical AI:

- each student can receive a distinct variant
- paper artifacts remain traceable and auditable
- scan ingestion is idempotent and explicit about ambiguity
- grading prefers visible uncertainty over silent guessing
- the local machine remains the source of truth

## What the user actually touches

There are two real surfaces here.

### 1. The local web GUI

This is the normal front door.

The GUI lets an instructor or operator:

- author assessments
- choose or create a grading target
- ingest a batch of scans
- inspect the review queue
- persist manual resolutions
- export final results

It is intentionally thin over the durable workflow and database model. The GUI
is not its own application brain; it is the practical operator surface over the
same underlying truth the CLI and contracts use.

### 2. Project Paint Dry

Paint Dry is the live grading view used during short-answer smoke runs and
model work. It is not the professor-facing front door. It is the operator and
developer surface for watching a run happen while it is happening.

Paint Dry is useful when the interesting question is not just "what score came
out?" but:

- what evidence is the model looking at?
- why is it hesitating?
- is this a real ambiguity or just a bad crop?
- is the prompt pushing the model toward the right kind of caution?

The Paint Dry reader combines:

- live narration of the grading run
- a focus preview of the current handwritten evidence
- score accounting
- retained history with scrollback
- in-window focus-region annotation for the active item

That turns hard short-answer cases into something inspectable instead of
forcing the operator to reconstruct everything from logs afterward.

## What the system does

At a high level, the product has six jobs.

### Author assessments

Assessments are defined from structured templates instead of one-off documents.
Questions can vary numerically per student while remaining reproducible and
reviewable.

The durable authored format remains structured and strict. The GUI sits over
that model so the user does not have to begin from blank YAML just to create an
assessment or pick something to grade.

### Generate individualized paper artifacts

From a roster and an assessment definition, the system generates per-student
instances with stable identity, answer-key mapping, and printable paper
artifacts.

Reproducibility is a hard requirement. The system must be able to answer, later,
"what exactly did this student receive?"

### Put stable identity on paper

Printed pages carry enough information to survive copier reality:

- duplicated QR identity markers
- human-readable fallback identifiers
- registration markers for scan normalization

The goal is not a beautiful publishing pipeline. The goal is reliable recovery
under ordinary institutional scanning conditions.

### Ingest scans without guessing

The ingest path:

- accepts scan images
- reads identity markers where possible
- normalizes pages back into canonical page space
- associates pages with the right exam instance
- records clear outcomes for matched, unmatched, and ambiguous cases

Idempotence matters here. Re-ingesting the same material must not corrupt state,
and failure cases must remain visible instead of disappearing into best-effort
heuristics.

### Grade and surface uncertainty

Once a page is identified and registered, the system reads responses, applies
the correct per-student key, and scores what it can score lawfully.

The grading design goal is simple:

- clear answers should be scored automatically
- ambiguous answers should be surfaced
- the system should not fake certainty it does not have

That applies both to deterministic MC grading and to the short-answer VLM
surfaces.

### Review and export

The final operator workflow includes:

- resolving unmatched or ambiguous cases
- reviewing flagged marks
- finalizing grades
- exporting results

That workflow exists in the local web GUI and remains backed by the same
database truth as the rest of the system.

## Current shape of the product

Today the strongest completed lane is the multiple-choice paper workflow:

- authored assessments in the local web GUI
- generated answer-sheet artifacts and printable PDFs
- QR-backed paper identity and scan normalization
- batch scan ingest with explicit ambiguity outcomes
- deterministic MC grading with a review queue
- database-backed result persistence and export

The repo also contains a real short-answer grading and observability lane:

- VLM-backed grading smoke harness
- Project Paint Dry live grading view
- focus-region annotation and refresh
- run comparison and truth/backfill tooling

That short-answer lane is where most of the active model and prompt work lives.

## Local GUI and dev surfaces

The primary human-facing surface is the local web GUI.

The developer and operator-facing surface for live short-answer work is Paint
Dry.

Both matter, but they solve different problems:

- the GUI is the product workflow
- Paint Dry is the live observability and intervention surface

## Eval harness model servers (dev only)

The eval harness (`scripts/smoke_vlm.py`) talks to OpenAI-compatible local
servers for two logical roles:

- the grader model
- the Paint Dry narrator

Those roles may run on separate servers or on the same server.

The standard split configuration is:

- grader on the big-box OMLX server at `http://macbook-pro-2.local:8001`
- narrator on the dedicated Bonsai surface at `http://nlm2pr.local:8002`

The harness also supports a single-model configuration, where the same strong
model serves both grading and narration. In that mode, `--narrator-url` and
`--narrator-model` point at the same backend as the grader.

The durable default narrator address for the separate Bonsai lane is
`http://nlm2pr.local:8002`; that mDNS name lets the narrator follow the correct
machine instead of depending on which box you happen to be sitting on. Use
`http://127.0.0.1:8002` only as an explicit local-box override when you have
intentionally launched Bonsai on the same machine running the smoke.

Bonsai needs the PRISM MLX fork specifically — stock MLX doesn't support
`bits=1` quantization. Setup, launch command, verification, and troubleshooting
are documented in [`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md).

## Project Paint Dry — live grading view (dev only)

When the eval harness runs with `--narrate`, it opens the Paint Dry reader: a
live grading view over the model's reasoning and the current evidence crop.

Paint Dry is where short-answer grading becomes inspectable. It is especially
useful on the cases that are annoying to reason about from final artifacts
alone:

- a scribbled digit that could be read two ways
- a structure that is locally wrong but contextually salvageable
- a disagreement where the real question is what the model is defending

Paint Dry supports both:

- a dedicated narrator model on its own server
- a single-model path where the same strong grader model also narrates

The reader shows:

- live narration
- the active focus preview
- score accounting
- retained history with scrollback
- a rejected-summary/debug surface

What it is good for:

- smoke visibility
- ambiguity diagnosis
- prompt tuning
- operator correction during a live run

The history pane is scrollable throughout the run:

- `k` / `j` move one row
- `u` / `d` page up/down
- `0` returns to the live edge
- `a` reopens focus-region annotation for the current item

While scrolled up, new history continues to accumulate without yanking the
viewport back to newest, and rows that fall out of the default live-edge view
remain recoverable through scrollback.

Each run persists narrator output to `runs/<ts>-<model>/narrator.jsonl`
(machine-replayable) and `runs/<ts>-<model>/narrator.txt` (human-readable
transcript), alongside the ordinary grading artifacts.

The focus preview uses canonical focus-region data from
`eval/focus_regions.yaml`. Those regions can be reviewed and adjusted with
`scripts/annotate_focus_regions.py`, and the reader can reopen that annotator
directly for the active item via `a`, then refresh the preview in place.

## Workflow

### Professor flow

1. Create or open a project.
2. Import a roster.
3. Author or select an assessment in the local web GUI.
4. Generate printable paper artifacts.
5. Print and administer the assessment.
6. Ingest the scan batch.
7. Review the flagged cases.
8. Finalize and export results.

### Failure modes the system is expected to survive

- duplicate scans or rescans
- unreadable QR codes
- missing pages
- ambiguous marks
- unexpected pages in a scan batch
- accidental reruns of ingest or grading
- roster changes between runs

These are not edge-case fantasies. They are normal workflow hazards, and the
system should surface them explicitly instead of quietly improvising around
them.

## Current frontier

The next forcing case is the real short-answer Quiz 5 family in
`auto-grader-assets/exams/`:

- `260326_Quiz _5 A.pdf`
- `260326_Quiz _5 B.pdf`

That lane is about reconstructing a canonical authored family from the legacy
variants, generating a sibling `C`, and then running a DB-backed VLM-facing
trial against the real student scripts.

## Data model

This is high-level and intentionally does not commit to table names yet.

For the initial schema slice and its contract tests, we intentionally adopt a concrete
working vocabulary so the first migrations and tests are not underspecified. The default
v0 names are:

- `students`
- `template_versions`
- `exam_definitions`
- `exam_instances`
- `exam_pages`
- `scan_artifacts`
- `grade_records`
- `audit_events`
- `mc_scan_sessions`
- `mc_scan_pages`
- `mc_question_outcomes`
- `mc_review_resolutions`

Those names are not meant as an irreversible architectural claim, but they are the
default names the first schema/tests will target.

Entities and relationships are expected to include:

- student identity (possibly hashed or otherwise privacy-preserving representations of
  institutional IDs)
- roster membership (which students are included for an exam)
- template definitions (versioned)
- exam definitions (composition of templates plus configuration, versioned)
- exam instances (per student per exam attempt; includes identifiers and generation
  metadata)
- scan artifacts (file paths, checksums, decoded IDs, status)
- extracted responses (per question)
- grade records (computed plus finalized state)
- audit trail and events (likely useful, even if lightweight)

For the initial schema contract, exam instances also carry a required,
paper-facing `opaque_instance_code` that is globally unique for the generated artifacts.
The initial `audit_events` table is intentionally lightweight but should still capture a
subject (`entity_type`, `entity_id`), an `event_type`, and a payload with a recorded
timestamp.

This spec intentionally avoids prematurely committing to exact naming, indexing strategy,
or normalization patterns, but it assumes relational constraints will be used to prevent
impossible states.

## Critical invariants

These are conceptual. Implementation should encode them via DB constraints, application
logic, and tests.

- A student must not end up with two finalized grades for the same exam attempt.
- An exam instance must be uniquely identifiable and traceable from paper artifacts.
- Template and exam versions used for a generated instance must be immutable and
  reconstructable.
- Ingestion must never silently guess identity; unmatched or ambiguous scans must be
  quarantined.
- Duplicate ingestion must not duplicate or corrupt results.
- Automated grading must never confidently produce an answer where confidence is low;
  ambiguity must be surfaced.
- Failed scans must remain auditable artifacts with clear status and traceable provenance.
- The system must prefer explicit state transitions over implicit best-effort mutation.

## Schema contract authority and test policy

To keep behavior stable and auditable, we treat docs and tests as a paired contract:

- This README defines intent-level invariants, vocabulary, and workflow expectations.
- `tests/test_db_connection_contract.py` defines the enforceable `create_connection`
  API and URL-handling contract for the Postgres hard cut.
- `tests/test_db_postgres_contract.py` defines the enforceable Postgres schema
  contract when run against an explicit disposable `TEST_DATABASE_URL`.
- `tests/test_mc_scan_db_contract.py` defines the enforceable MC scan session
  DB persistence contract: idempotency, supersession, divergence detection,
  atomicity, and review-required flag propagation.
- `tests/test_template_schema_contract.py` defines the enforceable YAML template
  schema contract: structural validation, variable declarations, answer-type rules,
  expression evaluator safety, and integration tests against the real CHM 141 exam
  template.
- `tests/test_generation_contract.py` defines the enforceable initial MC answer-sheet
  generation contract for the deterministic OpenCV lane.
- `python -m auto_grader.contract_test_runner` is the preferred low-friction local
  entrypoint for the authoritative contract suites. It always runs the metadata,
  connection, Postgres harness, runner, template schema, eval harness, shimmer,
  generation, PDF rendering, scan readback, scan registration, bubble
  interpretation, scoring, matched-page extraction, paper-calibration, threshold,
  and discovery guardrail contracts, and includes the Postgres schema and MC
  scan DB contracts
  when `TEST_DATABASE_URL` is set.
- Contract changes must update both this README and contract tests in the same change.
- New schema behavior should follow fail-first discipline:
  add a non-vacuous failing contract test first, then implement.
- Contract tests should assert concrete invariants, not only "it runs" signals.

## Template format goals

The authoring schema should support:

- variable declarations (type, constraints, formatting expectations)
- prompt text with placeholders
- correct answer computation or specification
- distractor strategies or explicit distractors
- option shuffling and stable per-student mapping
- validation tooling (static validation plus randomized sampling sanity checks)
- versioning (explicit or content-hash based)

The schema should remain compact and strict. A small, validated vocabulary is preferable
to permissive anything-goes configuration.

## Helper tools

These tools reduce professor burden and prevent the blank YAML from scratch tax.

- Template validator: checks schema validity, missing fields, unsafe configs, and likely
  collisions such as duplicate choices after rounding.
- Preview generator: shows a single generated instance, or several, for sanity checking.
- Roster import/export tools: CSV parsing, consistent identifiers, minimal friction.
- Skeletonizer (non-ML): takes an existing exam document and drafts a template by
  identifying numeric literals and replacing them with placeholders.
- Optional LLM-assisted inferencer (local): drafts template configs from multiple
  historical versions of a question, producing a reviewable YAML with plausible variable
  ranges and formatting.

## Local LLM-assisted inference

This is explicitly an authoring helper, not a grading dependency.

### Motivation

If the professor already has multiple versions of the same question across past exams
where only numbers changed, much of the variable structure is inferable. The professor
would rather review plausible defaults than fill everything from scratch.

### Approach (conceptual)

For each question:

- Input: multiple historical variants of the same question, ideally 2-3 or more,
  including prompts and answer keys or answers.
- The model proposes:
  - which literals should become variables versus constants
  - variable types (`int` or `float`), formatting expectations (significant figures), and
    plausible ranges
  - an initial correct-answer computation rule, where inferable
  - plausible distractor patterns, where inferable
- Output: a draft YAML/JSON template and a structured reasoning/debug artifact describing
  what it inferred.

### Guardrails

- The output is not trusted by default.
- Drafts must be reviewable and pass deterministic checks before being accepted.
- Checks may include:
  - reconstruction against historical variants, where possible
  - sanity validation of formatting rules and uniqueness of correct answers
  - sampling-based validation to catch collisions or obviously bad ranges
- Ranges are allowed to be broad and coherently plausible rather than perfect.
- The goal is to reduce professor effort by providing reasonable defaults and letting them
  tighten as needed.

### Why do it at all?

- Portfolio relevance is a nice side benefit.
- The real value is reducing professor friction and avoiding the fill 60 configs by hand
  failure mode.

## Known favorable constraints

This project benefits from several unusually favorable constraints:

- Development and deployment both target recent Apple Silicon Macs on closely matched
  macOS versions.
- The professor's laptop is accessible for debugging, deployment, and packaging work.
- Historical scan images from the existing institutional scanning workflow are available
  for evaluating image quality and consistency, even though they do not contain the new
  identity markers.
- The observed scan quality appears closer to high-quality photographic copies than
  aggressively compressed black-and-white document scans.
- Registration consistency in the existing scan corpus appears high.
- The paper format is under project control, which substantially reduces CV complexity
  compared to hostile or unconstrained document inputs.

These constraints should be treated as real scope reducers, not incidental details.

## Local-first operational constraints

Assumptions:

- Development and deployment target recent Apple Silicon Macs on closely matched macOS
  versions.
- The professor uses their own laptop, without university device-management complications.
- The core should function offline and avoid cloud dependencies.
- Initial packaging may be lightweight.
- Cross-platform support is explicitly out of scope for early versions.

Database development strategy:

- Build Postgres-first from the start (no "SQLite now, migrate later" phase).
- Use native local Postgres on the main laptop (no Docker requirement).
- Keep memory usage conservative so inference workloads stay prioritized.
- Keep all DB configuration URL-driven so we can move the DB host later if needed.
- Optional fallback: if local Postgres materially interferes with model workloads, move
  Postgres to a second laptop and keep application code unchanged except `DATABASE_URL`.

Locked Postgres contract decisions (v0):

- Connector: use `psycopg` v3, with `psycopg[binary]` as the default local dev/test
  install target.
- `create_connection` API: `create_connection(database_url=None, connect_fn=None)`.
- URL source precedence: explicit `database_url` wins over `DATABASE_URL`.
- Invalid explicit `database_url`: raise `ValueError`; never fallback to env URL.
- Missing explicit URL: read `DATABASE_URL`; if missing or blank, raise `ValueError`.
- Accepted URL schemes: `postgres://` and `postgresql://` (case-insensitive).
- URL handling policy: validate early; reject blank values and leading/trailing
  whitespace; normalize accepted Postgres scheme casing to lowercase for connector
  compatibility; otherwise avoid heavy normalization.
- Nonblank text semantics: DB-enforced "nonblank" text fields reject any all-
  whitespace value, including spaces, tabs, and newlines.
- Legacy rerun policy for strengthened nonblank constraints: `initialize_schema()`
  intentionally hard-fails on existing schemas that still contain newly
  forbidden whitespace-only values under legacy `btrim(...) <> ''` checks;
  clean those rows before rerunning schema initialization.
- Legacy SQLite path API: unsupported (hard-cut from `path`-style connection contract).
- Primary keys for v0: surrogate integer identity keys; keep business uniqueness as
  separate constraints.
- Versioned table immutability: enforce at DB layer with triggers (for example
  `template_versions`, `exam_definitions`).
- Event payload and timestamp storage: `JSONB` payloads and `TIMESTAMPTZ` timestamps,
  treated as UTC in app logic/tests.
- Postgres contract authority: keep the connection and schema contract suites green
  against an explicit disposable `TEST_DATABASE_URL`.

Preferred commands:
- Default local contract run:
  `python -m auto_grader.contract_test_runner`
- Explicit DB-backed contract run:
  `TEST_DATABASE_URL=... uv run python -m auto_grader.contract_test_runner --require-postgres`
- One-command disposable DB-backed run:
  `./scripts/run_postgres_contracts.sh`

Disposable local Postgres for contract tests:
If you just want the low-friction path, run `./scripts/run_postgres_contracts.sh`.
It bootstraps a disposable UTF-8 local cluster, sets `TEST_DATABASE_URL`, runs the
authoritative DB-backed contract runner, and tears the cluster down afterward.
It binds Postgres to a per-run Unix socket directory with TCP disabled; set
`AUTO_GRADER_POSTGRES_PORT` if you want to pin an explicit socket port instead.

1. Create a temporary native Postgres cluster and socket directory. The per-run
   socket directory keeps this isolated from unrelated local TCP listeners; the
   `55432` port just gives the socket file and DSN a predictable name. Pin UTF-8
   explicitly so the contract suite does not false-red against a SQL_ASCII cluster.
   ```bash
   tmp_root="$(mktemp -d /tmp/auto-grader-pg.XXXXXX)"
   mkdir -p "$tmp_root/socket"
   initdb -D "$tmp_root/data" -U postgres -A trust --no-locale -E UTF8
   pg_ctl -D "$tmp_root/data" \
     -l "$tmp_root/postgres.log" \
     -w \
     -o "-k $tmp_root/socket -p 55432 -h ''" \
     start
   ```
2. Verify the disposable server is reachable through the Unix socket directory.
   ```bash
   psql -h "$tmp_root/socket" -p 55432 -U postgres -d postgres -Atqc 'select 1'
   ```
3. Run the DB-backed contract suite against that disposable instance.
   ```bash
   export TEST_DATABASE_URL="postgresql://postgres@/postgres?host=$tmp_root/socket&port=55432"
   uv run python -m auto_grader.contract_test_runner --require-postgres
   ```
4. Stop the disposable server when finished.
   ```bash
   pg_ctl -D "$tmp_root/data" stop
   ```

Packaging is important but not a v0 requirement. v0 can be developer-run. v1 should
minimize terminal rituals for the professor.

## What done looks like (MVP definition)

An MVP is successful when:

- a professor can define a small set of templates in YAML/JSON
- the system generates per-student exam instances with correct, varied answer keys
- page-level identity markers on paper allow reliable scan association
- scan ingestion is robust to typical scanner/copy workflow behavior
- grading is mostly automatic with a small manual review queue
- results export cleanly and reproducibly
- the system maintains an audit trail and does not silently corrupt state

## Where an agent should start (implementation order)

This is the intended build order for an agent-driven solo workflow:

1. Define the minimal template schema and validator.
2. Implement the database spine, core entities, invariants, and migrations.
3. Implement the generation pipeline (roster -> instances -> PDFs) with page-level
   identity markers.
4. Implement the ingestion pipeline (scan files -> matched pages -> stored artifacts).
5. Implement the grading pipeline (registered page -> response extraction -> scoring).
6. Implement the minimal review workflow and export.
7. Add the skeletonizer helper.
8. Add the optional local LLM inferencer helper, guardrailed by validators.

At every step, prioritize end-to-end vertical slices and reproducibility over feature
breadth.

## Open questions

These are areas to discuss before committing to implementation decisions:

- What is the minimal template spec that covers 80% of the professor's question set?
- How are correct answers expressed: expressions, lookup, or function hooks?
- How strict should formatting and rounding rules be to avoid collisions?
- How much versioning and audit logging is required for the professor's comfort?
- What is the acceptable level of manual review per exam?
- What QR size, redundancy, and error-correction settings are appropriate once tested
  against the actual scan pipeline?
- What registration tolerances and preprocessing steps are actually needed, given the
  observed scan consistency?
- What data privacy constraints matter, such as storing raw student IDs versus hashed
  keys?

Supporting rationale and deferred Postgres topics are tracked in
`docs/postgres_contract_decision_worksheet.md`.
Execution sequencing for the migration is tracked in
`docs/postgres_migration_checklist.md`.

## Guiding principles

Make the common case boring and automatic. Make the weird case survivable and auditable.

Prefer explicit state, explicit logs, and recoverable failure modes over cleverness.
