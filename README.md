# Personalized Exam Generator + Local Auto-Grading

Project README / Scope Spec

## What this is

This project is a local-first system for generating per-student variants of chemistry exams
(Chem 101 / Orgo 101 style) from reusable question templates, then grading them
automatically from scanned paper submissions using a robust, intentionally boring
computer-vision pipeline.

The point is exam integrity (each student gets their own numerical variant and answer-key
mapping) and operational sanity (grading becomes fast, auditable, and resilient to the
ordinary weirdness of paper/scanner workflows without assuming chaotic inputs).

This README is intentionally written as a scope/spec document for a project that has been
scoped well enough to start building, but not yet implemented. It emphasizes problem
space, invariants, workflow, and favorable constraints rather than premature
implementation detail.

## Non-goals

These are here to keep the project from exploding.

- No full question-type library with a GUI editor. Authoring is power-user YAML/JSON plus
  helper scripts.
- No LMS integration (Canvas, Blackboard, etc.) in v1. CSV in and out is sufficient.
- No cloud dependency required for the core workflow. The system should run on a
  professor's laptop.
- No attempt to solve all chemistry. v1 focuses on numeric parameterization and
  multiple-choice grading, with a path to expand later.
- No cross-platform support requirement in early versions. Initial development and
  deployment target recent Apple Silicon Macs only.

## Core ideas and decisions already locked in

- Database-backed spine: the system is stateful and auditable; it uses a database to
  preserve identity, reproducibility, and workflow state.
- Template-driven exam generation: exams are built from a schema describing templates and
  variable parameterization.
- Per-student exam instances: each student receives a unique variant with unique
  answer-key mapping.
- Page-level identity markers on paper: printed artifacts include duplicated QR codes and
  human-readable fallback codes for reliable scan association and manual recovery.
- Local scan ingestion and grading: the grading pipeline operates on scanned photographic
  copies (batch scans) using local CV tooling.
- Local LLM-assisted template inference (optional helper): a local model can assist in
  inferring templated configs from historical exam variants, but outputs must be
  reviewable and verifiable. The model is not a source of truth.
- Tamper-evident, auditable workflow: the system is designed so anomalies are surfaced,
  quarantined, and traceable rather than silently guessed away.

## Why a database is not optional here

This system interacts with the physical world. Paper moves, pages get shuffled, scans
duplicate, files get re-uploaded, and mistakes happen. The database exists to:

- map paper artifacts back to canonical exam instances
- preserve what happened for audit and dispute resolution
- enforce invariants like no two finalized grades for the same student plus exam attempt
- support idempotent ingestion so re-scans do not corrupt state
- enable reconciliation between the expected set of exam artifacts and the observed set of
  scanned artifacts

The database should not be treated as a dumping ground for blobs. It is the spine of truth
and process.

## System overview

### 1. Authoring (Templates)

Professors create question templates and exam definitions using YAML/JSON. The authoring
format is designed for researchers: structured, explicit, validated, and not dependent on
terminal expertise.

Key properties:

- templates represent "same question, different numbers"
- variables are typed (`int` or `float`), constrained (ranges, step), and formatted
  (significant figures and rounding rules)
- questions define correct answers and distractor generation strategies, or allow explicit
  distractor lists
- template versions are immutable once used to generate student exams

### 2. Generation (Per-student exams)

Given:

- a roster
- an exam definition (template set plus configuration)

The system generates:

- per-student exam instances
- per-student answer keys (mapping correct choice to bubble position)
- printable PDFs (exam pages and answer sheets, depending on layout)

A central requirement is reproducibility. It must always be possible to reconstruct what a
given student received, even later.

Initial contract slice for the deterministic MC lane:

- generation must be able to emit a per-student MC answer-sheet artifact before the full
  PDF pipeline exists
- the artifact must include a stable, non-semantic `opaque_instance_code` that does not
  expose student or template identifiers
- it must include a per-page fallback code for paper recovery
- it must include rendered MC questions with any shuffled choice order made explicit
- variableized MC rendering must be stable across equivalent variable declaration order
- it must include the answer-key mapping from logical choice key to physical bubble label
- it must include canonical page-space bubble rectangles for each rendered bubble
- it must declare the layout coordinate contract explicitly (`units`, `origin`, `y_axis`,
  `layout_version`) so later PDF and OpenCV stages do not invent a second layout truth

This narrow artifact is the handoff between generation and deterministic OpenCV grading.
Later PDF rendering should consume the same contract rather than inventing a second layout
truth.

### 3. Paper artifacts (Identification + Layout)

Printed artifacts include:

- duplicated QR codes encoding an opaque exam-instance identifier plus page number
- human-readable fallback identifiers for manual recovery
- alignment markers and registration aids for scan normalization

The goal is not fancy. The goal is "works under copier reality."

Current implementation status on the MC/OpenCV prerequisite lane:

- `auto_grader.generation` produces the canonical answer-sheet artifact
- `auto_grader.pdf_rendering` now renders a minimal answer-sheet PDF directly
  from that artifact
- the rendered PDF currently carries visible instance/page recovery codes,
  rendered prompt text, filled registration markers for scan normalization,
  circular answer bubbles with direct `A/B/C/D` labels, and a vertically
  stacked choice list placed underneath each prompt to the left of the bubble
  row, with larger question typography and human-markable bubble sizing, all
  still derived from the same page contract
- dense MC sheets now paginate across multiple pages instead of forcing a fake
  one-page contract, and long prompts wrap within the question block rather
  than bleeding across the sheet
- QR-code placement is still future work; it should be added as an explicit
  extension of the same page artifact rather than as a second layout truth

### 4. Ingestion (Scans -> Identified pages)

The ingestion pipeline:

- accepts scanned page images produced by the institutional batch scanning system
- decodes identity markers where possible
- registers and normalizes the page to a canonical coordinate space
- associates each scan with the correct exam instance and page number
- stores ingestion outcomes (`matched`, `unmatched`, `ambiguous`, `duplicate`) without
  silent corruption

Importantly:

- ingestion must be idempotent
- failed identification attempts must still be recorded as tracked artifacts with explicit
  failure reasons
- ingestion must never silently drop pages
- ingestion must never silently guess identity

For the initial schema contract, the supported scan status vocabulary is `matched`,
`unmatched`, and `ambiguous`. `matched` is treated as a success state and should not carry
a failure reason. `unmatched` and `ambiguous` are treated as supported tracked failure
states and must record an explicit, non-blank failure reason. Exact re-ingestion of the
same file should be idempotent via a unique checksum key. Richer `duplicate` artifact
linkage is deferred until there is canonical-artifact linkage to point at.

### 5. Grading (Bubbles -> Responses -> Score)

Once pages are identified and registered:

- bubble regions are interpreted
- responses are recorded with confidence and ambiguity flags
- answer keys are applied per-student variant
- scores are computed
- ambiguous cases are flagged for review rather than guessed

### 6. Review + Export

The system needs a minimal workflow for:

- resolving unmatched scans by manual entry of fallback identifiers
- reviewing ambiguous marks
- finalizing grades
- exporting results as CSV

A rough, cheap GUI is acceptable. A local web UI served from the app is acceptable. The UI
is a tool, not the product.

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
- `tests/test_template_schema_contract.py` defines the enforceable YAML template
  schema contract: structural validation, variable declarations, answer-type rules,
  expression evaluator safety, and integration tests against the real CHM 141 exam
  template.
- `tests/test_generation_contract.py` defines the enforceable initial MC answer-sheet
  generation contract for the deterministic OpenCV lane.
- `python -m auto_grader.contract_test_runner` is the preferred low-friction local
  entrypoint for the authoritative contract suites. It always runs the metadata,
  connection, Postgres harness, runner, template schema, generation, and discovery
  guardrail contracts, and includes the Postgres schema contract when
  `TEST_DATABASE_URL` is set.
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

## Project workflow

### Professor flow (happy path)

1. Create or open a project folder.
2. Import roster CSV.
3. Create or edit templates, or run skeletonizer / LLM inferencer and review drafts.
4. Validate templates.
5. Preview a few generated variants.
6. Generate exam PDFs.
7. Print and administer exams.
8. Drop scan files into an ingestion folder, or import them.
9. Run grading and review flagged items.
10. Finalize and export grades CSV.

### Failure modes that must be supported

These are expected tail cases, not assumed common-case behavior.

- Scans contain duplicates or rescans.
- Some pages have unreadable QR codes.
- Some pages are missing.
- Some bubble marks are ambiguous.
- A scan batch contains unexpected pages.
- A professor reruns ingestion or grading by accident.
- The roster changes between runs and the system must respond with explicit, logged
  behavior.

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
