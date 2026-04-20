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

This README started as a scope/spec document, but it now also needs to stay honest about
the slices that already exist. It still emphasizes problem space, invariants, workflow,
and favorable constraints, but the MC/OpenCV sections below describe real implemented
surfaces rather than a purely aspirational plan.

## Non-goals

These are here to keep the project from exploding.

- No full question-type library with polished end-to-end GUI support. The
  landed professor-facing authoring surface now covers the real MC workflow and
  assessment-authoring path, but unsupported or more advanced question types
  still require schema-level work and follow-on implementation.
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

The durable authored representation is still YAML/JSON, but the project now
has a professor-facing local GUI for the first real authoring and workflow
surfaces. Professors no longer need to start from blank YAML just to create an
assessment, create/select a grading target, or drive the MC grading workflow.
The underlying format remains structured, explicit, validated, and reviewable.

Key properties:

- templates represent "same question, different numbers"
- variables are typed (`int` or `float`), constrained (ranges, step), and formatted
  (significant figures and rounding rules)
- questions define correct answers and distractor generation strategies, or allow explicit
  distractor lists
- template versions are immutable once used to generate student exams

Current implementation status on this authoring surface:

- `auto_grader.mc_workflow_gui` plus
  `scripts/launch_mc_workflow_gui.py` now provide a thin local web GUI for:
  - creating assessments through a professor-facing authoring flow
  - creating or selecting the exam to grade without exposing raw database ids
  - ingesting scans, reviewing flagged questions, persisting decisions, and
    exporting results
- the GUI intentionally remains thin over the landed durable model and DB-backed
  workflow surfaces instead of inventing a second source of truth
- static-choice multiple-choice authoring is landed and validated before
  persistence
- computed-distractor authoring is intentionally fenced off until the
  generation/rendering path supports it end to end

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

Current implementation status on the generated-exam demo surface:

- `auto_grader.generated_exam_demo` and
  `scripts/render_generated_mc_exam_demo.py` can now load a real exam template
  such as `templates/chm141-final-fall2023.yaml`, build one printable MC-only
  answer-sheet artifact for a named student through the canonical generation
  path, and write the resulting PDF plus artifact/metadata JSON bundle to disk
- this surface deliberately reuses `load_template(...)`,
  `build_mc_answer_sheet(...)`, and `render_mc_answer_sheet_pdf(...)` instead
  of inventing a packet-local generation fork, so the later paper demo can run
  against a real generated exam rather than only the calibration packets
- that generated artifact shape is also now pinned against the landed
  `auto_grader.mc_opencv_demo` runner, so a real chemistry-template packet can
  already flow through the same end-to-end ingest/scoring surface used by the
  earlier calibration-packet demo

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
- each rendered page now carries two duplicated QR codes encoding the page
  fallback code from the same artifact contract
- the rendered PDF currently carries visible instance/page recovery codes,
  rendered prompt text, filled registration markers for scan normalization,
  duplicated QR identity markers, circular answer bubbles with direct `A/B/C/D`
  labels, and a vertically stacked choice list placed underneath each prompt to
  the left of the bubble row, with larger question typography and human-markable
  bubble sizing, all still derived from the same page contract
- dense MC sheets now paginate across multiple pages instead of forcing a fake
  one-page contract, and long prompts wrap within the question block rather
  than bleeding across the sheet
- `auto_grader.scan_readback` now performs the first OpenCV-facing readback
  slice by decoding duplicated page-identity QR payloads from scan images and
  rejecting mismatched payload pairs as ambiguous
- `auto_grader.scan_registration` now performs the first page-registration
  slice by using the filled corner markers to normalize a skewed page back into
  canonical page aspect and marker-aligned page space while preserving practical
  scan resolution
- `auto_grader.bubble_interpretation` now performs the first bubble-readback
  slice by reading filled bubble labels from the normalized page image while
  keeping blanks explicit and preserving multiple marks as visible ambiguity
  instead of collapsing them into fake single answers; that layer has also now
  survived a first real-paper calibration pass, and its `marked` versus
  `illegible` boundary has been retuned against actual scanned pencil fills
  rather than only procedural probes

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

Current implementation status on this ingest surface:

- scan-level MC ingest packaging is implemented via `auto_grader.mc_scan_ingest`
  so one batch of scanned page images plus a known exam artifact can now produce
  explicit `matched`, `unmatched`, and `ambiguous` outcomes without forcing
  downstream callers to hand-compose QR readback, page matching, and matched-page
  extraction
- unique matched scans carry the full normalized/scored page bundle from
  `auto_grader.mc_page_extraction`
- duplicate claims on the same page code stay explicit `ambiguous` outcomes
  instead of silently picking a canonical scan
- unreadable scans remain tracked `unmatched` artifacts with explicit failure
  reasons
- durable on-disk scan-session persistence is implemented via
  `auto_grader.mc_scan_session`; it writes one `session_manifest.json` plus
  per-scan normalized page images for matched scans, keeps exact re-runs
  idempotent at the artifact identity level, rejects unsafe `scan_id` values
  before they become output paths, and now treats normalized-image write
  failures as real errors instead of silently replacing output files with
  corrupt temp data
- a thin prototype demo runner is implemented via `auto_grader.mc_opencv_demo`
  plus `scripts/run_mc_opencv_demo.py`; it takes one known artifact JSON plus
  one directory of scans, runs the landed ingest surface, writes a sanitized
  `ingest_result.json`, writes per-scan normalized page images for matched
  pages, and emits a small human-readable summary so the lane can be smoked end
  to end without inventing a second persistence format; that runner has now
  also survived a first real-paper smoke against the durable two-page
  pencil-and-scanner calibration packet, and the only bug it exposed was a
  page-local scoring-accounting issue that has since been fixed in
  `auto_grader.mc_page_extraction`

### 5. Grading (Bubbles -> Responses -> Score)

Once pages are identified and registered:

- bubble regions are interpreted
- responses are recorded with confidence and ambiguity flags
- answer keys are applied per-student variant
- scores are computed
- ambiguous cases are flagged for review rather than guessed

Current implementation status on this grading surface:

- page-identity QR readback is implemented via `auto_grader.scan_readback`
- page registration is implemented via `auto_grader.scan_registration`
- first-pass bubble readback is implemented via `auto_grader.bubble_interpretation`
- first-pass scoring decisions are implemented via `auto_grader.mc_scoring`
  for the core MC statuses `correct`, `incorrect`, `blank`, `multiple_marked`,
  `ambiguous_mark`, and `illegible_mark`; that scorer now also performs a
  narrow question-level dominance arbitration pass so one clearly stronger
  deliberate fill can beat much weaker secondary traces without forcing routine
  manual review, while genuinely substantial co-equal fills still remain
  explicit review work
- matched-page extraction packaging is implemented via `auto_grader.mc_page_extraction`
  so downstream callers can consume one bundle containing the normalized page,
  marked bubble labels, per-bubble evidence, and scored MC outcomes for an
  already-matched page
- a synthetic student-mark smoke harness is implemented via
  `auto_grader.mark_profile_smoke`; it renders plausible filled, scribbled,
  off-center, smudged, faint, double-marked, hostile-glance, glancing-stray-only,
  tiny-center-dot, correct-plus-wrong-dot, ambiguous-patchy, and
  scratchout-illegible bubbles, runs them through a small ladder of
  realistic scan variants (`clean_scan`, `office_scan`, `stressed_scan`) on the
  real QR/readback/registration/scoring path, and records both the observed
  behavior band (`grade`, `review`, `ignore`, or outright pipeline failure) and
  the current strongest-handled boundary without requiring an immediate
  pen-and-scanner loop
- that same harness now also reports an explicit incidental-mark pathology line
  on the ordinary `office_scan` tier: the strongest specimen still ignored as
  non-attempt noise, and the first stronger specimen that the current pipeline
  no longer feels safe ignoring; today that line sits between a short interior
  slash that is still ignored and a compact dark interior scribble that is
  treated as a real fill attempt
- a dedicated printable paper-calibration packet is implemented via
  `auto_grader.paper_calibration_packet`; the first real pencil-and-scanner
  pass on that packet exposed an over-aggressive `illegible` heuristic, and the
  follow-on retune brought the same two scanned pages from 3/12 page-local
  matches to 12/12 without weakening the existing scratchout-to-`illegible`
  contract
- a human review-override surface is implemented via
  `auto_grader.mc_review_override`; reviewers can now resolve flagged
  `multiple_marked`, `ambiguous_mark`, and `illegible_mark` cases to a final
  bubble choice or blank result while preserving the original machine status as
  provenance in the corrected scoring record
- database-backed MC scan session persistence is implemented via
  `auto_grader.mc_scan_db`; scan sessions, per-page match status, and
  per-question scored outcomes are written atomically into the DB spine with
  idempotent re-run semantics, append-only supersession for re-scans, and
  divergence detection when overlapping scans carry different outcomes across
  sessions
- database-backed MC review-resolution persistence is implemented via
  `auto_grader.mc_review_db`; human override decisions are now recorded
  durably against persisted machine question outcomes with original-status
  provenance, current resolved bubble or blank outcome, idempotent re-apply
  semantics for identical decisions, and audit-event history when a reviewer
  later changes the decision
- database-backed current-final MC truth reads are implemented via
  `auto_grader.mc_results_db`; callers can now ask for one authoritative
  current result surface per `exam_instance_id`, with the latest scan session,
  persisted machine outcomes, latest human review resolutions, and still-open
  review-required questions composed together instead of reconstructed ad hoc
- a compact DB-backed demo/export surface is implemented via
  `auto_grader.mc_results_demo_export` plus
  `scripts/export_mc_results_demo.py`; this intentionally sits downstream of
  the authoritative read model and writes one compact JSON bundle plus one
  human-readable summary file instead of re-inventing result semantics in the
  script layer

### 6. Review + Export

The system needs a minimal workflow for:

- resolving unmatched scans by manual entry of fallback identifiers
- reviewing ambiguous marks
- finalizing grades
- exporting results as CSV

A rough, cheap GUI is acceptable. A local web UI served from the app is acceptable. The UI
is a tool, not the product.

Current implementation status on this review surface:

- `auto_grader.mc_review_override` now provides the first explicit human
  resolution seam for flagged MC questions
- `auto_grader.mc_review_db` now persists those review resolutions into the
  database spine without mutating away the underlying machine-scored outcome;
  the current row stores the latest reviewed answer while audit events preserve
  create/update history
- `auto_grader.mc_results_db` now exposes the authoritative DB-backed current
  MC truth surface for one exam instance, combining the latest persisted scan
  session with any persisted review resolutions while keeping unresolved
  review-required questions explicit
- `auto_grader.mc_db_round_trip` now composes the landed DB primitives into
  one thin workflow seam that persists a scan manifest, optionally persists
  human review resolutions by scan, and returns the authoritative current-final
  MC truth without redefining read-model semantics
- `auto_grader.mc_results_demo_export` now turns that DB-backed current truth
  into a compact demo/export bundle for one exam instance, with a small text
  summary that is suitable for same-day operator/demo use without a GUI
- `auto_grader.mc_workflow` now provides a professor-facing workflow surface
  that composes the landed MC/OpenCV and DB-backed primitives into four
  operations: ingest a directory of scan images into the DB-backed workflow,
  show the review queue, resolve flagged questions using a simple
  `{question_id: bubble_label}` map, and export final results as
  JSON/CSV/text
- `scripts/mc_workflow.py` exposes those operations as CLI subcommands
  (`ingest`, `review`, `resolve`, `export`) so the professor does not need to
  know about internal module boundaries or prepare complex JSON

Example same-day DB-backed round-trip invocation:

```bash
python scripts/run_mc_db_round_trip.py \
  --manifest-json /tmp/mc-session/manifest.json \
  --exam-instance-id 123 \
  --output-dir /tmp/mc-db-round-trip
```

Example same-day export invocation:

```bash
python scripts/export_mc_results_demo.py \
  --exam-instance-id 123 \
  --output-dir /tmp/mc-results-demo
```

Example professor-facing workflow (ingest + review + resolve + export):

```bash
# Ingest scans from a directory into the DB-backed MC/OpenCV workflow
python scripts/mc_workflow.py ingest \
  --exam-instance-id 123 \
  --artifact-json /tmp/mc-generated-exam-demo-artifact.json \
  --scan-dir /tmp/mc-scans \
  --output-dir /tmp/mc-ingest

# See what needs review
python scripts/mc_workflow.py review \
  --exam-instance-id 123 \
  --output-dir /tmp/mc-review

# Resolve flagged questions (simple JSON: question_id -> bubble label or null)
echo '{"mc-1": "B", "mc-3": null}' > /tmp/resolutions.json
python scripts/mc_workflow.py resolve \
  --exam-instance-id 123 \
  --resolutions-json /tmp/resolutions.json \
  --output-dir /tmp/mc-resolve

# Export final results as CSV
python scripts/mc_workflow.py export \
  --exam-instance-id 123 \
  --format csv \
  --output-dir /tmp/mc-export
```

Example first-pass professor GUI:

```bash
python scripts/launch_mc_workflow_gui.py \
  --open-browser \
  --database-url postgresql:///postgres \
  --artifact-json /tmp/mc-generated-exam-demo-artifact.json \
  --scan-dir /tmp/mc-scans \
  --output-dir /tmp/mc-gui-output
```

The GUI is intentionally thin over the landed workflow: it lets an operator
ingest scans, inspect the review queue, persist resolutions, and export final
results without re-owning any of the underlying DB truth or MC/OpenCV logic.
It now also supports a more professor-shaped grading-target flow:

- a primary `Grade Scans` section for choosing an existing exam and ingesting
  scans into it
- a separate `Need a Different Exam?` section for creating a new exam from an
  existing assessment template when the desired target is not already listed
- the GUI keeps internal database ids behind the curtain and exposes the
  creation flow in the same local web surface instead of requiring raw YAML,
  SQL, or ad hoc terminal glue just to pick or create the grading target

## Near-term forcing case

The next concrete forcing case is not another MC packet. It is a real
short-answer quiz family already sitting in `auto-grader-assets/exams/`:

- `260326_Quiz _5 A.pdf`
- `260326_Quiz _5 B.pdf`

These are fixed-layout short-answer chemistry quizzes with typed prompts and
boxed final-answer fields. The observed `A/B` differences appear structured
enough that the project should be able to:

- reconstruct a canonical authored quiz family from the legacy variant PDFs
- generate a reviewable sibling variant `C`
- render QR-/identity-bearing tracked artifacts for `A`, `B`, and `C`
- and run a DB-backed VLM-facing trial against the real student scripts that
  arrive next week

The first part of that chapter is now landed:

- canonical reconstruction of the observed `Quiz 5 A/B` family
- a repo-local CLI for reconstructing the family from the legacy PDFs
- generation of a reviewable sibling variant `C`

The immediate frontier is now the next layer on top of that foundation:

- render QR-/identity-bearing tracked artifacts for `A`, `B`, and `C`
- and run a DB-backed VLM-facing trial against the real student scripts that
  arrive next week

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

## Eval harness model servers (dev only)

The eval harness (`scripts/smoke_vlm.py`) talks to two OpenAI-compatible
local servers — one for the actual grader model (typically Qwen3.5 or
Gemma-4 on the big-box machine, mDNS-resolved at
`http://macbook-pro-2.local:8001`), and one for the "Project Paint Dry"
narrator (Bonsai-8B-mlx-1bit on the local machine, served via a
PRISM-patched `mlx-openai-server` on its own narrator surface). The
durable default narrator address is `http://nlm2pr.local:8002`; that
mDNS name is what the runtime uses so the narrator follows the correct
machine instead of depending on which box you happen to be sitting on.
Use `http://127.0.0.1:8002` only as an explicit local-box override when
you have intentionally launched Bonsai on the same machine running the
smoke. They do not share a port, and the narrator stays separate from
the main grader server on purpose.

Bonsai needs the PRISM MLX fork specifically — stock MLX doesn't
support `bits=1` quantization. Setup, launch command, verification,
and troubleshooting are documented in
[`docs/bonsai_server_setup.md`](docs/bonsai_server_setup.md). Read
that file before trying to start the narrator from scratch.

The grader server on the big box uses standard OMLX with non-1-bit
models and isn't covered by the bonsai doc.

## Project Paint Dry — live narrator (dev only)

When the eval harness runs with `--narrate`, it opens a second terminal
window showing a live play-by-play of the grading VLM's reasoning.
A small local model (Bonsai 8B, 1-bit) watches the VLM's reasoning
token stream and produces short running commentary as each item is
graded — effectively narrating the grading process in real time.

The narrator window (top to bottom):

- **Title bar**: project name, narrator status, and running counters
  for emitted / dedup-rejected / empty-rejected summaries.
- **Scoreboard**: current model, exam set, item counter, and a row of
  tall-digit scoring dials — total elapsed, turn elapsed, on-target
  fraction, left-on-table, and bad calls.
- **Status + live pane**: the current bonsai dispatch streaming
  character-by-character as the VLM thinks, with a one-line status
  label above it.
- **Focus preview**: a cropped scan of the exam region currently being
  graded, rendered inline via Kitty graphics protocol. Shows the
  student's actual handwritten answer for the active item.
- **History pane**: completed narrator summaries grouped by grading
  item, newest at top. Each item shows a header (item ID, question
  type, point value), any accepted narrator lines (core-issue flags,
  basis summaries), and a verdict line with elapsed time, grader
  score vs professor score, and a brief comparison.
- **Rejected pane**: summaries dropped by the dedup or empty filters,
  shown with strikethrough. Debug surface only.

The history pane is scrollable. Key bindings are shown in the footer
after the session ends, but they are active throughout the run:
`k`/`j` scroll up/down one row, `u`/`d` page up/down, `0` returns to
the live edge. While scrolled up, new history continues to accumulate
without yanking the viewport back to newest. After the session ends,
scroll keys remain active and any non-scroll key closes the window.

Each run persists narrator output to `runs/<ts>-<model>/narrator.jsonl`
(machine-replayable) and `runs/<ts>-<model>/narrator.txt` (human-readable
transcript).

The focus preview uses canonical focus-region data from
`eval/focus_regions.yaml`. Those regions can be reviewed and adjusted
with `scripts/annotate_focus_regions.py`, which gives the project one
authoritative focus-box seam instead of ad hoc crop constants scattered
through the smoke and narrator code.

## Project workflow

### Professor flow (happy path)

1. Create or open a project folder.
2. Import roster CSV.
3. Create or edit assessments through the local authoring surface, or run the
   skeletonizer / LLM inferencer and review drafts.
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
