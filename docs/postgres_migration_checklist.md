# Postgres Migration Checklist

Last updated: 2026-03-20

Purpose:
- Give the migration a visible burn-down list.
- Separate "contract parity still needed" from "runtime implementation work."
- Make it obvious when it is finally safe to retire the legacy SQLite reference suite.

Current status snapshot:
- `tests/test_db_connection_contract.py` exists and defines the Postgres
  `create_connection()` hard-cut contract.
- `tests/test_db_postgres_contract.py` exists and covers a substantial first
  slice of Postgres schema behavior against an explicit `TEST_DATABASE_URL`.
- `tests/test_db_contract.py` still exists as the shrinking SQLite reference
  suite for parity comparison during migration.
- `auto_grader/db.py` is still pre-migration runtime code:
  `create_connection(path=":memory:")` plus stub `initialize_schema(...)`.

## Phase 1: Contract parity before runtime work

- [x] Lock Postgres connector, URL, and `create_connection()` API decisions in
  docs and contract tests.
- [x] Add a project-metadata contract for local bootstrap and Postgres driver
  declaration.
- [x] Build a real Postgres contract harness gated by explicit
  `TEST_DATABASE_URL` and isolated schemas.
- [x] Cover all eight workflow tables in the Postgres suite at the table-shape
  level.
- [x] Cover core invariants for students, exam instances, exam pages,
  scan artifacts, grade records, audit events, template versions, and exam
  definitions.
- [x] Cover locked Postgres-specific decisions:
  `JSONB`, `TIMESTAMPTZ`, DB-level immutability for versioned rows, and the
  Postgres-first connection contract.
- [x] Audit the remaining legacy SQLite reference tests one by one and either:
  port the invariant to Postgres, or explicitly decide it is intentionally
  retired/merged into another stronger Postgres test.
- [x] Write down the disposition of every remaining legacy-only assertion so the
  suite can be retired without ambiguity.

## Remaining parity audit checklist

- [x] `exam_definitions` update guard parity:
  verify we still explicitly test that updates cannot repoint a row to a missing
  `template_version_id`.
- [x] `scan_artifacts` split parity:
  verify every legacy rule is either ported directly or intentionally merged:
  supported statuses, lowercase-only values, status required, failure reason
  rules, checksum required/nonblank/64-hex shape, filename required/nonblank,
  and uniqueness.
- [x] `grade_records` split parity:
  verify supported statuses, lowercase-only values, nonblank status, status and
  score field requirements, nonnegative `score_points`, positive `max_points`,
  `score_points <= max_points`, and one-finalized-grade rule.
- [x] `audit_events` split parity:
  verify complete subject fields, nonblank `event_type`, payload required,
  nonblank payload, valid JSON only, default timestamp present/current, explicit
  timestamp preservation, and explicit timestamp validation.
- [x] Delete/update lifecycle parity pass:
  verify every legacy FK-protection and allowed-delete case is covered in the
  Postgres suite, not only the high-traffic tables.

## Phase 2: Runtime implementation after parity is acceptable

- [x] Implement `create_connection(database_url=None, connect_fn=None)` in
  `auto_grader/db.py`.
- [x] Make `create_connection()` use `psycopg` v3 with autocommit enabled and
  rows addressable by column name.
- [x] Make the connection contract suite pass:
  `python -m unittest tests.test_db_connection_contract -q`
- [x] Implement Postgres `initialize_schema(connection)` in dependency order.
- [x] Make the Postgres schema contract suite pass:
  `TEST_DATABASE_URL=... uv run python -m unittest tests.test_db_postgres_contract -q`

## Phase 3: Cleanup and cutover

- [x] Decide when the Postgres suite is authoritative enough to stop using the
  SQLite reference suite as a parity checklist.
- [x] Retire `tests/test_db_contract.py` once its remaining assertions have
  either been ported or intentionally dropped with rationale.
- [x] Remove legacy SQLite connection/schema code paths from `auto_grader/db.py`.
- [x] Update README migration wording from "gate before implementation" to the
  steady-state Postgres authority once the suite is green.

Result:
- This checklist is complete for the Postgres hard cut.
- `tests/test_db_connection_contract.py` and
  `tests/test_db_postgres_contract.py` are now the active DB contract suites.
