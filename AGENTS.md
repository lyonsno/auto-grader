# AGENTS.md — auto-grader repo notes

These repo-specific instructions supplement the global Codex defaults.

## Postgres contract suite

- Keep `tests.test_db_postgres_contract` as a real Postgres-backed contract suite.
- Do not weaken that suite with extra mocks just to satisfy sandbox restrictions.
- In Codex's sandbox, local Postgres TCP or Unix-socket connections may fail with
  `Operation not permitted` even when the disposable database is healthy.
- If that happens, treat it as an environment limitation and rerun the DB-backed
  suite outside the sandbox / with escalation instead of changing the contract.
- If DB-backed smoke or schema coverage would materially reduce uncertainty, do
  not hesitate to run it. Prefer requesting escalation / permissions over
  skipping the run due to sandbox limits.
- The always-on non-DB contract suites should remain sandbox-safe.

Preferred DB-backed commands:

- Single suite:
  `TEST_DATABASE_URL=... .venv/bin/python -m unittest tests.test_db_postgres_contract -q`
- Full contract runner with Postgres required:
  `TEST_DATABASE_URL=... .venv/bin/python -m auto_grader.contract_test_runner --require-postgres`
- One-command disposable DB-backed run:
  `./scripts/run_postgres_contracts.sh`

Notes:

- Prefer a disposable local Postgres instance for contract runs.
- A Unix-socket DSN is fine when local TCP is restricted by the environment.
