# Postgres Contract Decision Worksheet

Last updated: 2026-03-11

Purpose:
- Keep short, practical rationale notes for Postgres contract decisions.
- Capture deferred topics so we do not over-design before the contract suite is
  fully converted.

Authority:
- Canonical locked contract decisions live in `README.md` under "Locked
  Postgres contract decisions (v0)".
- This worksheet is a companion note, not the source of truth.

Scope:
- v0 contract behavior for local-first development.
- Not a long-term scaling/performance architecture document.

## Quick primer: what a connector does

A connector (driver) is the Python library that speaks Postgres protocol. It
opens connections, sends SQL safely with parameters, manages transactions, and
returns results/errors to Python code.

## Locked decisions and rationale (summary)

### Connector choice

Locked default:
- Use `psycopg` v3.
- Use `psycopg[binary]` for local dev/test ergonomics.

Rationale:
- Most boring/standard Python Postgres option with strong docs.
- Lower setup friction than c-extension-first paths.

### `create_connection` contract shape

Locked default:
```python
def create_connection(
    database_url: str | None = None,
    connect_fn: Callable[[str], Connection] | None = None,
) -> Connection:
    ...
```

Locked behavior:
- Explicit `database_url` wins over `DATABASE_URL`.
- Invalid explicit URL raises `ValueError`; never fallback to env.
- Missing explicit URL reads `DATABASE_URL`.
- Missing/blank URL raises `ValueError`.
- `connect_fn` remains optional for deterministic tests.
- Legacy SQLite `path` contract remains unsupported.

Rationale:
- Keeps production call sites simple.
- Keeps testability high without network dependency.
- Avoids ambiguous dual-mode SQLite/Postgres behavior.

### URL validation semantics

Locked behavior:
- Accept `postgres://` and `postgresql://` schemes (case-insensitive).
- Reject blank/whitespace URL values.
- Validate scheme before connector call.
- If explicit URL is valid, ignore invalid env URL.
- Pass URL through after validation without heavy normalization.

Rationale:
- Early, explicit validation gives cleaner failures.
- Minimal normalization avoids hidden behavior and brittle coupling.

### Primary keys and versioned-row immutability

Locked behavior:
- Use surrogate integer identity keys in v0 core tables.
- Keep business uniqueness in explicit unique constraints.
- Enforce versioned-table immutability in DB with `BEFORE UPDATE` triggers.

Rationale:
- Integer keys keep v0 simple and fast.
- DB-level immutability prevents accidental mutation across code paths.

### JSON and timestamp representation

Locked behavior:
- Use `JSONB` for payload columns.
- Use `TIMESTAMPTZ` for recorded times, defaulting to current timestamp.
- Treat timestamps as UTC in app logic/tests.

Rationale:
- Queryable validated payload storage with minimal complexity.
- Avoid timezone ambiguity in audit data.

### Migration posture

Locked behavior:
- Convert contract tests first.
- Do not implement Postgres runtime wiring until Postgres contract coverage has
  required parity.

Rationale:
- Prevents implementation churn before contract behavior is clear.

## Deferred decisions (not required yet)

- Connection pooling and retry strategy.
- Advanced indexing/perf tuning beyond contract-critical constraints.
- Backup/replication/failover.
- Partitioning/retention policy.
- ORM adoption vs raw SQL long-term architecture.
