# AGENTS.md — auto-grader repo notes

These repo-specific instructions supplement the global Codex defaults.

## Topothesia review surfaces

- `auto-grader` uses Topothesia review surfaces for clean-scan authority and
  contamination-boundary review routing.
- When the human says `Make this durable for review`, consult
  `docs/review_surfaces.toml` and `docs/review-authority-surfaces.md` before
  turning scan-routing or compatibility-fallback findings into code changes.
- Use Topothesia review surfaces when the issue is about authority,
  canonical-vs-fallback interpretation, or allowed divergence between clean and
  contaminated scan paths.
- Use Prilosec instead when the issue is a recurring acknowledged false
  positive or accepted finding family that should be suppressed or demoted
  across future review runs.

## Bonsai narrator server

- On `MacBook-Pro-2.local`, the canonical local Bonsai narrator surface is a
  separate `mlx-openai-server`, not the upstream OMLX server on `:8001`.
  Stock OMLX still does not actually serve the 1-bit Bonsai model here.
- When the user asks to "set up Bonsai" or otherwise bring up the narrator on
  this box, default to `http://127.0.0.1:8002` with `--prompt-cache-size 2`.
- The canonical local model path is
  `/Users/noahlyons/.cache/huggingface/hub/models--prism-ml--bonsai-8b-mlx-1bit/snapshots/d95a01f5e78184d278e21c4cfd57ff417a60ae22`.
- The canonical model id exposed by `/v1/models` is that full snapshot path, so
  default `--narrator-model` to the same full path unless the live server says
  otherwise.
- The PRISM MLX fork used to make this work is at
  `/Users/noahlyons/dev/spoke/mlx`. The working-box provenance and bring-up
  packet live in
  `~/dev/epistaxis/metadosis/auto-grader_bonsai-working-box-packet_2026-04-08.md`.
- If Bonsai needs to be relaunched, keep it on its own port and do not try to
  fold it back into OMLX unless the user explicitly asks for that experiment.

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
