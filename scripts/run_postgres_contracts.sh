#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/auto-grader-pg.XXXXXX")"
data_dir="$tmp_root/data"
socket_dir="$tmp_root/socket"
log_path="$tmp_root/postgres.log"
port="${AUTO_GRADER_POSTGRES_PORT:-55432}"

cleanup() {
  if [[ -f "$data_dir/postmaster.pid" ]]; then
    pg_ctl -D "$data_dir" -m fast stop >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp_root"
}

trap cleanup EXIT

mkdir -p "$socket_dir"

initdb -D "$data_dir" -U postgres -A trust --no-locale -E UTF8 >/dev/null
pg_ctl -D "$data_dir" \
  -l "$log_path" \
  -w \
  start \
  -o "-k $socket_dir -p $port -h ''"

export TEST_DATABASE_URL="postgresql://postgres@/postgres?host=${socket_dir}&port=${port}"

cd "$repo_root"

if command -v uv >/dev/null 2>&1; then
  uv run python -m auto_grader.contract_test_runner --require-postgres
  exit $?
fi

python3 -m auto_grader.contract_test_runner --require-postgres
