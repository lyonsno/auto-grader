from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_postgres_contracts.sh"


class PostgresBootstrapScriptContractTests(unittest.TestCase):
    def _write_executable(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def _run_script_with_fake_postgres_tools(
        self,
        *,
        explicit_port: str | None = None,
        fail_start_port: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], str, str]:
        with tempfile.TemporaryDirectory(prefix="agpg-script.", dir="/tmp") as tmpdir:
            tmp_path = Path(tmpdir)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            pg_ctl_log = tmp_path / "pg_ctl.log"
            uv_dsn_log = tmp_path / "uv_dsn.log"
            uv_args_log = tmp_path / "uv_args.log"

            self._write_executable(
                fake_bin / "initdb",
                """#!/usr/bin/env bash
set -euo pipefail
data_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -D)
      data_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$data_dir"
: >"$data_dir/PG_VERSION"
""",
            )
            self._write_executable(
                fake_bin / "pg_ctl",
                """#!/usr/bin/env bash
set -euo pipefail
data_dir=""
operation=""
pg_options=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -D)
      data_dir="$2"
      shift 2
      ;;
    -l)
      shift 2
      ;;
    -m)
      shift 2
      ;;
    -o)
      pg_options="$2"
      shift 2
      ;;
    -w)
      shift
      ;;
    start|stop)
      operation="$1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done
port=""
if [[ "$pg_options" =~ -p[[:space:]]+([0-9]+) ]]; then
  port="${BASH_REMATCH[1]}"
fi
printf '%s|%s\\n' "$operation" "$port" >>"$FAKE_PG_CTL_LOG"
if [[ "$operation" == "start" ]]; then
  if [[ -n "${FAKE_PG_CTL_FAIL_PORT:-}" && "$port" == "$FAKE_PG_CTL_FAIL_PORT" ]]; then
    echo "port $port already in use" >&2
    exit 1
  fi
  : >"$data_dir/postmaster.pid"
fi
if [[ "$operation" == "stop" ]]; then
  rm -f "$data_dir/postmaster.pid"
fi
""",
            )
            self._write_executable(
                fake_bin / "uv",
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s' "${TEST_DATABASE_URL:-}" >"$FAKE_UV_DSN_LOG"
printf '%s' "$*" >"$FAKE_UV_ARGS_LOG"
""",
            )

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            env["TMPDIR"] = tmpdir
            env["FAKE_PG_CTL_LOG"] = str(pg_ctl_log)
            env["FAKE_UV_DSN_LOG"] = str(uv_dsn_log)
            env["FAKE_UV_ARGS_LOG"] = str(uv_args_log)
            env.pop("TEST_DATABASE_URL", None)
            if explicit_port is None:
                env.pop("AUTO_GRADER_POSTGRES_PORT", None)
            else:
                env["AUTO_GRADER_POSTGRES_PORT"] = explicit_port
            if fail_start_port is None:
                env.pop("FAKE_PG_CTL_FAIL_PORT", None)
            else:
                env["FAKE_PG_CTL_FAIL_PORT"] = fail_start_port

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            pg_ctl_entries = (
                pg_ctl_log.read_text(encoding="utf-8").splitlines()
                if pg_ctl_log.exists()
                else []
            )
            uv_dsn = uv_dsn_log.read_text(encoding="utf-8") if uv_dsn_log.exists() else ""
            uv_args = uv_args_log.read_text(encoding="utf-8") if uv_args_log.exists() else ""
            return result, pg_ctl_entries, uv_dsn, uv_args

    def test_repo_provides_one_command_bootstrap_script_for_db_backed_contracts(
        self,
    ) -> None:
        self.assertTrue(
            SCRIPT_PATH.exists(),
            "Add a repo-local one-command script for bootstrapping a disposable "
            "native Postgres instance and running the authoritative DB-backed "
            "contract suite.",
        )
        script_text = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertRegex(
            script_text,
            r"(?m)^#!",
            "The disposable Postgres bootstrap entrypoint should be directly "
            "invokable from the shell with a script shebang.",
        )
        self.assertRegex(
            script_text,
            r"(?is)\binitdb\b.*(?:-E\s*UTF8|--encoding(?:=|\s+)UTF8)",
            "The bootstrap script should pin UTF-8 cluster encoding so the "
            "DB-backed contract suite does not false-red against SQL_ASCII.",
        )
        self.assertRegex(
            script_text,
            r"(?is)\bpg_ctl\b.*\bstart\b",
            "The bootstrap script should start a disposable local Postgres server "
            "instead of assuming one is already running.",
        )
        self.assertRegex(
            script_text,
            r"""(?is)-o\s+"[^"]*-k\s+\$socket_dir\s+-p\s+\$[A-Za-z_][A-Za-z0-9_]*\s+-h\s+''""",
            "The bootstrap script should bind Postgres to a per-run Unix socket "
            "directory with TCP disabled, so each disposable run stays isolated "
            "from unrelated local TCP listeners.",
        )
        self.assertRegex(
            script_text,
            r"(?is)\bTEST_DATABASE_URL\b",
            "The bootstrap script should set TEST_DATABASE_URL for the "
            "authoritative DB-backed contract run.",
        )
        self.assertRegex(
            script_text,
            r"(?is)auto_grader\.contract_test_runner.*--require-postgres",
            "The bootstrap script should invoke the authoritative DB-backed "
            "contract runner. TEST_DATABASE_URL may be exported separately from "
            "the runner invocation.",
        )
        self.assertRegex(
            script_text,
            r"(?is)\bpg_ctl\b.*\bstop\b",
            "The bootstrap script should stop the disposable Postgres server when "
            "the contract run completes or fails.",
        )

    def test_bootstrap_script_honors_explicit_port_override(self) -> None:
        result, pg_ctl_entries, uv_dsn, uv_args = self._run_script_with_fake_postgres_tools(
            explicit_port="56543"
        )

        self.assertEqual(
            result.returncode,
            0,
            "The bootstrap script should accept an explicit AUTO_GRADER_POSTGRES_PORT "
            "override for callers that need a known disposable port. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn(
            "start|56543",
            pg_ctl_entries,
            "When AUTO_GRADER_POSTGRES_PORT is set, the bootstrap script should "
            "pass that explicit port through to pg_ctl start.",
        )
        self.assertIn(
            "port=56543",
            uv_dsn,
            "The TEST_DATABASE_URL exported to the contract runner should include "
            "the caller-specified Postgres port.",
        )
        self.assertEqual(
            uv_args,
            "run python -m auto_grader.contract_test_runner --require-postgres",
            "The bootstrap script should invoke the authoritative DB-backed "
            "contract runner via `uv run` when uv is available.",
        )

    def test_bootstrap_script_surfaces_startup_failure_instead_of_retrying_default_port(
        self,
    ) -> None:
        result, _pg_ctl_entries, uv_dsn, _uv_args = self._run_script_with_fake_postgres_tools(
            fail_start_port="55432"
        )

        self.assertNotEqual(
            result.returncode,
            0,
            "When pg_ctl start fails, the bootstrap script should surface that "
            "startup failure directly instead of retrying through a synthetic "
            "replacement-port path. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertEqual(
            uv_dsn,
            "",
            "A failed bootstrap start should not invoke the contract runner with "
            "a synthesized TEST_DATABASE_URL after pg_ctl has already failed.",
        )
        self.assertIn(
            "port 55432 already in use",
            result.stderr,
            "The bootstrap script should preserve the pg_ctl startup failure so "
            "local debugging stays grounded in the real error.",
        )


if __name__ == "__main__":
    unittest.main()
