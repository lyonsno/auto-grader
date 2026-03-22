from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_postgres_contracts.sh"


class PostgresBootstrapScriptContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
