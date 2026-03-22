from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_postgres_contracts.sh"


def _bootstrap_smoke_env(tmpdir: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TEST_DATABASE_URL", None)
    env.pop("AUTO_GRADER_POSTGRES_PORT", None)
    env["TMPDIR"] = tmpdir
    env["UV_CACHE_DIR"] = "/tmp/uv-cache"
    env["AUTO_GRADER_SKIP_BOOTSTRAP_SCRIPT_SMOKE_PROBE"] = "1"
    return env


class PostgresBootstrapScriptSmokeContractTests(unittest.TestCase):
    def test_bootstrap_smoke_probe_sanitizes_parent_env_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TEST_DATABASE_URL": "postgresql:///leaked-parent-dsn",
                "AUTO_GRADER_POSTGRES_PORT": "6543",
                "UV_CACHE_DIR": "/tmp/leaked-parent-cache",
            },
            clear=False,
        ):
            env = _bootstrap_smoke_env("/tmp/agpg-hermetic")

        self.assertNotIn(
            "TEST_DATABASE_URL",
            env,
            "The bootstrap smoke probe should not inherit a parent test database "
            "URL when it is verifying the script's own bootstrap flow.",
        )
        self.assertNotIn(
            "AUTO_GRADER_POSTGRES_PORT",
            env,
            "The bootstrap smoke probe should ignore parent port overrides so the "
            "script exercises its own default disposable-port behavior.",
        )
        self.assertEqual(
            env["UV_CACHE_DIR"],
            "/tmp/uv-cache",
            "The bootstrap smoke probe should force a deterministic UV cache "
            "location instead of inheriting a caller-specific override.",
        )
        self.assertEqual(env["TMPDIR"], "/tmp/agpg-hermetic")
        self.assertEqual(env["AUTO_GRADER_SKIP_BOOTSTRAP_SCRIPT_SMOKE_PROBE"], "1")

    def test_bootstrap_script_runs_db_backed_contract_flow_and_cleans_up(self) -> None:
        if os.environ.get("AUTO_GRADER_SKIP_BOOTSTRAP_SCRIPT_SMOKE_PROBE") == "1":
            self.skipTest(
                "Skip recursive bootstrap-script smoke probe inside the spawned "
                "contract-runner subprocess."
            )
        if shutil.which("initdb") is None or shutil.which("pg_ctl") is None:
            self.skipTest(
                "Native Postgres tools are required to behaviorally smoke-test "
                "the bootstrap script."
            )

        with tempfile.TemporaryDirectory(prefix="agpg.", dir="/tmp") as tmpdir:
            env = _bootstrap_smoke_env(tmpdir)

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            combined_output = result.stdout + result.stderr

            self.assertEqual(
                result.returncode,
                0,
                "The one-command bootstrap script should successfully bring up "
                "a disposable Postgres instance, run the DB-backed contract "
                "runner, and exit cleanly. "
                f"stdout={result.stdout!r} stderr={result.stderr!r}",
            )
            self.assertNotRegex(
                combined_output,
                r"(?is)Skipping Postgres schema contract suite because TEST_DATABASE_URL is not set.",
                "The bootstrap script should set TEST_DATABASE_URL and run the "
                "DB-backed contract path instead of falling through to the "
                "runner's non-DB skip behavior.",
            )
            self.assertGreaterEqual(
                len(re.findall(r"(?m)^Ran \d+ tests? in ", combined_output)),
                8,
                "The bootstrap script should drive the full contract runner, "
                "including the DB-backed smoke and schema suites, rather than "
                "only a partial local test slice.",
            )
            self.assertEqual(
                list(Path(tmpdir).glob("auto-grader-pg.*")),
                [],
                "The bootstrap script should clean up its disposable Postgres "
                "cluster directory after the contract run completes.",
            )


if __name__ == "__main__":
    unittest.main()
