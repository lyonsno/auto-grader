from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import importlib
import os
import unittest
from unittest import mock


@contextmanager
def _patched_env(**updates: str | None):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_runner_module(test_case: unittest.TestCase):
    try:
        return importlib.import_module("auto_grader.contract_test_runner")
    except ModuleNotFoundError as exc:
        test_case.fail(
            "Add `auto_grader.contract_test_runner` so the authoritative "
            "contract suites can be run from one repo-local command."
        )


class ContractTestRunnerTests(unittest.TestCase):
    def test_default_run_executes_always_on_contract_suites(self) -> None:
        runner = _load_runner_module(self)
        commands: list[list[str]] = []

        def fake_run(command, *, env=None, check=False):
            commands.append(list(command))
            return SimpleNamespace(returncode=0)

        stdout = StringIO()
        with _patched_env(TEST_DATABASE_URL=None), redirect_stdout(stdout), mock.patch.object(
            runner.subprocess,
            "run",
            side_effect=fake_run,
        ):
            exit_code = runner.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [command[-2] for command in commands],
            [
                "tests.test_project_metadata_contract",
                "tests.test_db_connection_contract",
            ],
            "Default contract-runner invocation should execute the always-on "
            "metadata and connection suites in a fixed, repo-local order.",
        )
        self.assertNotIn(
            "tests.test_db_postgres_contract",
            [command[-2] for command in commands],
            "Postgres schema tests should not run implicitly when "
            "TEST_DATABASE_URL is unset.",
        )
        self.assertRegex(
            stdout.getvalue(),
            r"(?is)skipp.*TEST_DATABASE_URL",
            "Default runner output should explain why the Postgres schema suite "
            "was not executed.",
        )

    def test_runner_includes_postgres_schema_suite_when_database_url_is_set(
        self,
    ) -> None:
        runner = _load_runner_module(self)
        commands: list[list[str]] = []

        def fake_run(command, *, env=None, check=False):
            commands.append(list(command))
            return SimpleNamespace(returncode=0)

        with _patched_env(TEST_DATABASE_URL="postgresql:///postgres"), mock.patch.object(
            runner.subprocess,
            "run",
            side_effect=fake_run,
        ):
            exit_code = runner.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [command[-2] for command in commands],
            [
                "tests.test_project_metadata_contract",
                "tests.test_db_connection_contract",
                "tests.test_db_postgres_contract",
            ],
            "Setting TEST_DATABASE_URL should cause the repo-local runner to "
            "include the authoritative Postgres schema contract suite.",
        )

    def test_runner_require_postgres_fails_without_database_url(self) -> None:
        runner = _load_runner_module(self)
        stderr = StringIO()

        with _patched_env(TEST_DATABASE_URL=None), redirect_stderr(stderr), mock.patch.object(
            runner.subprocess,
            "run",
        ) as run_mock:
            exit_code = runner.main(["--require-postgres"])

        self.assertNotEqual(exit_code, 0)
        run_mock.assert_not_called()
        self.assertRegex(
            stderr.getvalue(),
            r"(?is)TEST_DATABASE_URL",
            "The runner should fail fast with a clear message when Postgres is "
            "explicitly required but no disposable database URL was provided.",
        )

    def test_runner_rejects_blank_database_url_as_misconfiguration(self) -> None:
        runner = _load_runner_module(self)
        stderr = StringIO()

        for database_url in ("", "   "):
            with self.subTest(database_url=database_url):
                with _patched_env(
                    TEST_DATABASE_URL=database_url
                ), redirect_stderr(stderr), mock.patch.object(
                    runner.subprocess,
                    "run",
                ) as run_mock:
                    exit_code = runner.main([])

                self.assertNotEqual(exit_code, 0)
                run_mock.assert_not_called()
                self.assertRegex(
                    stderr.getvalue(),
                    r"(?is)TEST_DATABASE_URL.*(?:blank|empty|whitespace)",
                    "Blank TEST_DATABASE_URL values must fail as runner "
                    "misconfiguration instead of being treated as absent.",
                )
                stderr.seek(0)
                stderr.truncate(0)


if __name__ == "__main__":
    unittest.main()
