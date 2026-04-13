from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import importlib
import os
import subprocess
import sys
import tempfile
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
                "tests.test_postgres_contract_bootstrap_script",
                "tests.test_db_connection_contract",
                "tests.test_db_postgres_harness_contract",
                "tests.test_contract_test_runner",
                "tests.test_template_schema_contract",
                "tests.test_eval_harness_contract",
                "tests.test_shimmer_phases",
                "tests.test_generation_contract",
                "tests.test_pdf_rendering_contract",
                "tests.test_scan_readback_contract",
                "tests.test_scan_registration_contract",
                "tests.test_bubble_interpretation_contract",
                "tests.test_mc_scoring_contract",
                "tests.test_mc_page_extraction_contract",
                "tests.test_mc_scan_ingest_contract",
                "tests.test_mc_scan_session_persist_contract",
                "tests.test_mc_opencv_demo_contract",
                "tests.test_mc_review_override_contract",
                "tests.test_mark_profile_smoke_contract",
                "tests.test_paper_calibration_packet_contract",
                "tests.test_paper_threshold_packet_contract",
                "tests.test_generated_exam_demo_contract",
                "tests.test_mc_workflow_gui_contract",
                "test_unittest_discovery_contract",
            ],
            "Default contract-runner invocation should execute the always-on "
            "metadata, bootstrap-script, connection, Postgres harness, runner, "
            "template schema, eval harness, shimmer phases, generation, PDF "
            "rendering, scan readback, scan registration, bubble "
            "interpretation, MC scoring, matched-page extraction, scan-level "
            "ingest packaging, scan-session persistence, the prototype demo "
            "runner, review override, the synthetic mark-profile smoke "
                "contract, the professor-workflow GUI contract, and the "
                "discovery guardrail suites in a fixed, "
                "repo-local order.",
        )
        self.assertNotIn(
            "tests.test_db_postgres_contract",
            [command[-2] for command in commands],
            "Postgres schema tests should not run implicitly when "
            "TEST_DATABASE_URL is unset.",
        )
        self.assertNotIn(
            "tests.postgres_contract_bootstrap_script_smoke_contract",
            [command[-2] for command in commands],
            "The DB-backed bootstrap smoke probe should stay out of the default "
            "sandbox-safe runner path until Postgres is explicitly configured.",
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
                "tests.test_postgres_contract_bootstrap_script",
                "tests.test_db_connection_contract",
                "tests.test_db_postgres_harness_contract",
                "tests.test_contract_test_runner",
                "tests.test_template_schema_contract",
                "tests.test_eval_harness_contract",
                "tests.test_shimmer_phases",
                "tests.test_generation_contract",
                "tests.test_pdf_rendering_contract",
                "tests.test_scan_readback_contract",
                "tests.test_scan_registration_contract",
                "tests.test_bubble_interpretation_contract",
                "tests.test_mc_scoring_contract",
                "tests.test_mc_page_extraction_contract",
                "tests.test_mc_scan_ingest_contract",
                "tests.test_mc_scan_session_persist_contract",
                "tests.test_mc_opencv_demo_contract",
                "tests.test_mc_review_override_contract",
                "tests.test_mark_profile_smoke_contract",
                "tests.test_paper_calibration_packet_contract",
                "tests.test_paper_threshold_packet_contract",
                "tests.test_generated_exam_demo_contract",
                "tests.test_mc_workflow_gui_contract",
                "test_unittest_discovery_contract",
                "tests.test_db_postgres_smoke_contract",
                "tests.postgres_contract_bootstrap_script_smoke_contract",
                "tests.test_db_postgres_contract",
                "tests.test_mc_scan_db_contract",
                "tests.test_mc_review_resolution_db_contract",
                "tests.test_mc_final_results_db_contract",
                "tests.test_mc_db_round_trip_contract",
                "tests.test_mc_results_demo_export_contract",
                "tests.test_mc_workflow_contract",
            ],
            "Setting TEST_DATABASE_URL should cause the repo-local runner to "
            "run the fail-fast schema smoke suite, the one-command bootstrap "
            "script smoke suite, and then the full authoritative schema, scan "
            "DB persistence, review-resolution DB, current-final-results DB, "
            "DB-backed MC round-trip, MC demo-export, and professor-workflow "
            "DB contract suites alongside the always-on guardrail suites.",
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

    def test_runner_rejects_database_url_with_surrounding_whitespace(self) -> None:
        runner = _load_runner_module(self)
        stderr = StringIO()

        for database_url in (
            " postgresql:///postgres ",
            "\npostgresql:///postgres",
            "postgresql:///postgres\n",
            "\tpostgresql:///postgres\t",
        ):
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
                    r"(?is)TEST_DATABASE_URL.*(?:leading|trailing|whitespace|trim)",
                    "TEST_DATABASE_URL values with surrounding whitespace must "
                    "fail fast instead of activating the Postgres contract path.",
                )
                stderr.seek(0)
                stderr.truncate(0)

    def test_runner_executes_always_on_suites_from_repo_root_when_invoked_elsewhere(
        self,
    ) -> None:
        if os.environ.get("AUTO_GRADER_OUT_OF_TREE_RUNNER_PROBE") == "1":
            self.skipTest(
                "Skip recursive out-of-tree runner probe inside the spawned "
                "contract-runner subprocess."
            )

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            repo_root
            if not existing_pythonpath
            else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        )
        env.pop("TEST_DATABASE_URL", None)
        env["AUTO_GRADER_OUT_OF_TREE_RUNNER_PROBE"] = "1"
        env["AUTO_GRADER_SKIP_UPPERCASE_HARNESS_NORMALIZATION_PROBES"] = "1"

        with tempfile.TemporaryDirectory() as tempdir:
            result = subprocess.run(
                [sys.executable, "-m", "auto_grader.contract_test_runner"],
                cwd=tempdir,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            result.returncode,
            0,
            "Module entrypoint runs from outside the repo root should still "
            "succeed when auto_grader is importable. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertRegex(
            result.stdout,
            r"(?is)Skipping Postgres schema contract suite because TEST_DATABASE_URL is not set.",
            "A successful out-of-tree runner invocation should still reach the "
            "normal Postgres-suite skip message when TEST_DATABASE_URL is unset.",
        )


if __name__ == "__main__":
    unittest.main()
