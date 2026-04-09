from __future__ import annotations

import argparse
import os
import subprocess
import sys


_ALWAYS_ON_SUITES = (
    "tests.test_project_metadata_contract",
    "tests.test_postgres_contract_bootstrap_script",
    "tests.test_db_connection_contract",
    "tests.test_db_postgres_harness_contract",
    "tests.test_contract_test_runner",
    "tests.test_template_schema_contract",
    "tests.test_generation_contract",
    "tests.test_pdf_rendering_contract",
    "test_unittest_discovery_contract",
)
_POSTGRES_SUITES = (
    "tests.test_db_postgres_smoke_contract",
    "tests.postgres_contract_bootstrap_script_smoke_contract",
    "tests.test_db_postgres_contract",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authoritative contract test suites for the repo in a "
            "stable, low-friction order."
        )
    )
    parser.add_argument(
        "--require-postgres",
        action="store_true",
        help=(
            "Require TEST_DATABASE_URL and include the Postgres schema contract "
            "suite instead of treating it as optional."
        ),
    )
    return parser.parse_args(argv)


def _get_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


def _run_suite(module_name: str) -> int:
    command = [sys.executable, "-m", "unittest", module_name, "-q"]
    result = subprocess.run(command, env=os.environ.copy(), check=False)
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])
    database_url = _get_test_database_url()

    if database_url is not None:
        stripped_database_url = database_url.strip()
        if stripped_database_url == "":
            print(
                "TEST_DATABASE_URL must be non-blank when explicitly set for the "
                "contract test runner.",
                file=sys.stderr,
            )
            return 2
        if stripped_database_url != database_url:
            print(
                "TEST_DATABASE_URL must not include leading or trailing "
                "whitespace for the contract test runner.",
                file=sys.stderr,
            )
            return 2

    if args.require_postgres and database_url is None:
        print(
            "--require-postgres needs TEST_DATABASE_URL set to an explicit "
            "disposable Postgres instance.",
            file=sys.stderr,
        )
        return 2

    for suite in _ALWAYS_ON_SUITES:
        exit_code = _run_suite(suite)
        if exit_code != 0:
            return exit_code

    if database_url is None:
        print(
            "Skipping Postgres schema contract suite because TEST_DATABASE_URL "
            "is not set.",
        )
        return 0

    for suite in _POSTGRES_SUITES:
        exit_code = _run_suite(suite)
        if exit_code != 0:
            return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
