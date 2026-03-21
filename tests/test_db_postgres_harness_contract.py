from __future__ import annotations

import importlib
import unittest
from unittest import mock


postgres_contract = importlib.import_module("tests.test_db_postgres_contract")


class PostgresHarnessContractTests(unittest.TestCase):
    def test_schema_suite_skips_when_test_database_url_is_not_configured(self) -> None:
        with mock.patch.object(
            postgres_contract,
            "_postgres_test_database_url",
            return_value=None,
        ), mock.patch.object(postgres_contract, "psycopg", object()):
            with self.assertRaises(unittest.SkipTest):
                postgres_contract.PostgresDatabaseContractTests.setUpClass()

    def test_schema_suite_fails_when_test_database_url_is_blank(self) -> None:
        for database_url in ("", "   "):
            with self.subTest(database_url=database_url):
                fake_psycopg = mock.Mock()
                with mock.patch.object(
                    postgres_contract,
                    "_postgres_test_database_url",
                    return_value=database_url,
                ), mock.patch.object(postgres_contract, "psycopg", fake_psycopg):
                    try:
                        postgres_contract.PostgresDatabaseContractTests.setUpClass()
                    except unittest.SkipTest as exc:
                        self.fail(
                            "Blank TEST_DATABASE_URL must fail as schema-harness "
                            f"misconfiguration, not skip: {exc}"
                        )
                    except AssertionError as exc:
                        fake_psycopg.connect.assert_not_called()
                        self.assertRegex(
                            str(exc),
                            r"(?is)TEST_DATABASE_URL.*(?:blank|empty|whitespace)",
                        )
                    else:
                        self.fail(
                            "Blank TEST_DATABASE_URL must raise AssertionError instead "
                            "of being treated like an unset schema harness."
                        )

    def test_schema_suite_fails_when_explicit_database_url_lacks_driver(self) -> None:
        with mock.patch.object(
            postgres_contract,
            "_postgres_test_database_url",
            return_value="postgresql:///postgres",
        ), mock.patch.object(postgres_contract, "psycopg", None):
            with self.assertRaisesRegex(AssertionError, "psycopg"):
                postgres_contract.PostgresDatabaseContractTests.setUpClass()

    def test_schema_suite_fails_when_explicit_database_url_is_unreachable(self) -> None:
        fake_psycopg = mock.Mock()
        fake_psycopg.connect.side_effect = RuntimeError("connection refused")

        with mock.patch.object(
            postgres_contract,
            "_postgres_test_database_url",
            return_value="postgresql:///postgres",
        ), mock.patch.object(postgres_contract, "psycopg", fake_psycopg):
            with self.assertRaisesRegex(
                AssertionError,
                "reachable database access and CREATE SCHEMA privilege",
            ):
                postgres_contract.PostgresDatabaseContractTests.setUpClass()
