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

    def test_schema_suite_fails_when_explicit_database_url_lacks_driver(self) -> None:
        with mock.patch.object(
            postgres_contract,
            "_postgres_test_database_url",
            return_value="postgresql:///postgres",
        ), mock.patch.object(postgres_contract, "psycopg", None):
            with self.assertRaisesRegex(AssertionError, "psycopg"):
                postgres_contract.PostgresDatabaseContractTests.setUpClass()
