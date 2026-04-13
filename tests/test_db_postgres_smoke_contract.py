from __future__ import annotations

from contextlib import suppress
import os
import unittest
import uuid

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:
    psycopg = None
    sql = None
    dict_row = None

from auto_grader import db as db_module
from auto_grader.db import initialize_schema

_EXPECTED_WORKFLOW_TABLES = {
    "students",
    "template_versions",
    "exam_definitions",
    "exam_instances",
    "exam_pages",
    "scan_artifacts",
    "grade_records",
    "audit_events",
    "mc_review_resolutions",
}


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


class PostgresSmokeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run the Postgres smoke contract suite "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the Postgres smoke contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "Postgres smoke contract tests require psycopg in the active "
                "environment. Run `uv sync` first."
            )

    def setUp(self) -> None:
        self.schema_name = f"ag_smoke_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_connection:
                admin_connection.execute(
                    sql.SQL("CREATE SCHEMA {}").format(
                        sql.Identifier(self.schema_name)
                    )
                )
            self._schema_created = True
            self.connection = psycopg.connect(
                self.database_url,
                autocommit=True,
                row_factory=dict_row,
            )
            self.connection.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self.schema_name)
                )
            )
        except Exception as exc:
            self._cleanup_schema()
            raise AssertionError(
                "Postgres smoke contract test setup failed unexpectedly at "
                f"{self.database_url!r}: {exc}"
            ) from exc

    def tearDown(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self._cleanup_schema()

    def test_initialize_schema_creates_all_workflow_tables_in_real_postgres(self) -> None:
        initialize_schema(self.connection)

        actual_tables = self._list_table_names()
        missing_tables = _EXPECTED_WORKFLOW_TABLES - actual_tables
        self.assertFalse(
            missing_tables,
            "The DB-backed smoke contract must fail fast when initialize_schema() "
            "does not create the expected workflow tables in a real disposable "
            f"Postgres schema. Missing={sorted(missing_tables)} "
            f"actual={sorted(actual_tables)}",
        )

    def _cleanup_schema(self) -> None:
        if not getattr(self, "_schema_created", False):
            return
        with suppress(Exception):
            with psycopg.connect(self.database_url, autocommit=True) as admin_connection:
                admin_connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.schema_name)
                    )
                )
        self._schema_created = False

    def _list_table_names(self) -> set[str]:
        rows = self.connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
            """
        ).fetchall()
        return {str(row["table_name"]) for row in rows}


if __name__ == "__main__":
    unittest.main()
