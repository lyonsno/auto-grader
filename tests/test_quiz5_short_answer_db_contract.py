"""DB-backed contracts for Quiz #5 short-answer variant registration."""

from __future__ import annotations

import os
from pathlib import Path
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


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class Quiz5ShortAnswerDbContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest("Set TEST_DATABASE_URL to run Quiz #5 short-answer DB contract tests.")
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url, label="TEST_DATABASE_URL"
            )
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        if psycopg is None:
            raise AssertionError("Quiz #5 short-answer DB contract tests require psycopg.")

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_quiz5_short_answer_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema_name))
                )
            self._schema_created = True
            self.connection = psycopg.connect(
                self.database_url, autocommit=True, row_factory=dict_row
            )
            self.connection.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self.schema_name)
                )
            )
        except Exception as exc:
            self._cleanup_schema()
            raise AssertionError(f"Setup failed: {exc}") from exc

        initialize_schema(self.connection)

    def tearDown(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self._cleanup_schema()

    def _cleanup_schema(self) -> None:
        if not self._schema_created:
            return
        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(self.schema_name)
                    )
                )
        except Exception:
            pass

    def _family(self):
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        return reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])

    def _packets(self):
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        family = self._family()
        return {
            "A": build_quiz5_short_answer_variant_packet(
                family, variant_id="A", opaque_instance_code="QUIZ5-A"
            ),
            "B": build_quiz5_short_answer_variant_packet(
                family, variant_id="B", opaque_instance_code="QUIZ5-B"
            ),
            "C": build_quiz5_short_answer_variant_packet(
                family, variant_id="C", opaque_instance_code="QUIZ5-C"
            ),
        }, family

    def test_register_variants_creates_definition_instances_and_pages(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            register_quiz5_short_answer_variants,
        )

        packets, family = self._packets()
        result = register_quiz5_short_answer_variants(
            family=family,
            packets=packets,
            connection=self.connection,
        )

        self.assertIn("template_version_id", result)
        self.assertIn("exam_definition_id", result)
        self.assertEqual(set(result["variants"]), {"A", "B", "C"})

        exam_instances = self.connection.execute(
            "SELECT opaque_instance_code FROM exam_instances ORDER BY opaque_instance_code"
        ).fetchall()
        self.assertEqual(
            [row["opaque_instance_code"] for row in exam_instances],
            ["QUIZ5-A", "QUIZ5-B", "QUIZ5-C"],
        )

        exam_pages = self.connection.execute(
            """
            SELECT fallback_page_code
            FROM exam_pages
            ORDER BY fallback_page_code
            """
        ).fetchall()
        self.assertEqual(
            [row["fallback_page_code"] for row in exam_pages],
            [
                "QUIZ5-A-p1",
                "QUIZ5-A-p2",
                "QUIZ5-B-p1",
                "QUIZ5-B-p2",
                "QUIZ5-C-p1",
                "QUIZ5-C-p2",
            ],
        )

    def test_register_variants_is_idempotent(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            register_quiz5_short_answer_variants,
        )

        packets, family = self._packets()
        first = register_quiz5_short_answer_variants(
            family=family,
            packets=packets,
            connection=self.connection,
        )
        second = register_quiz5_short_answer_variants(
            family=family,
            packets=packets,
            connection=self.connection,
        )

        self.assertEqual(first["exam_definition_id"], second["exam_definition_id"])
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM template_versions").fetchone()["n"],
            1,
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM exam_definitions").fetchone()["n"],
            1,
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM exam_instances").fetchone()["n"],
            3,
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM exam_pages").fetchone()["n"],
            6,
        )


if __name__ == "__main__":
    unittest.main()
