from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import os
import unittest
import uuid

try:
    import psycopg
    from psycopg import errors, sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:
    psycopg = None
    errors = None
    sql = None
    dict_row = None

from auto_grader.db import initialize_schema

_UNSET = object()


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


class PostgresDatabaseContractTests(unittest.TestCase):
    """Fail-first Postgres-backed contract tests for the first schema slice."""

    @classmethod
    def setUpClass(cls) -> None:
        if psycopg is None:
            raise AssertionError(
                "Postgres contract tests require psycopg in the active environment. "
                "Run `uv sync` and `uv run python -m unittest "
                "tests.test_db_postgres_contract -q`."
            )
        cls.database_url = _postgres_test_database_url()
        if not cls.database_url:
            raise AssertionError(
                "Set TEST_DATABASE_URL to run Postgres schema contract tests "
                "against an explicit disposable Postgres instance."
            )
        probe_schema = f"ag_contract_probe_{uuid.uuid4().hex}"
        try:
            with psycopg.connect(cls.database_url, autocommit=True) as admin_connection:
                admin_connection.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(probe_schema))
                )
                admin_connection.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(probe_schema)
                    )
                )
        except Exception as exc:
            raise AssertionError(
                "Postgres contract tests require reachable database access and "
                f"CREATE SCHEMA privilege at {cls.database_url!r}: {exc}"
            ) from exc

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_contract_{uuid.uuid4().hex}"
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
                "Postgres contract test setup failed unexpectedly at "
                f"{self.database_url!r}: {exc}"
            ) from exc

        initialize_schema(self.connection)

    def tearDown(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self._cleanup_schema()

    def test_initialize_schema_creates_workflow_tables_in_current_schema(self) -> None:
        actual_tables = self._list_table_names()
        expected_tables = {
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
            "scan_artifacts",
            "grade_records",
            "audit_events",
        }

        missing_tables = expected_tables - actual_tables
        self.assertFalse(
            missing_tables,
            f"Missing Postgres workflow tables in schema {self.schema_name}: "
            f"{sorted(missing_tables)}",
        )

    def test_students_require_unique_nonblank_student_keys(self) -> None:
        self._require_tables("students")

        self._insert_student("student-001")

        with self.assertRaises(errors.UniqueViolation):
            self._insert_student("student-001")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_student(None)

        with self.assertRaises(errors.CheckViolation):
            self._insert_student("")

        with self.assertRaises(errors.CheckViolation):
            self._insert_student("   ")

    def test_template_versions_require_unique_slug_version_pairs(self) -> None:
        self._require_tables("template_versions")

        self._insert_template_version(slug="stoich-q1", version=1)

        with self.assertRaises(errors.UniqueViolation):
            self._insert_template_version(slug="stoich-q1", version=1)

    def test_exam_definitions_require_existing_template_versions_and_unique_slug_versions(
        self,
    ) -> None:
        self._require_tables("template_versions", "exam_definitions")

        template_version_id = self._insert_template_version(
            slug="chem101-midterm-template",
            version=1,
        )
        self._insert_exam_definition(
            slug="chem101-midterm",
            version=1,
            template_version_id=template_version_id,
        )

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_definition(
                slug="chem101-midterm",
                version=1,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.ForeignKeyViolation):
            self._insert_exam_definition(
                slug="chem101-final",
                version=1,
                template_version_id=999_001,
            )

    def test_exam_definitions_do_not_repoint_to_missing_template_versions(
        self,
    ) -> None:
        self._require_tables("template_versions", "exam_definitions")

        template_version_id = self._insert_template_version(
            slug="chem101-update-guard-template",
            version=1,
        )
        self._insert_exam_definition(
            slug="chem101-update-guard",
            version=1,
            template_version_id=template_version_id,
        )

        self._assert_row_is_immutable(
            table_name="exam_definitions",
            key_columns={"slug": "chem101-update-guard", "version": 1},
            assignments={"template_version_id": 999_105},
            expected_row={
                "slug": "chem101-update-guard",
                "version": 1,
                "title": "Chem 101 Midterm",
                "template_version_id": template_version_id,
            },
        )

    def test_exam_instances_require_existing_student_and_exam_definition(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition(slug="chem101-midterm")

        with self.assertRaises(errors.ForeignKeyViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=999_001,
                attempt_number=1,
                opaque_instance_code="inst-missing-student",
            )

        with self.assertRaises(errors.ForeignKeyViolation):
            self._insert_exam_instance_record(
                exam_definition_id=999_002,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="inst-missing-exam",
            )

    def test_exam_instances_reject_updates_to_missing_student_or_exam_definition(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        student_id = self._insert_student("student-update-guard")
        exam_definition_id = self._insert_exam_definition(slug="chem101-update-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-update-guard",
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE exam_instances
                SET student_id = %s
                WHERE opaque_instance_code = %s
                """,
                (999_101, "inst-update-guard"),
            )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE exam_instances
                SET exam_definition_id = %s
                WHERE opaque_instance_code = %s
                """,
                (999_102, "inst-update-guard"),
            )

        row = self.connection.execute(
            """
            SELECT student_id, exam_definition_id
            FROM exam_instances
            WHERE opaque_instance_code = %s
            """,
            ("inst-update-guard",),
        ).fetchone()
        self.assertEqual(row["student_id"], student_id)
        self.assertEqual(row["exam_definition_id"], exam_definition_id)

    def test_exam_instances_require_unique_attempts_and_opaque_codes(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        first_student_id = self._insert_student("student-001")
        second_student_id = self._insert_student("student-002")
        exam_definition_id = self._insert_exam_definition(slug="chem101-midterm")
        other_exam_definition_id = self._insert_exam_definition(slug="chem101-final")

        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="opaque-001",
        )
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=first_student_id,
            attempt_number=2,
            opaque_instance_code="opaque-002",
        )
        self._insert_exam_instance_record(
            exam_definition_id=other_exam_definition_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="opaque-003",
        )

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=first_student_id,
                attempt_number=1,
                opaque_instance_code="opaque-002",
            )

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_instance_record(
                exam_definition_id=other_exam_definition_id,
                student_id=second_student_id,
                attempt_number=1,
                opaque_instance_code="opaque-001",
            )

    def test_exam_instances_require_nonblank_opaque_code_and_positive_attempt_number(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition(slug="chem101-midterm")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="   ",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=0,
                opaque_instance_code="inst-zero-attempt",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=-1,
                opaque_instance_code="inst-negative-attempt",
            )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=None,
                opaque_instance_code="inst-null-attempt",
            )

    def test_students_cannot_be_deleted_while_exam_instances_reference_them(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        student_id = self._insert_student("student-delete-guard")
        exam_definition_id = self._insert_exam_definition(slug="chem101-delete-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-student",
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                "DELETE FROM students WHERE student_key = %s",
                ("student-delete-guard",),
            )

        self.assertEqual(
            self._count_rows("students", "student_key = %s", ("student-delete-guard",)),
            1,
        )

    def test_students_can_be_deleted_when_unreferenced(self) -> None:
        self._require_tables("students")

        self._insert_student("student-delete-no-children")

        self.connection.execute(
            "DELETE FROM students WHERE student_key = %s",
            ("student-delete-no-children",),
        )
        self.assertEqual(
            self._count_rows(
                "students",
                "student_key = %s",
                ("student-delete-no-children",),
            ),
            0,
        )

    def test_exam_definitions_cannot_be_deleted_while_exam_instances_reference_them(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        student_id = self._insert_student("student-delete-guard")
        exam_definition_id = self._insert_exam_definition(slug="chem101-delete-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-exam",
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                DELETE FROM exam_definitions
                WHERE slug = %s AND version = %s
                """,
                ("chem101-delete-guard", 1),
            )

        self.assertEqual(
            self._count_rows(
                "exam_definitions",
                "slug = %s AND version = %s",
                ("chem101-delete-guard", 1),
            ),
            1,
        )

    def test_exam_definitions_can_be_deleted_when_unreferenced(self) -> None:
        self._require_tables("template_versions", "exam_definitions")

        self._insert_exam_definition(slug="chem101-delete-no-children")

        self.connection.execute(
            """
            DELETE FROM exam_definitions
            WHERE slug = %s AND version = %s
            """,
            ("chem101-delete-no-children", 1),
        )
        self.assertEqual(
            self._count_rows(
                "exam_definitions",
                "slug = %s AND version = %s",
                ("chem101-delete-no-children", 1),
            ),
            0,
        )

    def test_exam_pages_require_existing_exam_instance_and_unique_page_numbers(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )

        exam_definition_id = self._insert_exam_definition(slug="chem101-midterm")
        first_student_id = self._insert_student("student-001")
        second_student_id = self._insert_student("student-002")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="inst-001",
        )
        other_exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=second_student_id,
            attempt_number=1,
            opaque_instance_code="inst-002",
        )

        self._insert_exam_page(exam_instance_id, 1, "MIDTERM-P1-A1")
        self._insert_exam_page(exam_instance_id, 2, "MIDTERM-P2-A1")
        self._insert_exam_page(other_exam_instance_id, 1, "MIDTERM-P1-B1")

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_page(exam_instance_id, 1, "MIDTERM-P1-A2")

        with self.assertRaises(errors.ForeignKeyViolation):
            self._insert_exam_page(999_003, 1, "MIDTERM-P1-ORPHAN")

    def test_exam_pages_require_positive_numbers_and_nonblank_fallback_codes(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )

        exam_definition_id = self._insert_exam_definition(slug="chem101-midterm")
        student_id = self._insert_student("student-001")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-pages-001",
        )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(exam_instance_id, 0, "MIDTERM-P0-INVALID")

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(exam_instance_id, -1, "MIDTERM-PNEG-INVALID")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_page(exam_instance_id, 1, None)

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(exam_instance_id, 1, "")

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(exam_instance_id, 1, "   ")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_page(exam_instance_id, None, "MIDTERM-PNULL-INVALID")

    def test_exam_page_fallback_codes_are_unique_for_manual_recovery(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )

        exam_definition_id = self._insert_exam_definition("chem101-midterm")
        first_student_id = self._insert_student("student-001")
        second_student_id = self._insert_student("student-002")
        first_exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="inst-recovery-001",
        )
        second_exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=second_student_id,
            attempt_number=1,
            opaque_instance_code="inst-recovery-002",
        )

        self._insert_exam_page(first_exam_instance_id, 1, "RECOVER-P1-A")
        self._insert_exam_page(first_exam_instance_id, 2, "RECOVER-P2-A")

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_page(second_exam_instance_id, 1, "RECOVER-P1-A")

    def test_exam_pages_reject_updates_to_missing_exam_instances(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )

        exam_instance_id = self._insert_exam_instance(
            student_key="student-update-guard",
            exam_slug="chem101-update-guard",
            attempt_number=1,
            opaque_instance_code="inst-update-guard-pages",
        )
        self._insert_exam_page(exam_instance_id, 1, "UPDATE-GUARD-P1")

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE exam_pages
                SET exam_instance_id = %s
                WHERE fallback_page_code = %s
                """,
                (999_103, "UPDATE-GUARD-P1"),
            )

        row = self.connection.execute(
            """
            SELECT exam_instance_id
            FROM exam_pages
            WHERE fallback_page_code = %s
            """,
            ("UPDATE-GUARD-P1",),
        ).fetchone()
        self.assertEqual(row["exam_instance_id"], exam_instance_id)

    def test_scan_artifacts_enforce_status_reason_and_sha256_contract(self) -> None:
        self._require_tables("scan_artifacts")

        digest = self._sha256(9)
        self._insert_scan_artifact(
            sha256=digest,
            original_filename="scan-009.png",
            status="matched",
            failure_reason=None,
        )
        self._insert_scan_artifact(
            sha256=self._sha256(10),
            original_filename="scan-010.png",
            status="unmatched",
            failure_reason="qr_decode_failed",
        )
        self._insert_scan_artifact(
            sha256=self._sha256(11),
            original_filename="scan-011.png",
            status="ambiguous",
            failure_reason="multiple_qr_candidates",
        )

        with self.assertRaises(errors.UniqueViolation):
            self._insert_scan_artifact(
                sha256=digest,
                original_filename="scan-009-rerun.png",
                status="matched",
                failure_reason=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(12),
                original_filename="scan-012.png",
                status="bogus",
                failure_reason="status_not_in_supported_vocabulary",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(13),
                original_filename="scan-013.png",
                status="matched",
                failure_reason="should_not_exist_for_success",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(14),
                original_filename="scan-014.png",
                status="unmatched",
                failure_reason=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(15),
                original_filename="scan-015.png",
                status="ambiguous",
                failure_reason=None,
            )

    def test_scan_artifacts_require_required_fields_lowercase_status_and_sha256_shape(
        self,
    ) -> None:
        self._require_tables("scan_artifacts")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_scan_artifact(
                sha256=None,
                original_filename="scan-missing-digest.png",
                status="matched",
                failure_reason=None,
            )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(15),
                original_filename=None,
                status="matched",
                failure_reason=None,
            )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(16),
                original_filename="scan-missing-status.png",
                status=None,
                failure_reason=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(17),
                original_filename="scan-case-status.png",
                status="Matched",
                failure_reason=None,
            )

        for digest_seed, status_value in (
            (23, ""),
            (24, "   "),
        ):
            with self.subTest(status=status_value):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_scan_artifact(
                        sha256=self._sha256(digest_seed),
                        original_filename=f"scan-blank-status-{digest_seed}.png",
                        status=status_value,
                        failure_reason="status_not_in_supported_vocabulary",
                    )

        for sha256_value in ("", "   ", "abc", "f" * 63, "z" * 64):
            with self.subTest(sha256=sha256_value):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename="scan-bad-digest.png",
                        status="matched",
                        failure_reason=None,
                    )

        for original_filename in ("", "   "):
            with self.subTest(original_filename=original_filename):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_scan_artifact(
                        sha256=self._sha256(18),
                        original_filename=original_filename,
                        status="matched",
                        failure_reason=None,
                    )

        for digest_seed, status, failure_reason, filename in (
            (19, "unmatched", "", "scan-bad-reason-unmatched-empty.png"),
            (20, "unmatched", "   ", "scan-bad-reason-unmatched-blank.png"),
            (21, "ambiguous", "", "scan-bad-reason-ambiguous-empty.png"),
            (22, "ambiguous", "   ", "scan-bad-reason-ambiguous-blank.png"),
        ):
            with self.subTest(status=status, failure_reason=failure_reason):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_scan_artifact(
                        sha256=self._sha256(digest_seed),
                        original_filename=filename,
                        status=status,
                        failure_reason=failure_reason,
                    )

    def test_grade_records_require_existing_exam_instance_and_only_one_finalized_grade(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self._insert_grade_record(
                exam_instance_id=999_004,
                status="draft",
                score_points=10.0,
                max_points=20.0,
            )

        exam_definition_id = self._insert_exam_definition("chem101-midterm")
        first_student_id = self._insert_student("student-001")
        second_student_id = self._insert_student("student-002")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="inst-grade-001",
        )
        other_exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=second_student_id,
            attempt_number=1,
            opaque_instance_code="inst-grade-002",
        )

        self._insert_grade_record(exam_instance_id, "draft", 18.0, 20.0)
        self._insert_grade_record(exam_instance_id, "finalized", 18.0, 20.0)
        self._insert_grade_record(other_exam_instance_id, "finalized", 16.0, 20.0)

        with self.assertRaises(errors.UniqueViolation):
            self._insert_grade_record(exam_instance_id, "finalized", 19.0, 20.0)

    def test_grade_records_require_supported_lowercase_status_and_score_bounds(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        exam_definition_id = self._insert_exam_definition("chem101-midterm")
        student_id = self._insert_student("student-001")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-grade-bounds-001",
        )

        with self.assertRaises(errors.CheckViolation):
            self._insert_grade_record(exam_instance_id, "bogus", 10.0, 20.0)

        with self.assertRaises(errors.CheckViolation):
            self._insert_grade_record(exam_instance_id, "DRAFT", 10.0, 20.0)

        for status_value in ("", "   "):
            with self.subTest(status=status_value):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_grade_record(
                        exam_instance_id,
                        status_value,
                        10.0,
                        20.0,
                    )

        with self.assertRaises(errors.CheckViolation):
            self._insert_grade_record(exam_instance_id, "draft", -0.5, 20.0)

        with self.assertRaises(errors.CheckViolation):
            self._insert_grade_record(exam_instance_id, "draft", 20.5, 20.0)

        for max_points in (0.0, -5.0):
            with self.subTest(max_points=max_points):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_grade_record(
                        exam_instance_id,
                        "draft",
                        0.0,
                        max_points,
                    )

    def test_grade_records_require_status_and_score_fields(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        exam_definition_id = self._insert_exam_definition("chem101-midterm")
        student_id = self._insert_student("student-001")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-grade-required-001",
        )

        for status_value, score_points, max_points in (
            (None, 10.0, 20.0),
            ("draft", None, 20.0),
            ("draft", 10.0, None),
        ):
            with self.subTest(
                status=status_value,
                score_points=score_points,
                max_points=max_points,
            ):
                with self.assertRaises(errors.NotNullViolation):
                    self._insert_grade_record(
                        exam_instance_id,
                        status_value,
                        score_points,
                        max_points,
                    )

    def test_grade_records_reject_updates_to_missing_exam_instances(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        exam_instance_id = self._insert_exam_instance(
            student_key="student-update-guard",
            exam_slug="chem101-update-guard",
            attempt_number=1,
            opaque_instance_code="inst-update-guard-grades",
        )
        self._insert_grade_record(exam_instance_id, "draft", 10.0, 20.0)

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE grade_records
                SET exam_instance_id = %s
                WHERE exam_instance_id = %s
                """,
                (999_104, exam_instance_id),
            )

        self.assertEqual(
            self._count_rows(
                "grade_records",
                "exam_instance_id = %s",
                (exam_instance_id,),
            ),
            1,
        )

    def test_exam_instances_cannot_be_deleted_while_exam_pages_reference_them(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )

        exam_instance_id = self._insert_exam_instance(
            student_key="student-delete-guard",
            exam_slug="chem101-delete-guard-pages",
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-with-page",
        )
        self._insert_exam_page(
            exam_instance_id=exam_instance_id,
            page_number=1,
            fallback_page_code="DELETE-GUARD-P1",
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                "DELETE FROM exam_instances WHERE opaque_instance_code = %s",
                ("inst-delete-guard-with-page",),
            )

        self.assertEqual(
            self._count_rows(
                "exam_instances",
                "opaque_instance_code = %s",
                ("inst-delete-guard-with-page",),
            ),
            1,
        )

    def test_exam_instances_cannot_be_deleted_while_grade_records_reference_them(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        exam_instance_id = self._insert_exam_instance(
            student_key="student-delete-guard",
            exam_slug="chem101-delete-guard-grades",
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-with-grade",
        )
        self._insert_grade_record(
            exam_instance_id=exam_instance_id,
            status="draft",
            score_points=10.0,
            max_points=20.0,
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                "DELETE FROM exam_instances WHERE opaque_instance_code = %s",
                ("inst-delete-guard-with-grade",),
            )

        self.assertEqual(
            self._count_rows(
                "exam_instances",
                "opaque_instance_code = %s",
                ("inst-delete-guard-with-grade",),
            ),
            1,
        )

    def test_exam_instances_can_be_deleted_when_unreferenced(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )

        self._insert_exam_instance(
            student_key="student-delete-clean",
            exam_slug="chem101-delete-clean",
            attempt_number=1,
            opaque_instance_code="inst-delete-clean",
        )

        self.connection.execute(
            "DELETE FROM exam_instances WHERE opaque_instance_code = %s",
            ("inst-delete-clean",),
        )
        self.assertEqual(
            self._count_rows(
                "exam_instances",
                "opaque_instance_code = %s",
                ("inst-delete-clean",),
            ),
            0,
        )

    def test_audit_events_require_payload_json_and_default_recorded_timestamp(
        self,
    ) -> None:
        self._require_tables("audit_events")

        before_insert = self._database_clock_now()

        with self.assertRaises(errors.NotNullViolation):
            self._insert_audit_event(
                entity_type="scan_artifact",
                entity_id=1,
                event_type="scan_recorded",
                payload_json=None,
            )

        self._insert_audit_event(
            entity_type="scan_artifact",
            entity_id=1,
            event_type="scan_recorded",
            payload_json="{}",
        )

        row = self.connection.execute(
            "SELECT recorded_at FROM audit_events"
        ).fetchone()
        self.assertIsNotNone(
            row["recorded_at"],
            "audit_events.recorded_at should default to a present timestamp.",
        )
        recorded_at = row["recorded_at"].astimezone(timezone.utc)
        after_insert = self._database_clock_now()
        self.assertGreaterEqual(recorded_at, before_insert)
        self.assertLessEqual(recorded_at, after_insert)

    def test_audit_events_store_jsonb_payloads_and_timestamptz_values(self) -> None:
        self._require_tables("audit_events")

        explicit_recorded_at = "2026-03-07T12:34:56+00:00"
        self._insert_audit_event(
            entity_type="scan_artifact",
            entity_id=1,
            event_type="scan_recorded",
            payload_json='{"event":"scan_recorded"}',
            recorded_at=explicit_recorded_at,
        )

        row = self.connection.execute(
            """
            SELECT
                pg_typeof(payload_json)::text AS payload_type,
                pg_typeof(recorded_at)::text AS recorded_type,
                payload_json ->> 'event' AS payload_event,
                recorded_at
            FROM audit_events
            """
        ).fetchone()
        self.assertEqual(row["payload_type"], "jsonb")
        self.assertEqual(row["recorded_type"], "timestamp with time zone")
        self.assertEqual(row["payload_event"], "scan_recorded")
        self.assertEqual(
            row["recorded_at"].astimezone(timezone.utc),
            datetime(2026, 3, 7, 12, 34, 56, tzinfo=timezone.utc),
        )

    def test_audit_events_require_complete_subject_event_type_and_valid_payload_json(
        self,
    ) -> None:
        self._require_tables("audit_events")

        for entity_type, entity_id in (
            ("", 1),
            ("   ", 1),
            (None, 1),
            ("scan_artifact", None),
        ):
            with self.subTest(entity_type=entity_type, entity_id=entity_id):
                with self.assertRaises((errors.NotNullViolation, errors.CheckViolation)):
                    self._insert_audit_event(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        event_type="scan_recorded",
                        payload_json="{}",
                    )

        for event_type in (None, "", "   "):
            with self.subTest(event_type=event_type):
                with self.assertRaises((errors.NotNullViolation, errors.CheckViolation)):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type=event_type,
                        payload_json="{}",
                    )

        for payload_json in ("not-json", "{"):
            with self.subTest(payload_json=payload_json):
                with self.assertRaises(errors.InvalidTextRepresentation):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json=payload_json,
                    )

    def test_audit_events_reject_blank_payload_json_strings(self) -> None:
        self._require_tables("audit_events")

        for payload_json in ("", "   "):
            with self.subTest(payload_json=payload_json):
                with self.assertRaises(errors.InvalidTextRepresentation):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json=payload_json,
                    )

    def test_audit_events_require_explicit_recorded_timestamp_when_supplied(
        self,
    ) -> None:
        self._require_tables("audit_events")

        for recorded_at, expected_exception in (
            (None, errors.NotNullViolation),
            ("", errors.InvalidDatetimeFormat),
            ("   ", errors.InvalidDatetimeFormat),
        ):
            with self.subTest(recorded_at=recorded_at):
                with self.assertRaises(expected_exception):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json="{}",
                        recorded_at=recorded_at,
                    )

    def test_audit_events_reject_invalid_explicit_recorded_timestamps(self) -> None:
        self._require_tables("audit_events")

        for recorded_at, expected_exception in (
            ("not-a-timestamp", errors.InvalidDatetimeFormat),
            ("2026-99-99 25:61:00", errors.DatetimeFieldOverflow),
        ):
            with self.subTest(recorded_at=recorded_at):
                with self.assertRaises(expected_exception):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json="{}",
                        recorded_at=recorded_at,
                    )

    def test_template_versions_preserve_exact_source_and_are_immutable(self) -> None:
        self._require_tables("template_versions")

        source_yaml = (
            "slug: stoichiometry-q8\n"
            "version: 8\n"
            "prompt: Balance the combustion reaction."
        )
        self._insert_template_version(
            slug="stoichiometry-q8",
            version=8,
            source_yaml=source_yaml,
        )

        row = self.connection.execute(
            """
            SELECT slug, version, source_yaml
            FROM template_versions
            WHERE slug = %s AND version = %s
            """,
            ("stoichiometry-q8", 8),
        ).fetchone()
        self.assertEqual(row["slug"], "stoichiometry-q8")
        self.assertEqual(row["version"], 8)
        self.assertEqual(row["source_yaml"], source_yaml)

        self._assert_row_is_immutable(
            table_name="template_versions",
            key_columns={"slug": "stoichiometry-q8", "version": 8},
            assignments={"slug": "stoichiometry-q8-mutated"},
            expected_row={
                "slug": "stoichiometry-q8",
                "version": 8,
                "source_yaml": source_yaml,
            },
        )
        self._assert_row_is_immutable(
            table_name="template_versions",
            key_columns={"slug": "stoichiometry-q8", "version": 8},
            assignments={"version": 9},
            expected_row={
                "slug": "stoichiometry-q8",
                "version": 8,
                "source_yaml": source_yaml,
            },
        )
        self._assert_row_is_immutable(
            table_name="template_versions",
            key_columns={"slug": "stoichiometry-q8", "version": 8},
            assignments={"source_yaml": "slug: changed-template"},
            expected_row={
                "slug": "stoichiometry-q8",
                "version": 8,
                "source_yaml": source_yaml,
            },
        )

    def test_template_versions_cannot_be_deleted_while_exam_definitions_reference_them(
        self,
    ) -> None:
        self._require_tables("template_versions", "exam_definitions")

        template_version_id = self._insert_template_version(
            slug="chem101-delete-guard-template",
            version=1,
        )
        self._insert_exam_definition(
            slug="chem101-delete-guard-exam",
            version=1,
            template_version_id=template_version_id,
        )

        with self.assertRaises(errors.ForeignKeyViolation):
            self.connection.execute(
                """
                DELETE FROM template_versions
                WHERE slug = %s AND version = %s
                """,
                ("chem101-delete-guard-template", 1),
            )

        self.assertEqual(
            self._count_rows(
                "template_versions",
                "slug = %s AND version = %s",
                ("chem101-delete-guard-template", 1),
            ),
            1,
        )

    def test_template_versions_can_be_deleted_when_unreferenced(self) -> None:
        self._require_tables("template_versions")

        self._insert_template_version(
            slug="chem101-template-delete-clean",
            version=1,
        )

        self.connection.execute(
            """
            DELETE FROM template_versions
            WHERE slug = %s AND version = %s
            """,
            ("chem101-template-delete-clean", 1),
        )
        self.assertEqual(
            self._count_rows(
                "template_versions",
                "slug = %s AND version = %s",
                ("chem101-template-delete-clean", 1),
            ),
            0,
        )

    def test_template_versions_require_positive_versions_and_complete_metadata_fields(
        self,
    ) -> None:
        self._require_tables("template_versions")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_template_version(slug="stoichiometry-q2", version=None)

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(slug="stoichiometry-q3", version=0)

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(slug="stoichiometry-q4", version=-1)

        with self.assertRaises(errors.NotNullViolation):
            self._insert_template_version(slug=None, version=1)

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(slug="", version=2)

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(slug="   ", version=3)

        with self.assertRaises(errors.NotNullViolation):
            self._insert_template_version(
                slug="stoichiometry-q5",
                version=1,
                source_yaml=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(
                slug="stoichiometry-q6",
                version=1,
                source_yaml="",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(
                slug="stoichiometry-q7",
                version=1,
                source_yaml="   ",
            )

    def test_exam_definitions_preserve_exact_metadata_and_are_immutable(self) -> None:
        self._require_tables("template_versions", "exam_definitions")

        template_version_id = self._insert_template_version(
            slug="chem101-reconstruct-template",
            version=1,
            source_yaml="slug: chem101-reconstruct-template\nversion: 1",
        )
        self._insert_exam_definition(
            slug="chem101-reconstruct-midterm",
            version=2,
            template_version_id=template_version_id,
            title="Chem 101 Midterm Variant B",
        )

        row = self.connection.execute(
            """
            SELECT slug, version, title, template_version_id
            FROM exam_definitions
            WHERE slug = %s AND version = %s
            """,
            ("chem101-reconstruct-midterm", 2),
        ).fetchone()
        self.assertEqual(row["slug"], "chem101-reconstruct-midterm")
        self.assertEqual(row["version"], 2)
        self.assertEqual(row["title"], "Chem 101 Midterm Variant B")
        self.assertEqual(row["template_version_id"], template_version_id)

        replacement_template_id = self._insert_template_version(
            slug="chem101-reconstruct-template",
            version=2,
            source_yaml="slug: chem101-reconstruct-template\nversion: 2",
        )
        expected_row = {
            "slug": "chem101-reconstruct-midterm",
            "version": 2,
            "title": "Chem 101 Midterm Variant B",
            "template_version_id": template_version_id,
        }
        self._assert_row_is_immutable(
            table_name="exam_definitions",
            key_columns={"slug": "chem101-reconstruct-midterm", "version": 2},
            assignments={"slug": "chem101-reconstruct-midterm-mutated"},
            expected_row=expected_row,
        )
        self._assert_row_is_immutable(
            table_name="exam_definitions",
            key_columns={"slug": "chem101-reconstruct-midterm", "version": 2},
            assignments={"version": 3},
            expected_row=expected_row,
        )
        self._assert_row_is_immutable(
            table_name="exam_definitions",
            key_columns={"slug": "chem101-reconstruct-midterm", "version": 2},
            assignments={"title": "Chem 101 Midterm Mutated"},
            expected_row=expected_row,
        )
        self._assert_row_is_immutable(
            table_name="exam_definitions",
            key_columns={"slug": "chem101-reconstruct-midterm", "version": 2},
            assignments={"template_version_id": replacement_template_id},
            expected_row=expected_row,
        )

    def test_exam_definitions_require_positive_versions_and_complete_metadata_fields(
        self,
    ) -> None:
        self._require_tables("template_versions", "exam_definitions")

        template_version_id = self._insert_template_version(
            slug="chem101-metadata-template",
            version=1,
        )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_definition(
                slug="chem101-null-version",
                version=None,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="chem101-zero-version",
                version=0,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="chem101-negative-version",
                version=-1,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_definition(
                slug=None,
                version=1,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="",
                version=2,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="   ",
                version=3,
                template_version_id=template_version_id,
            )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_definition(
                slug="chem101-null-title",
                version=4,
                template_version_id=template_version_id,
                title=None,
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="chem101-empty-title",
                version=5,
                template_version_id=template_version_id,
                title="",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="chem101-blank-title",
                version=6,
                template_version_id=template_version_id,
                title="   ",
            )

    def _require_tables(self, *table_names: str) -> None:
        actual_tables = self._list_table_names()
        missing_tables = [name for name in table_names if name not in actual_tables]
        if missing_tables:
            requirement = self._testMethodName.removeprefix("test_").replace("_", " ")
            self.fail(
                "Cannot verify "
                f"{requirement} because initialize_schema() is still missing "
                f"required table(s) in schema {self.schema_name}: {missing_tables}"
            )

    def _list_table_names(self) -> set[str]:
        return {
            row["tablename"]
            for row in self.connection.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                """
            )
        }

    def _database_clock_now(self) -> datetime:
        row = self.connection.execute(
            "SELECT clock_timestamp() AS current_time"
        ).fetchone()
        assert row is not None
        return row["current_time"].astimezone(timezone.utc)

    def _assert_row_is_immutable(
        self,
        table_name: str,
        key_columns: dict[str, object],
        assignments: dict[str, object],
        expected_row: dict[str, object],
    ) -> None:
        set_clause = ", ".join(f"{column} = %s" for column in assignments)
        where_clause = " AND ".join(f"{column} = %s" for column in key_columns)
        values = list(assignments.values()) + list(key_columns.values())
        exact_where_clause = " AND ".join(f"{column} = %s" for column in expected_row)
        assignment_where_clause = " AND ".join(f"{column} = %s" for column in assignments)
        stable_key_columns = {
            column: value
            for column, value in key_columns.items()
            if column not in assignments
        }
        stable_scope_clause = " AND ".join(
            f"{column} = %s" for column in stable_key_columns
        )
        mutation_scope_predicate = assignment_where_clause
        mutation_scope_values: list[object] = list(assignments.values())
        if stable_scope_clause:
            mutation_scope_predicate = (
                f"{mutation_scope_predicate} AND {stable_scope_clause}"
            )
            mutation_scope_values.extend(stable_key_columns.values())
        assignment_filter_values = tuple(mutation_scope_values) + tuple(
            expected_row.values()
        )
        preexisting_mutated_count = self.connection.execute(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {table_name}
            WHERE {mutation_scope_predicate}
              AND NOT ({exact_where_clause})
            """,
            assignment_filter_values,
        ).fetchone()["row_count"]

        try:
            self.connection.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}",
                values,
            )
        except psycopg.Error:
            pass

        exact_row_count = self.connection.execute(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {table_name}
            WHERE {exact_where_clause}
            """,
            tuple(expected_row.values()),
        ).fetchone()["row_count"]
        self.assertEqual(
            exact_row_count,
            1,
            f"{table_name} should still contain exactly one unchanged row for {expected_row}.",
        )

        row = self.connection.execute(
            f"""
            SELECT {", ".join(expected_row)}
            FROM {table_name}
            WHERE {where_clause}
            """,
            tuple(key_columns.values()),
        ).fetchone()
        self.assertIsNotNone(
            row,
            f"{table_name} row addressed by {key_columns} disappeared after mutation.",
        )
        self.assertEqual(dict(row), expected_row)

        unexpected_mutated_count = self.connection.execute(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {table_name}
            WHERE {mutation_scope_predicate}
              AND NOT ({exact_where_clause})
            """,
            assignment_filter_values,
        ).fetchone()["row_count"]
        self.assertEqual(
            unexpected_mutated_count,
            preexisting_mutated_count,
            f"{table_name} should not gain side-effect rows matching {assignments}.",
        )

    def _cleanup_schema(self) -> None:
        if not self._schema_created:
            return
        with suppress(Exception):
            with psycopg.connect(self.database_url, autocommit=True) as admin_connection:
                admin_connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.schema_name)
                    )
                )
        self._schema_created = False

    def _insert_student(self, student_key: str | None) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO students (student_key)
            VALUES (%s)
            RETURNING id
            """,
            (student_key,),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_template_version(
        self,
        slug: str | None = "stoichiometry-q1",
        version: int | None = 1,
        source_yaml: str | None | object = _UNSET,
    ) -> int:
        if source_yaml is _UNSET:
            source_yaml = f"slug: {slug}"
        cursor = self.connection.execute(
            """
            INSERT INTO template_versions (slug, version, source_yaml)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (slug, version, source_yaml),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_exam_definition(
        self,
        slug: str | None = "chem101-midterm",
        version: int | None = 1,
        template_version_id: int | None = None,
        title: str | None = "Chem 101 Midterm",
    ) -> int:
        if template_version_id is None:
            template_version_id = self._insert_template_version(
                slug=f"{slug}-template",
                version=version,
            )
        cursor = self.connection.execute(
            """
            INSERT INTO exam_definitions (slug, version, title, template_version_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (slug, version, title, template_version_id),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_exam_instance(
        self,
        student_key: str = "student-001",
        exam_slug: str = "chem101-midterm",
        attempt_number: int = 1,
        opaque_instance_code: str = "inst-001",
    ) -> int:
        student_id = self._insert_student(student_key)
        exam_definition_id = self._insert_exam_definition(slug=exam_slug)
        return self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=attempt_number,
            opaque_instance_code=opaque_instance_code,
        )

    def _insert_exam_instance_record(
        self,
        exam_definition_id: int,
        student_id: int,
        attempt_number: int | None,
        opaque_instance_code: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO exam_instances (
                exam_definition_id,
                student_id,
                attempt_number,
                opaque_instance_code
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                exam_definition_id,
                student_id,
                attempt_number,
                opaque_instance_code,
            ),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_exam_page(
        self,
        exam_instance_id: int,
        page_number: int | None,
        fallback_page_code: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO exam_pages (exam_instance_id, page_number, fallback_page_code)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (exam_instance_id, page_number, fallback_page_code),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_scan_artifact(
        self,
        sha256: str | None,
        original_filename: str | None,
        status: str | None,
        failure_reason: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO scan_artifacts (
                sha256,
                original_filename,
                status,
                failure_reason
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (sha256, original_filename, status, failure_reason),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_grade_record(
        self,
        exam_instance_id: int,
        status: str | None,
        score_points: float | None,
        max_points: float | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO grade_records (
                exam_instance_id,
                status,
                score_points,
                max_points
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (exam_instance_id, status, score_points, max_points),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _insert_audit_event(
        self,
        entity_type: str | None,
        entity_id: int | None,
        event_type: str | None,
        payload_json: str | None,
        recorded_at: str | None | object = _UNSET,
    ) -> int:
        if recorded_at is _UNSET:
            cursor = self.connection.execute(
                """
                INSERT INTO audit_events (
                    entity_type,
                    entity_id,
                    event_type,
                    payload_json
                )
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (entity_type, entity_id, event_type, payload_json),
            )
        else:
            cursor = self.connection.execute(
                """
                INSERT INTO audit_events (
                    entity_type,
                    entity_id,
                    event_type,
                    payload_json,
                    recorded_at
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::timestamptz)
                RETURNING id
                """,
                (entity_type, entity_id, event_type, payload_json, recorded_at),
            )
        row = cursor.fetchone()
        assert row is not None
        return int(row["id"])

    def _count_rows(
        self,
        table_name: str,
        where_clause: str = "",
        params: tuple[object, ...] = (),
    ) -> int:
        predicate = f" WHERE {where_clause}" if where_clause else ""
        row = self.connection.execute(
            f"SELECT COUNT(*) AS row_count FROM {table_name}{predicate}",
            params,
        ).fetchone()
        assert row is not None
        return int(row["row_count"])

    def _sha256(self, value: int) -> str:
        return f"{value:064x}"


if __name__ == "__main__":
    unittest.main()
