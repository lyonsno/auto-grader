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

from auto_grader import db as db_module
from auto_grader.db import initialize_schema

_UNSET = object()
_WHITESPACE_ONLY_TEXT_VALUES = ("", "   ", "\n", "\t", "\n\t")


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


class PostgresDatabaseContractTests(unittest.TestCase):
    """Fail-first Postgres-backed contract tests for the first schema slice."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run Postgres schema contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the Postgres schema contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "Postgres contract tests require psycopg in the active environment. "
                "Run `uv sync` and `uv run python -m unittest "
                "tests.test_db_postgres_contract -q`."
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

    def test_initialize_schema_is_idempotent_for_existing_schema(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
            "scan_artifacts",
            "grade_records",
            "audit_events",
        )

        source_yaml = "slug: idempotent-template\nversion: 1"
        template_version_id = self._insert_template_version(
            slug="idempotent-template",
            version=1,
            source_yaml=source_yaml,
        )
        self._assert_row_is_immutable(
            table_name="template_versions",
            key_columns={"slug": "idempotent-template", "version": 1},
            assignments={"source_yaml": "slug: mutated-template"},
            expected_row={
                "slug": "idempotent-template",
                "version": 1,
                "source_yaml": source_yaml,
            },
        )

        exam_definition_id = self._insert_exam_definition(
            slug="idempotent-midterm",
            version=1,
            template_version_id=template_version_id,
            title="Idempotent Midterm",
        )
        student_id = self._insert_student("student-idempotent")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-idempotent-001",
        )
        exam_page_id = self._insert_exam_page(
            exam_instance_id,
            1,
            "IDEMPOTENT-P1",
        )
        grade_record_id = self._insert_grade_record(
            exam_instance_id,
            "finalized",
            18.0,
            20.0,
        )
        scan_artifact_id = self._insert_scan_artifact(
            sha256=self._sha256(901),
            original_filename="idempotent-scan.png",
            status="matched",
            failure_reason=None,
        )
        audit_event_id = self._insert_audit_event(
            entity_type="exam_instance",
            entity_id=exam_instance_id,
            event_type="bootstrapped",
            payload_json='{"phase":"before-rerun"}',
        )

        initialize_schema(self.connection)

        self.assertEqual(
            self._list_table_names(),
            {
                "students",
                "template_versions",
                "exam_definitions",
                "exam_instances",
                "exam_pages",
                "scan_artifacts",
                "grade_records",
                "audit_events",
            },
            "initialize_schema() must be rerunnable against an existing schema "
            "without dropping or renaming workflow tables.",
        )
        self.assertEqual(self._count_rows("students"), 1)
        self.assertEqual(self._count_rows("template_versions"), 1)
        self.assertEqual(self._count_rows("exam_definitions"), 1)
        self.assertEqual(self._count_rows("exam_instances"), 1)
        self.assertEqual(self._count_rows("exam_pages"), 1)
        self.assertEqual(self._count_rows("grade_records"), 1)
        self.assertEqual(self._count_rows("scan_artifacts"), 1)
        self.assertEqual(self._count_rows("audit_events"), 1)

        exam_page_row = self.connection.execute(
            """
            SELECT id, exam_instance_id, page_number, fallback_page_code
            FROM exam_pages
            WHERE id = %s
            """,
            (exam_page_id,),
        ).fetchone()
        self.assertEqual(
            dict(exam_page_row),
            {
                "id": exam_page_id,
                "exam_instance_id": exam_instance_id,
                "page_number": 1,
                "fallback_page_code": "IDEMPOTENT-P1",
            },
            "Rerunning initialize_schema() must preserve populated exam page rows.",
        )
        grade_row = self.connection.execute(
            """
            SELECT id, status, score_points, max_points
            FROM grade_records
            WHERE id = %s
            """,
            (grade_record_id,),
        ).fetchone()
        self.assertEqual(
            dict(grade_row),
            {
                "id": grade_record_id,
                "status": "finalized",
                "score_points": 18.0,
                "max_points": 20.0,
            },
            "Rerunning initialize_schema() must preserve populated grade rows.",
        )
        self.assertEqual(
            self._count_rows("scan_artifacts", "id = %s", (scan_artifact_id,)),
            1,
            "Rerunning initialize_schema() must preserve existing scan artifacts.",
        )
        self.assertEqual(
            self._count_rows("audit_events", "id = %s", (audit_event_id,)),
            1,
            "Rerunning initialize_schema() must preserve existing audit events.",
        )

        with self.assertRaises(errors.UniqueViolation):
            self._insert_exam_page(exam_instance_id, 1, "IDEMPOTENT-P1-DUP")
        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(exam_instance_id, 2, "   ")

        self._assert_row_is_immutable(
            table_name="template_versions",
            key_columns={"slug": "idempotent-template", "version": 1},
            assignments={"source_yaml": "slug: mutated-template"},
            expected_row={
                "slug": "idempotent-template",
                "version": 1,
                "source_yaml": source_yaml,
            },
        )
        with self.assertRaises(errors.UniqueViolation):
            self._insert_grade_record(exam_instance_id, "finalized", 19.0, 20.0)

    def test_initialize_schema_hard_fails_when_legacy_rows_violate_strengthened_nonblank_constraints(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
            "scan_artifacts",
            "audit_events",
        )

        self._replace_nonblank_constraints_with_legacy_space_only_checks()

        self._insert_student("\n")

        with self.assertRaises(errors.CheckViolation) as exc_info:
            initialize_schema(self.connection)

        self.assertEqual(
            getattr(exc_info.exception.diag, "constraint_name", None),
            "students_student_key_nonblank",
            "Rerunning initialize_schema() must hard-fail at the strengthened "
            "constraint when legacy rows still contain newly forbidden "
            "whitespace-only values.",
        )

    def test_initialize_schema_strengthens_legacy_space_only_nonblank_constraints_when_existing_data_is_clean(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
            "scan_artifacts",
            "audit_events",
        )

        self._replace_nonblank_constraints_with_legacy_space_only_checks()

        initialize_schema(self.connection)

        with self.assertRaises(errors.CheckViolation):
            self._insert_student("\n")

        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(slug="\n", version=1)
        with self.assertRaises(errors.CheckViolation):
            self._insert_template_version(
                slug="legacy-upgrade-template",
                version=2,
                source_yaml="\t",
            )

        template_version_id = self._insert_template_version(
            slug="legacy-upgrade-template",
            version=3,
        )
        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="\n",
                version=1,
                template_version_id=template_version_id,
            )
        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_definition(
                slug="legacy-upgrade-exam",
                version=2,
                template_version_id=template_version_id,
                title="\t",
            )

        student_id = self._insert_student("legacy-upgrade-student")
        exam_definition_id = self._insert_exam_definition(
            slug="legacy-upgrade-midterm",
            version=1,
            template_version_id=template_version_id,
        )
        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="\n",
            )

        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=2,
            opaque_instance_code="legacy-upgrade-instance",
        )
        with self.assertRaises(errors.CheckViolation):
            self._insert_exam_page(
                exam_instance_id,
                1,
                "\t",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(950),
                original_filename="\n",
                status="matched",
                failure_reason=None,
            )
        with self.assertRaises(errors.CheckViolation):
            self._insert_scan_artifact(
                sha256=self._sha256(951),
                original_filename="legacy-upgrade.png",
                status="ambiguous",
                failure_reason="\t",
            )

        with self.assertRaises(errors.CheckViolation):
            self._insert_audit_event(
                entity_type="\n",
                entity_id=exam_instance_id,
                event_type="legacy_upgrade_check",
                payload_json='{"ok":true}',
            )
        with self.assertRaises(errors.CheckViolation):
            self._insert_audit_event(
                entity_type="exam_instance",
                entity_id=exam_instance_id,
                event_type="\t",
                payload_json='{"ok":true}',
            )

    def test_students_require_unique_nonblank_student_keys(self) -> None:
        self._require_tables("students")

        self._insert_student("student-001")

        with self.assertRaises(errors.UniqueViolation):
            self._insert_student("student-001")

        with self.assertRaises(errors.NotNullViolation):
            self._insert_student(None)

        for student_key in _WHITESPACE_ONLY_TEXT_VALUES:
            with self.subTest(student_key=student_key):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_student(student_key)

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

        for opaque_instance_code in _WHITESPACE_ONLY_TEXT_VALUES:
            with self.subTest(opaque_instance_code=opaque_instance_code):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_exam_instance_record(
                        exam_definition_id=exam_definition_id,
                        student_id=student_id,
                        attempt_number=1,
                        opaque_instance_code=opaque_instance_code,
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

        for fallback_page_code in _WHITESPACE_ONLY_TEXT_VALUES:
            with self.subTest(fallback_page_code=fallback_page_code):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_exam_page(
                        exam_instance_id,
                        1,
                        fallback_page_code,
                    )

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

        for original_filename in _WHITESPACE_ONLY_TEXT_VALUES:
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
            (21, "unmatched", "\n", "scan-bad-reason-unmatched-newline.png"),
            (22, "unmatched", "\t", "scan-bad-reason-unmatched-tab.png"),
            (23, "ambiguous", "", "scan-bad-reason-ambiguous-empty.png"),
            (24, "ambiguous", "   ", "scan-bad-reason-ambiguous-blank.png"),
            (25, "ambiguous", "\n", "scan-bad-reason-ambiguous-newline.png"),
            (26, "ambiguous", "\t", "scan-bad-reason-ambiguous-tab.png"),
        ):
            with self.subTest(status=status, failure_reason=failure_reason):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_scan_artifact(
                        sha256=self._sha256(digest_seed),
                        original_filename=filename,
                        status=status,
                        failure_reason=failure_reason,
                    )

    def test_scan_artifacts_reject_invalid_status_reason_transitions(self) -> None:
        self._require_tables("scan_artifacts")

        matched_id = self._insert_scan_artifact(
            sha256=self._sha256(30),
            original_filename="scan-transition-matched.png",
            status="matched",
            failure_reason=None,
        )
        unmatched_id = self._insert_scan_artifact(
            sha256=self._sha256(31),
            original_filename="scan-transition-unmatched.png",
            status="unmatched",
            failure_reason="qr_decode_failed",
        )
        ambiguous_id = self._insert_scan_artifact(
            sha256=self._sha256(32),
            original_filename="scan-transition-ambiguous.png",
            status="ambiguous",
            failure_reason="multiple_qr_candidates",
        )

        with self.assertRaises(errors.CheckViolation):
            self.connection.execute(
                """
                UPDATE scan_artifacts
                SET failure_reason = %s
                WHERE id = %s
                """,
                ("should_not_exist_for_success", matched_id),
            )

        with self.assertRaises(errors.CheckViolation):
            self.connection.execute(
                """
                UPDATE scan_artifacts
                SET status = %s
                WHERE id = %s
                """,
                ("matched", unmatched_id),
            )

        with self.assertRaises(errors.CheckViolation):
            self.connection.execute(
                """
                UPDATE scan_artifacts
                SET failure_reason = %s
                WHERE id = %s
                """,
                ("   ", ambiguous_id),
            )

        rows = self.connection.execute(
            """
            SELECT id, status, failure_reason
            FROM scan_artifacts
            WHERE id IN (%s, %s, %s)
            ORDER BY id
            """,
            (matched_id, unmatched_id, ambiguous_id),
        ).fetchall()
        self.assertEqual(
            [dict(row) for row in rows],
            [
                {
                    "id": matched_id,
                    "status": "matched",
                    "failure_reason": None,
                },
                {
                    "id": unmatched_id,
                    "status": "unmatched",
                    "failure_reason": "qr_decode_failed",
                },
                {
                    "id": ambiguous_id,
                    "status": "ambiguous",
                    "failure_reason": "multiple_qr_candidates",
                },
            ],
            "Invalid scan-artifact status transitions must be rejected and leave "
            "tracked artifact state unchanged.",
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

    def test_grade_records_block_second_finalize_transition_for_same_exam_instance(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        exam_definition_id = self._insert_exam_definition("chem101-finalize-transition")
        student_id = self._insert_student("student-finalize-transition")
        exam_instance_id = self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-finalize-transition",
        )
        first_grade_id = self._insert_grade_record(
            exam_instance_id,
            "draft",
            18.0,
            20.0,
        )
        second_grade_id = self._insert_grade_record(
            exam_instance_id,
            "draft",
            17.0,
            20.0,
        )

        self.connection.execute(
            """
            UPDATE grade_records
            SET status = 'finalized'
            WHERE id = %s
            """,
            (first_grade_id,),
        )

        with self.assertRaises(errors.UniqueViolation):
            self.connection.execute(
                """
                UPDATE grade_records
                SET status = 'finalized'
                WHERE id = %s
                """,
                (second_grade_id,),
            )

        rows = self.connection.execute(
            """
            SELECT id, status
            FROM grade_records
            WHERE exam_instance_id = %s
            ORDER BY id
            """,
            (exam_instance_id,),
        ).fetchall()
        self.assertEqual(
            [dict(row) for row in rows],
            [
                {"id": first_grade_id, "status": "finalized"},
                {"id": second_grade_id, "status": "draft"},
            ],
            "The one-finalized-grade invariant must hold on UPDATE, not only on "
            "INSERT.",
        )

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
            ("\n", 1),
            ("\t", 1),
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

        for event_type in (None, "", "   ", "\n", "\t"):
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

        for slug, version in zip(_WHITESPACE_ONLY_TEXT_VALUES, range(2, 7)):
            with self.subTest(slug=slug):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_template_version(slug=slug, version=version)

        with self.assertRaises(errors.NotNullViolation):
            self._insert_template_version(
                slug="stoichiometry-q5",
                version=1,
                source_yaml=None,
            )

        for source_yaml, suffix in zip(_WHITESPACE_ONLY_TEXT_VALUES, range(6, 11)):
            with self.subTest(source_yaml=source_yaml):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_template_version(
                        slug=f"stoichiometry-q{suffix}",
                        version=1,
                        source_yaml=source_yaml,
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

        for slug, version in zip(_WHITESPACE_ONLY_TEXT_VALUES, range(2, 7)):
            with self.subTest(slug=slug):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_exam_definition(
                        slug=slug,
                        version=version,
                        template_version_id=template_version_id,
                    )

        with self.assertRaises(errors.NotNullViolation):
            self._insert_exam_definition(
                slug="chem101-null-title",
                version=4,
                template_version_id=template_version_id,
                title=None,
            )

        for title, version in zip(_WHITESPACE_ONLY_TEXT_VALUES, range(5, 10)):
            with self.subTest(title=title):
                with self.assertRaises(errors.CheckViolation):
                    self._insert_exam_definition(
                        slug=f"chem101-title-{version}",
                        version=version,
                        template_version_id=template_version_id,
                        title=title,
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

        with self.assertRaisesRegex(
            psycopg.Error,
            "(?is)immutable|versioned table rows are immutable",
        ):
            self.connection.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}",
                values,
            )

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

    def _replace_check_constraint(
        self,
        table_name: str,
        constraint_name: str,
        check_expression: str,
    ) -> None:
        self.connection.execute(
            f"""
            ALTER TABLE {table_name}
            DROP CONSTRAINT IF EXISTS {constraint_name}
            """
        )
        self.connection.execute(
            f"""
            ALTER TABLE {table_name}
            ADD CONSTRAINT {constraint_name}
            CHECK ({check_expression})
            """
        )

    def _replace_nonblank_constraints_with_legacy_space_only_checks(self) -> None:
        self._replace_check_constraint(
            "students",
            "students_student_key_nonblank",
            "btrim(student_key) <> ''",
        )
        self._replace_check_constraint(
            "template_versions",
            "template_versions_slug_nonblank",
            "btrim(slug) <> ''",
        )
        self._replace_check_constraint(
            "template_versions",
            "template_versions_source_yaml_nonblank",
            "btrim(source_yaml) <> ''",
        )
        self._replace_check_constraint(
            "exam_definitions",
            "exam_definitions_slug_nonblank",
            "btrim(slug) <> ''",
        )
        self._replace_check_constraint(
            "exam_definitions",
            "exam_definitions_title_nonblank",
            "btrim(title) <> ''",
        )
        self._replace_check_constraint(
            "exam_instances",
            "exam_instances_opaque_code_nonblank",
            "btrim(opaque_instance_code) <> ''",
        )
        self._replace_check_constraint(
            "exam_pages",
            "exam_pages_fallback_page_code_nonblank",
            "btrim(fallback_page_code) <> ''",
        )
        self._replace_check_constraint(
            "scan_artifacts",
            "scan_artifacts_original_filename_nonblank",
            "btrim(original_filename) <> ''",
        )
        self._replace_check_constraint(
            "scan_artifacts",
            "scan_artifacts_failure_reason_rules",
            """
            (status = 'matched' AND failure_reason IS NULL)
            OR (
                status IN ('unmatched', 'ambiguous')
                AND failure_reason IS NOT NULL
                AND btrim(failure_reason) <> ''
            )
            """,
        )
        self._replace_check_constraint(
            "audit_events",
            "audit_events_entity_type_nonblank",
            "btrim(entity_type) <> ''",
        )
        self._replace_check_constraint(
            "audit_events",
            "audit_events_event_type_nonblank",
            "btrim(event_type) <> ''",
        )

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
