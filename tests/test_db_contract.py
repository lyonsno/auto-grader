from __future__ import annotations

import sqlite3
import unittest

from auto_grader.db import create_connection, initialize_schema


class DatabaseContractTests(unittest.TestCase):
    """Fail-first contract tests for the first database slice.

    These tests intentionally lock in a small working vocabulary for v0:
    SQLite, explicit status fields, and DB-level constraints for the most important
    invariants. Table and column names can still evolve later if we migrate the tests
    along with the schema.
    """

    def setUp(self) -> None:
        self.connection = create_connection()
        initialize_schema(self.connection)

    def tearDown(self) -> None:
        self.connection.close()

    def test_schema_exposes_core_workflow_tables(self) -> None:
        actual_tables = {
            row["name"]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

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
            f"Missing core workflow tables: {sorted(missing_tables)}",
        )

    def test_student_cannot_have_duplicate_exam_instance_for_same_attempt(self) -> None:
        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition("chem101-midterm")

        self.connection.execute(
            """
            INSERT INTO exam_instances (
                exam_definition_id,
                student_id,
                attempt_number,
                opaque_instance_code
            )
            VALUES (?, ?, ?, ?)
            """,
            (exam_definition_id, student_id, 1, "inst-001"),
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
            self.connection.execute(
                """
                INSERT INTO exam_instances (
                    exam_definition_id,
                    student_id,
                    attempt_number,
                    opaque_instance_code
                )
                VALUES (?, ?, ?, ?)
                """,
                (exam_definition_id, student_id, 1, "inst-002"),
            )

    def test_exam_page_identity_is_unique_within_an_exam_instance(self) -> None:
        exam_instance_id = self._insert_exam_instance()

        self.connection.execute(
            """
            INSERT INTO exam_pages (
                exam_instance_id,
                page_number,
                fallback_page_code
            )
            VALUES (?, ?, ?)
            """,
            (exam_instance_id, 1, "MIDTERM1-P1-A1"),
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
            self.connection.execute(
                """
                INSERT INTO exam_pages (
                    exam_instance_id,
                    page_number,
                    fallback_page_code
                )
                VALUES (?, ?, ?)
                """,
                (exam_instance_id, 1, "MIDTERM1-P1-A2"),
            )

    def test_scan_artifact_failure_reason_is_required_for_unmatched_status(self) -> None:
        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|NOT NULL"):
            self.connection.execute(
                """
                INSERT INTO scan_artifacts (
                    sha256,
                    original_filename,
                    status,
                    failure_reason
                )
                VALUES (?, ?, ?, ?)
                """,
                ("abc123", "scan-001.png", "unmatched", None),
            )

    def test_only_one_finalized_grade_is_allowed_per_exam_instance(self) -> None:
        exam_instance_id = self._insert_exam_instance()

        self.connection.execute(
            """
            INSERT INTO grade_records (
                exam_instance_id,
                status,
                score_points,
                max_points
            )
            VALUES (?, ?, ?, ?)
            """,
            (exam_instance_id, "draft", 18.0, 20.0),
        )
        self.connection.execute(
            """
            INSERT INTO grade_records (
                exam_instance_id,
                status,
                score_points,
                max_points
            )
            VALUES (?, ?, ?, ?)
            """,
            (exam_instance_id, "finalized", 18.0, 20.0),
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
            self.connection.execute(
                """
                INSERT INTO grade_records (
                    exam_instance_id,
                    status,
                    score_points,
                    max_points
                )
                VALUES (?, ?, ?, ?)
                """,
                (exam_instance_id, "finalized", 19.0, 20.0),
            )

    def _insert_student(self, student_key: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO students (
                student_key,
                display_name
            )
            VALUES (?, ?)
            """,
            (student_key, "Student Name"),
        )
        return int(cursor.lastrowid)

    def _insert_template_version(self, slug: str = "stoichiometry-q1", version: int = 1) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO template_versions (
                slug,
                version,
                source_yaml
            )
            VALUES (?, ?, ?)
            """,
            (slug, version, "slug: stoichiometry-q1"),
        )
        return int(cursor.lastrowid)

    def _insert_exam_definition(self, slug: str = "chem101-midterm") -> int:
        template_version_id = self._insert_template_version()
        cursor = self.connection.execute(
            """
            INSERT INTO exam_definitions (
                slug,
                version,
                title,
                template_version_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (slug, 1, "Chem 101 Midterm", template_version_id),
        )
        return int(cursor.lastrowid)

    def _insert_exam_instance(self) -> int:
        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition()
        cursor = self.connection.execute(
            """
            INSERT INTO exam_instances (
                exam_definition_id,
                student_id,
                attempt_number,
                opaque_instance_code
            )
            VALUES (?, ?, ?, ?)
            """,
            (exam_definition_id, student_id, 1, "inst-001"),
        )
        return int(cursor.lastrowid)


if __name__ == "__main__":
    unittest.main()
