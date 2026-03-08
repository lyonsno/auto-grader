from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
import unittest

from auto_grader.db import create_connection, initialize_schema

_DEFAULT = object()


class DatabaseContractTests(unittest.TestCase):
    """Fail-first contract tests for the first database slice.

    These tests intentionally lock in a small working vocabulary for v0:
    SQLite, explicit status fields, and DB-level constraints for the most important
    invariants.
    """

    def setUp(self) -> None:
        self.connection = create_connection()
        initialize_schema(self.connection)

    def tearDown(self) -> None:
        self.connection.close()

    def test_schema_exposes_initial_sqlite_workflow_tables(self) -> None:
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

    def test_students_require_unique_nonblank_student_keys(self) -> None:
        self._require_tables("students")

        self._insert_student("student-001")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_student("student-001")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_student(None)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_student("")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_student("   ")

    def test_student_cannot_have_duplicate_exam_instance_for_same_attempt(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-001")
        midterm_id = self._insert_exam_definition("chem101-midterm")
        final_id = self._insert_exam_definition("chem101-final")

        self._insert_exam_instance_record(
            exam_definition_id=midterm_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-001",
        )
        self._insert_exam_instance_record(
            exam_definition_id=midterm_id,
            student_id=student_id,
            attempt_number=2,
            opaque_instance_code="inst-002",
        )
        self._insert_exam_instance_record(
            exam_definition_id=final_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-003",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_exam_instance_record(
                exam_definition_id=midterm_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="inst-004",
            )

    def test_exam_instance_requires_unique_opaque_instance_code(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        first_student_id = self._insert_student("student-001")
        second_student_id = self._insert_student("student-002")
        midterm_id = self._insert_exam_definition("chem101-midterm")
        final_id = self._insert_exam_definition("chem101-final")

        self._insert_exam_instance_record(
            exam_definition_id=midterm_id,
            student_id=first_student_id,
            attempt_number=1,
            opaque_instance_code="opaque-001",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_exam_instance_record(
                exam_definition_id=final_id,
                student_id=second_student_id,
                attempt_number=1,
                opaque_instance_code="opaque-001",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_instance_record(
                exam_definition_id=final_id,
                student_id=second_student_id,
                attempt_number=2,
                opaque_instance_code=None,
            )

    def test_exam_instance_requires_nonblank_opaque_code_and_positive_attempt_number(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition("chem101-midterm")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="   ",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=0,
                opaque_instance_code="inst-zero-attempt",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=-1,
                opaque_instance_code="inst-negative-attempt",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=student_id,
                attempt_number=None,
                opaque_instance_code="inst-null-attempt",
            )

    def test_exam_instance_requires_existing_student_and_exam_definition(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-001")
        exam_definition_id = self._insert_exam_definition("chem101-midterm")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self._insert_exam_instance_record(
                exam_definition_id=exam_definition_id,
                student_id=999_001,
                attempt_number=1,
                opaque_instance_code="inst-missing-student",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self._insert_exam_instance_record(
                exam_definition_id=999_002,
                student_id=student_id,
                attempt_number=1,
                opaque_instance_code="inst-missing-exam",
            )

    def test_exam_instance_update_requires_existing_student_and_exam_definition(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-update-guard")
        exam_definition_id = self._insert_exam_definition("chem101-update-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-update-guard",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "UPDATE exam_instances SET student_id = ? WHERE opaque_instance_code = ?",
                (999_101, "inst-update-guard"),
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "UPDATE exam_instances SET exam_definition_id = ? WHERE opaque_instance_code = ?",
                (999_102, "inst-update-guard"),
            )

    def test_student_delete_is_blocked_when_exam_instances_exist(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-delete-guard")
        exam_definition_id = self._insert_exam_definition("chem101-delete-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-student",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "DELETE FROM students WHERE student_key = ?",
                ("student-delete-guard",),
            )

    def test_student_delete_succeeds_when_no_exam_instances_exist(self) -> None:
        self._require_tables("students")
        self._insert_student("student-delete-no-children")

        self.connection.execute(
            "DELETE FROM students WHERE student_key = ?",
            ("student-delete-no-children",),
        )
        remaining = self.connection.execute(
            "SELECT COUNT(*) FROM students WHERE student_key = ?",
            ("student-delete-no-children",),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_exam_definition_delete_is_blocked_when_exam_instances_exist(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
        )
        student_id = self._insert_student("student-delete-guard")
        exam_definition_id = self._insert_exam_definition("chem101-delete-guard")
        self._insert_exam_instance_record(
            exam_definition_id=exam_definition_id,
            student_id=student_id,
            attempt_number=1,
            opaque_instance_code="inst-delete-guard-exam",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "DELETE FROM exam_definitions WHERE slug = ? AND version = ?",
                ("chem101-delete-guard", 1),
            )

    def test_exam_definition_delete_succeeds_when_no_exam_instances_exist(self) -> None:
        self._require_tables("template_versions", "exam_definitions")
        self._insert_exam_definition("chem101-delete-no-children")

        self.connection.execute(
            "DELETE FROM exam_definitions WHERE slug = ? AND version = ?",
            ("chem101-delete-no-children", 1),
        )
        remaining = self.connection.execute(
            "SELECT COUNT(*) FROM exam_definitions WHERE slug = ? AND version = ?",
            ("chem101-delete-no-children", 1),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_exam_page_identity_is_unique_within_an_exam_instance(self) -> None:
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_exam_page(exam_instance_id, 1, "MIDTERM-P1-A2")

    def test_exam_page_requires_existing_exam_instance(self) -> None:
        self._require_tables("exam_instances", "exam_pages")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self._insert_exam_page(
                exam_instance_id=999_003,
                page_number=1,
                fallback_page_code="MIDTERM-P1-ORPHAN",
            )

    def test_exam_page_update_requires_existing_exam_instance(self) -> None:
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "UPDATE exam_pages SET exam_instance_id = ? WHERE fallback_page_code = ?",
                (999_103, "UPDATE-GUARD-P1"),
            )

    def test_exam_page_requires_positive_page_number_and_nonblank_fallback_code(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "exam_pages",
        )
        exam_instance_id = self._insert_exam_instance()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=0,
                fallback_page_code="MIDTERM-P0-INVALID",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=-1,
                fallback_page_code="MIDTERM-PNEG-INVALID",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=1,
                fallback_page_code=None,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=1,
                fallback_page_code="",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=1,
                fallback_page_code="   ",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_page(
                exam_instance_id=exam_instance_id,
                page_number=None,
                fallback_page_code="MIDTERM-PNULL-INVALID",
            )

    def test_scan_artifact_status_accepts_supported_values_and_rejects_unknown_values(
        self,
    ) -> None:
        self._require_tables("scan_artifacts")

        self._insert_scan_artifact(
            sha256=self._sha256(0),
            original_filename="scan-000.png",
            status="matched",
            failure_reason=None,
        )
        self._insert_scan_artifact(
            sha256=self._sha256(1),
            original_filename="scan-001.png",
            status="unmatched",
            failure_reason="qr_decode_failed",
        )
        self._insert_scan_artifact(
            sha256=self._sha256(2),
            original_filename="scan-002.png",
            status="ambiguous",
            failure_reason="multiple_qr_candidates",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|invalid"):
            self._insert_scan_artifact(
                sha256=self._sha256(3),
                original_filename="scan-003.png",
                status="bogus",
                failure_reason="status_not_in_supported_vocabulary",
            )

    def test_scan_artifact_status_rejects_blank_strings_as_unsupported_values(
        self,
    ) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, status_value, filename in (
            (self._sha256(15), "", "scan-015.png"),
            (self._sha256(16), "   ", "scan-016.png"),
        ):
            with self.subTest(status=status_value):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|blank|invalid"
                ):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=filename,
                        status=status_value,
                        failure_reason="status_not_in_supported_vocabulary",
                    )

    def test_scan_artifact_status_requires_canonical_lowercase_values(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, status_value, failure_reason in (
            (self._sha256(18), "UNMATCHED", "qr_decode_failed"),
            (self._sha256(19), "Ambiguous", "multiple_qr_candidates"),
            (self._sha256(20), "Matched", None),
        ):
            with self.subTest(status=status_value):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|invalid"):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=f"scan-case-{status_value}.png",
                        status=status_value,
                        failure_reason=failure_reason,
                    )

    def test_scan_artifact_status_is_required(self) -> None:
        self._require_tables("scan_artifacts")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_scan_artifact(
                sha256=self._sha256(12),
                original_filename="scan-012.png",
                status=None,
                failure_reason=None,
            )

    def test_scan_artifact_matched_status_requires_null_failure_reason(self) -> None:
        self._require_tables("scan_artifacts")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_scan_artifact(
                sha256=self._sha256(4),
                original_filename="scan-004.png",
                status="matched",
                failure_reason="should_not_exist_for_success",
            )

    def test_scan_artifact_failure_statuses_require_reason(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, status_value, filename in (
            (self._sha256(5), "unmatched", "scan-005.png"),
            (self._sha256(6), "ambiguous", "scan-006.png"),
        ):
            with self.subTest(status=status_value):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|required"
                ):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=filename,
                        status=status_value,
                        failure_reason=None,
                    )

    def test_scan_artifact_failure_reasons_must_be_nonblank_when_required(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, status_value, filename, failure_reason in (
            (self._sha256(7), "unmatched", "scan-007.png", ""),
            (self._sha256(8), "ambiguous", "scan-008.png", "   "),
        ):
            with self.subTest(status=status_value, failure_reason=failure_reason):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=filename,
                        status=status_value,
                        failure_reason=failure_reason,
                    )

    def test_scan_artifact_sha256_is_unique_for_schema_level_deduplication(self) -> None:
        self._require_tables("scan_artifacts")
        digest = self._sha256(9)

        self._insert_scan_artifact(
            sha256=digest,
            original_filename="scan-009.png",
            status="matched",
            failure_reason=None,
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_scan_artifact(
                sha256=digest,
                original_filename="scan-009-rerun.png",
                status="matched",
                failure_reason=None,
            )

    def test_scan_artifact_checksum_is_required(self) -> None:
        self._require_tables("scan_artifacts")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_scan_artifact(
                sha256=None,
                original_filename="scan-missing-digest.png",
                status="matched",
                failure_reason=None,
            )

    def test_scan_artifact_checksum_must_be_nonblank(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, filename in (
            ("", "scan-blank-digest.png"),
            ("   ", "scan-whitespace-digest.png"),
        ):
            with self.subTest(sha256=sha256_value):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=filename,
                        status="matched",
                        failure_reason=None,
                    )

    def test_scan_artifact_original_filename_is_required(self) -> None:
        self._require_tables("scan_artifacts")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_scan_artifact(
                sha256=self._sha256(10),
                original_filename=None,
                status="matched",
                failure_reason=None,
            )

    def test_scan_artifact_original_filename_must_be_nonblank(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, original_filename in (
            (self._sha256(17), ""),
            (self._sha256(11), "   "),
        ):
            with self.subTest(original_filename=original_filename):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=original_filename,
                        status="matched",
                        failure_reason=None,
                    )

    def test_scan_artifact_sha256_must_be_64_hex_characters(self) -> None:
        self._require_tables("scan_artifacts")

        for sha256_value, filename in (
            ("abc", "scan-short-digest.png"),
            ("f" * 63, "scan-short-by-one-digest.png"),
            ("z" * 64, "scan-nonhex-digest.png"),
        ):
            with self.subTest(sha256=sha256_value):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|hex|sha256"
                ):
                    self._insert_scan_artifact(
                        sha256=sha256_value,
                        original_filename=filename,
                        status="matched",
                        failure_reason=None,
                    )

    def test_only_one_finalized_grade_is_allowed_per_exam_instance(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_definition_id = self._insert_exam_definition("chem101-midterm")
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

        self._insert_grade_record(exam_instance_id, "draft", 18.0, 20.0)
        self._insert_grade_record(exam_instance_id, "draft", 17.5, 20.0)
        self._insert_grade_record(exam_instance_id, "finalized", 18.0, 20.0)
        self._insert_grade_record(other_exam_instance_id, "finalized", 16.0, 20.0)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_grade_record(exam_instance_id, "finalized", 19.0, 20.0)

    def test_grade_record_requires_existing_exam_instance(self) -> None:
        self._require_tables("exam_instances", "grade_records")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self._insert_grade_record(
                exam_instance_id=999_004,
                status="draft",
                score_points=10.0,
                max_points=20.0,
            )

    def test_grade_record_update_requires_existing_exam_instance(self) -> None:
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "UPDATE grade_records SET exam_instance_id = ? WHERE exam_instance_id = ?",
                (999_104, exam_instance_id),
            )

    def test_exam_instance_delete_is_blocked_when_pages_exist(self) -> None:
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "DELETE FROM exam_instances WHERE opaque_instance_code = ?",
                ("inst-delete-guard-with-page",),
            )

    def test_exam_instance_delete_succeeds_when_unreferenced(self) -> None:
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
            "DELETE FROM exam_instances WHERE opaque_instance_code = ?",
            ("inst-delete-clean",),
        )
        remaining = self.connection.execute(
            "SELECT COUNT(*) FROM exam_instances WHERE opaque_instance_code = ?",
            ("inst-delete-clean",),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_exam_instance_delete_is_blocked_when_grades_exist(self) -> None:
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "DELETE FROM exam_instances WHERE opaque_instance_code = ?",
                ("inst-delete-guard-with-grade",),
            )

    def test_grade_records_accept_supported_statuses_and_reject_unknown_values(
        self,
    ) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

        self._insert_grade_record(exam_instance_id, "draft", 0.0, 20.0)
        self._insert_grade_record(exam_instance_id, "finalized", 20.0, 20.0)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|invalid"):
            self._insert_grade_record(exam_instance_id, "bogus", 10.0, 20.0)

    def test_grade_records_status_requires_canonical_lowercase_values(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )

        for index, status_value in enumerate(("DRAFT", "Finalized")):
            exam_instance_id = self._insert_exam_instance(
                student_key=f"student-grade-case-{index}",
                exam_slug=f"chem101-grade-case-{index}",
                attempt_number=1,
                opaque_instance_code=f"inst-grade-case-{index}",
            )
            with self.subTest(status=status_value):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|invalid"):
                    self._insert_grade_record(exam_instance_id, status_value, 10.0, 20.0)

    def test_grade_records_require_nonblank_status(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

        for status_value in ("", "   "):
            with self.subTest(status=status_value):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
                    self._insert_grade_record(
                        exam_instance_id,
                        status_value,
                        10.0,
                        20.0,
                    )

    def test_grade_records_require_status_and_score_fields(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

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
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "NOT NULL|required"
                ):
                    self._insert_grade_record(
                        exam_instance_id,
                        status_value,
                        score_points,
                        max_points,
                    )

    def test_grade_records_require_nonnegative_score_points(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_grade_record(exam_instance_id, "draft", -0.5, 20.0)

    def test_grade_records_require_score_points_not_exceed_max_points(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_grade_record(exam_instance_id, "draft", 20.5, 20.0)

    def test_grade_records_require_positive_max_points(self) -> None:
        self._require_tables(
            "students",
            "template_versions",
            "exam_definitions",
            "exam_instances",
            "grade_records",
        )
        exam_instance_id = self._insert_exam_instance()

        for max_points in (0.0, -5.0):
            with self.subTest(max_points=max_points):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
                    self._insert_grade_record(
                        exam_instance_id, "draft", 0.0, max_points
                    )

    def test_template_versions_require_unique_slug_version_pairs(self) -> None:
        self._require_tables("template_versions")

        self._insert_template_version(slug="stoichiometry-q1", version=1)
        self._insert_template_version(slug="stoichiometry-q1", version=2)
        self._insert_template_version(slug="kinetics-q1", version=1)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_template_version(slug="stoichiometry-q1", version=1)

    def test_template_versions_require_positive_versions_and_complete_metadata_fields(
        self,
    ) -> None:
        self._require_tables("template_versions")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_template_version(slug="stoichiometry-q2", version=None)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_template_version(slug="stoichiometry-q3", version=0)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_template_version(slug="stoichiometry-q4", version=-1)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_template_version(slug=None, version=1)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_template_version(slug="", version=2)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_template_version(slug="   ", version=3)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_template_version(
                slug="stoichiometry-q5",
                version=1,
                source_yaml=None,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_template_version(
                slug="stoichiometry-q6",
                version=1,
                source_yaml="",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_template_version(
                slug="stoichiometry-q7",
                version=1,
                source_yaml="   ",
            )

    def test_template_versions_preserve_exact_source_and_are_immutable(self) -> None:
        self._require_tables("template_versions")

        source_yaml = (
            "slug: stoichiometry-q8\n"
            "version: 8\n"
            "prompt: Balance the combustion reaction."
        )
        template_version_id = self._insert_template_version(
            slug="stoichiometry-q8",
            version=8,
            source_yaml=source_yaml,
        )
        self.assertGreater(template_version_id, 0)

        row = self.connection.execute(
            """
            SELECT slug, version, source_yaml
            FROM template_versions
            WHERE slug = ? AND version = ?
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

    def test_exam_definitions_are_versioned_and_reference_existing_templates(self) -> None:
        self._require_tables("template_versions", "exam_definitions")
        template_v1 = self._insert_template_version(slug="chem101-midterm", version=1)
        template_v2 = self._insert_template_version(slug="chem101-midterm", version=2)

        self._insert_exam_definition(
            slug="chem101-midterm",
            version=1,
            template_version_id=template_v1,
        )
        self._insert_exam_definition(
            slug="chem101-midterm",
            version=2,
            template_version_id=template_v2,
        )
        self._insert_exam_definition(
            slug="chem101-final",
            version=1,
            template_version_id=template_v1,
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_exam_definition(
                slug="chem101-midterm",
                version=1,
                template_version_id=template_v1,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self._insert_exam_definition(
                slug="chem101-quiz",
                version=1,
                template_version_id=999_005,
            )

    def test_exam_definition_update_requires_existing_template_version(self) -> None:
        self._require_tables("template_versions", "exam_definitions")
        template_v1 = self._insert_template_version(slug="chem101-update-guard", version=1)
        self._insert_exam_definition(
            slug="chem101-update-guard",
            version=1,
            template_version_id=template_v1,
        )

        try:
            self.connection.execute(
                """
                UPDATE exam_definitions
                SET template_version_id = ?
                WHERE slug = ? AND version = ?
                """,
                (999_105, "chem101-update-guard", 1),
            )
        except sqlite3.DatabaseError:
            # Either FK enforcement or immutability enforcement is acceptable.
            pass

        row = self.connection.execute(
            """
            SELECT template_version_id
            FROM exam_definitions
            WHERE slug = ? AND version = ?
            """,
            ("chem101-update-guard", 1),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["template_version_id"], template_v1)

    def test_template_version_delete_is_blocked_when_exam_definitions_reference_it(
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "FOREIGN KEY"):
            self.connection.execute(
                "DELETE FROM template_versions WHERE slug = ? AND version = ?",
                ("chem101-delete-guard-template", 1),
            )

    def test_template_version_delete_succeeds_when_unreferenced(self) -> None:
        self._require_tables("template_versions")
        self._insert_template_version(
            slug="chem101-template-delete-clean",
            version=1,
        )

        self.connection.execute(
            "DELETE FROM template_versions WHERE slug = ? AND version = ?",
            ("chem101-template-delete-clean", 1),
        )
        remaining = self.connection.execute(
            "SELECT COUNT(*) FROM template_versions WHERE slug = ? AND version = ?",
            ("chem101-template-delete-clean", 1),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_exam_definitions_require_positive_versions_and_complete_metadata_fields(
        self,
    ) -> None:
        self._require_tables("template_versions", "exam_definitions")
        template_version_id = self._insert_template_version(
            slug="chem101-metadata-template",
            version=1,
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_definition(
                slug="chem101-null-version",
                version=None,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_definition(
                slug="chem101-zero-version",
                version=0,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
            self._insert_exam_definition(
                slug="chem101-negative-version",
                version=-1,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_definition(
                slug=None,
                version=1,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_definition(
                slug="",
                version=2,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_definition(
                slug="   ",
                version=3,
                template_version_id=template_version_id,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_exam_definition(
                slug="chem101-null-title",
                version=4,
                template_version_id=template_version_id,
                title=None,
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_definition(
                slug="chem101-empty-title",
                version=5,
                template_version_id=template_version_id,
                title="",
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
            self._insert_exam_definition(
                slug="chem101-blank-title",
                version=6,
                template_version_id=template_version_id,
                title="   ",
            )

    def test_exam_definitions_preserve_exact_metadata_and_are_immutable(self) -> None:
        self._require_tables("template_versions", "exam_definitions")
        template_version_id = self._insert_template_version(
            slug="chem101-reconstruct-template",
            version=1,
            source_yaml="slug: chem101-reconstruct-template\nversion: 1",
        )
        exam_definition_id = self._insert_exam_definition(
            slug="chem101-reconstruct-midterm",
            version=2,
            template_version_id=template_version_id,
            title="Chem 101 Midterm Variant B",
        )

        row = self.connection.execute(
            """
            SELECT slug, version, title, template_version_id
            FROM exam_definitions
            WHERE slug = ? AND version = ?
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

        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE|duplicate"):
            self._insert_exam_page(second_exam_instance_id, 1, "RECOVER-P1-A")

    def test_audit_events_default_recorded_timestamp_is_present_and_current(self) -> None:
        self._require_tables("audit_events")

        before_insert = datetime.now(timezone.utc) - timedelta(seconds=5)
        self._insert_audit_event(
            entity_type="scan_artifact",
            entity_id=1,
            event_type="scan_recorded",
            payload_json="{}",
        )

        row = self.connection.execute(
            "SELECT recorded_at FROM audit_events"
        ).fetchone()
        self.assertIsNotNone(row["recorded_at"])
        self.assertNotEqual(str(row["recorded_at"]).strip(), "")
        recorded_at = self._parse_recorded_at(str(row["recorded_at"]))
        after_insert = datetime.now(timezone.utc) + timedelta(seconds=5)
        self.assertGreaterEqual(recorded_at, before_insert)
        self.assertLessEqual(recorded_at, after_insert)

    def test_audit_events_preserve_valid_explicit_recorded_timestamps(self) -> None:
        self._require_tables("audit_events")

        explicit_recorded_at = "2026-03-07T12:34:56+00:00"
        self._insert_audit_event(
            entity_type="scan_artifact",
            entity_id=1,
            event_type="scan_recorded",
            payload_json="{}",
            recorded_at=explicit_recorded_at,
        )

        row = self.connection.execute(
            "SELECT recorded_at FROM audit_events"
        ).fetchone()
        self.assertEqual(
            self._parse_recorded_at(str(row["recorded_at"])),
            self._parse_recorded_at(explicit_recorded_at),
        )

    def test_audit_events_require_complete_subject(self) -> None:
        self._require_tables("audit_events")

        for entity_type, entity_id in (
            ("", 1),
            ("   ", 1),
            (None, 1),
            ("scan_artifact", None),
        ):
            with self.subTest(entity_type=entity_type, entity_id=entity_id):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|blank|NOT NULL|required"
                ):
                    self._insert_audit_event(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        event_type="scan_recorded",
                        payload_json="{}",
                    )

    def test_audit_events_require_event_type(self) -> None:
        self._require_tables("audit_events")

        for event_type in (None, "", "   "):
            with self.subTest(event_type=event_type):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|blank|NOT NULL|required"
                ):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type=event_type,
                        payload_json="{}",
                    )

    def test_audit_events_require_payload_json(self) -> None:
        self._require_tables("audit_events")

        with self.assertRaisesRegex(sqlite3.IntegrityError, "NOT NULL|required"):
            self._insert_audit_event(
                entity_type="scan_artifact",
                entity_id=1,
                event_type="scan_recorded",
                payload_json=None,
            )

    def test_audit_events_require_nonblank_payload_json(self) -> None:
        self._require_tables("audit_events")

        for payload_json in ("", "   "):
            with self.subTest(payload_json=payload_json):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK|blank"):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json=payload_json,
                    )

    def test_audit_events_require_valid_json_payloads(self) -> None:
        self._require_tables("audit_events")

        for payload_json in ("not-json", "{"):
            with self.subTest(payload_json=payload_json):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|json|malformed"
                ):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json=payload_json,
                    )

    def test_audit_events_require_explicit_recorded_timestamp_when_supplied(self) -> None:
        self._require_tables("audit_events")

        for recorded_at in (None, "", "   "):
            with self.subTest(recorded_at=recorded_at):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|blank|NOT NULL|required"
                ):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json="{}",
                        recorded_at=recorded_at,
                    )

    def test_audit_events_require_valid_explicit_recorded_timestamps(self) -> None:
        self._require_tables("audit_events")

        for recorded_at in ("not-a-timestamp", "2026-99-99 25:61:00"):
            with self.subTest(recorded_at=recorded_at):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "CHECK|timestamp|datetime"
                ):
                    self._insert_audit_event(
                        entity_type="scan_artifact",
                        entity_id=1,
                        event_type="scan_recorded",
                        payload_json="{}",
                        recorded_at=recorded_at,
                    )

    def _require_tables(self, *table_names: str) -> None:
        actual_tables = {
            row["name"]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = [name for name in table_names if name not in actual_tables]
        if missing_tables:
            requirement = self._testMethodName.removeprefix("test_").replace("_", " ")
            self.fail(
                "Cannot verify "
                f"{requirement} because initialize_schema() is still missing "
                f"required table(s): {missing_tables}"
            )

    def _insert_student(self, student_key: str | None) -> int:
        # `student_key` is the v0 identity contract from the README. `display_name`
        # is treated as optional helper metadata unless the schema explicitly adds it.
        student_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(students)")
        }
        if "display_name" in student_columns:
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
        else:
            cursor = self.connection.execute(
                """
                INSERT INTO students (
                    student_key
                )
                VALUES (?)
                """,
                (student_key,),
            )
        return int(cursor.lastrowid)

    def _insert_template_version(
        self,
        slug: str | None = "stoichiometry-q1",
        version: int | None = 1,
        source_yaml: str | None | object = _DEFAULT,
    ) -> int:
        if source_yaml is _DEFAULT:
            source_yaml = f"slug: {slug}"
        cursor = self.connection.execute(
            """
            INSERT INTO template_versions (
                slug,
                version,
                source_yaml
            )
            VALUES (?, ?, ?)
            """,
            (slug, version, source_yaml),
        )
        return int(cursor.lastrowid)

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
            INSERT INTO exam_definitions (
                slug,
                version,
                title,
                template_version_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (slug, version, title, template_version_id),
        )
        return int(cursor.lastrowid)

    def _insert_exam_instance(
        self,
        student_key: str = "student-001",
        exam_slug: str = "chem101-midterm",
        attempt_number: int = 1,
        opaque_instance_code: str = "inst-001",
    ) -> int:
        student_id = self._insert_student(student_key)
        exam_definition_id = self._insert_exam_definition(exam_slug)
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
            VALUES (?, ?, ?, ?)
            """,
            (
                exam_definition_id,
                student_id,
                attempt_number,
                opaque_instance_code,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_exam_page(
        self,
        exam_instance_id: int,
        page_number: int | None,
        fallback_page_code: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO exam_pages (
                exam_instance_id,
                page_number,
                fallback_page_code
            )
            VALUES (?, ?, ?)
            """,
            (exam_instance_id, page_number, fallback_page_code),
        )
        return int(cursor.lastrowid)

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
            VALUES (?, ?, ?, ?)
            """,
            (sha256, original_filename, status, failure_reason),
        )
        return int(cursor.lastrowid)

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
            VALUES (?, ?, ?, ?)
            """,
            (exam_instance_id, status, score_points, max_points),
        )
        return int(cursor.lastrowid)

    def _insert_audit_event(
        self,
        entity_type: str | None,
        entity_id: int | None,
        event_type: str | None,
        payload_json: str | None,
        recorded_at: str | None | object = _DEFAULT,
    ) -> int:
        if recorded_at is _DEFAULT:
            cursor = self.connection.execute(
                """
                INSERT INTO audit_events (
                    entity_type,
                    entity_id,
                    event_type,
                    payload_json
                )
                VALUES (?, ?, ?, ?)
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
                VALUES (?, ?, ?, ?, ?)
                """,
                (entity_type, entity_id, event_type, payload_json, recorded_at),
            )
        return int(cursor.lastrowid)

    def _assert_row_is_immutable(
        self,
        table_name: str,
        key_columns: dict[str, object],
        assignments: dict[str, object],
        expected_row: dict[str, object],
    ) -> None:
        set_clause = ", ".join(f"{column} = ?" for column in assignments)
        where_clause = " AND ".join(f"{column} = ?" for column in key_columns)
        values = list(assignments.values()) + list(key_columns.values())
        exact_where_clause = " AND ".join(f"{column} = ?" for column in expected_row)
        assignment_where_clause = " AND ".join(f"{column} = ?" for column in assignments)
        stable_key_columns = {
            column: value
            for column, value in key_columns.items()
            if column not in assignments
        }
        stable_scope_clause = " AND ".join(
            f"{column} = ?" for column in stable_key_columns
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
            SELECT COUNT(*)
            FROM {table_name}
            WHERE {mutation_scope_predicate}
              AND NOT ({exact_where_clause})
            """,
            assignment_filter_values,
        ).fetchone()[0]

        try:
            self.connection.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}",
                values,
            )
        except sqlite3.DatabaseError:
            # The contract is about preserving the original versioned row, not the
            # particular mechanism SQLite uses to reject or absorb an update attempt.
            pass

        exact_row_count = self.connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            WHERE {exact_where_clause}
            """,
            tuple(expected_row.values()),
        ).fetchone()[0]
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
            SELECT COUNT(*)
            FROM {table_name}
            WHERE {mutation_scope_predicate}
              AND NOT ({exact_where_clause})
            """,
            assignment_filter_values,
        ).fetchone()[0]
        self.assertEqual(
            unexpected_mutated_count,
            preexisting_mutated_count,
            f"{table_name} should not gain side-effect rows matching {assignments}.",
        )

    def _parse_recorded_at(self, recorded_at: str) -> datetime:
        normalized = recorded_at.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        if "T" not in normalized and len(normalized) >= 19 and normalized[10] == " ":
            normalized = normalized.replace(" ", "T", 1)

        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sha256(self, value: int) -> str:
        return f"{value:064x}"


if __name__ == "__main__":
    unittest.main()
