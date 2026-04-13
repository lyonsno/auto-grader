"""Postgres-backed contract tests for the current-final MC results read model."""

from __future__ import annotations

import hashlib
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


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


def _build_question(
    *,
    question_id: str,
    status: str,
    is_correct: bool,
    review_required: bool,
    marked_bubble_labels: list[str],
    resolved_bubble_labels: list[str],
    correct_bubble_label: str = "B",
    correct_choice_key: str = "choice-b",
    marked_choice_keys: list[str] | None = None,
) -> dict:
    if marked_choice_keys is None:
        marked_choice_keys = ["choice-b"] if resolved_bubble_labels == ["B"] else []
    return {
        "question_id": question_id,
        "status": status,
        "is_correct": is_correct,
        "review_required": review_required,
        "marked_bubble_labels": marked_bubble_labels,
        "resolved_bubble_labels": resolved_bubble_labels,
        "correct_bubble_label": correct_bubble_label,
        "correct_choice_key": correct_choice_key,
        "marked_choice_keys": marked_choice_keys,
    }


def _build_matched_scan_result(
    *,
    scan_id: str,
    page_number: int,
    fallback_page_code: str,
    scored_questions: dict[str, dict],
) -> dict:
    return {
        "scan_id": scan_id,
        "checksum": hashlib.sha256(scan_id.encode()).hexdigest(),
        "status": "matched",
        "failure_reason": None,
        "page_number": page_number,
        "fallback_page_code": fallback_page_code,
        "scored_questions": scored_questions,
    }


def _build_unmatched_scan_result(
    *,
    scan_id: str,
    failure_reason: str = "No QR code detected",
) -> dict:
    return {
        "scan_id": scan_id,
        "checksum": hashlib.sha256(scan_id.encode()).hexdigest(),
        "status": "unmatched",
        "failure_reason": failure_reason,
    }


def _build_manifest(*, scan_results: list[dict]) -> dict:
    summary = {
        "total_scans": len(scan_results),
        "matched": sum(1 for scan in scan_results if scan["status"] == "matched"),
        "unmatched": sum(1 for scan in scan_results if scan["status"] == "unmatched"),
        "ambiguous": sum(1 for scan in scan_results if scan["status"] == "ambiguous"),
        "review_required": sum(
            1
            for scan in scan_results
            if scan["status"] == "matched"
            and any(question["review_required"] for question in scan.get("scored_questions", {}).values())
        ),
    }
    return {
        "opaque_instance_code": "TEST-INSTANCE-001",
        "expected_page_codes": ["PAGE-1-CODE", "PAGE-2-CODE"],
        "scan_results": scan_results,
        "summary": summary,
    }


def _resolved_correct_for(question_id: str, *, bubble_label: str = "B") -> dict[str, dict]:
    return {
        question_id: {
            "question_id": question_id,
            "status": "correct",
            "is_correct": True,
            "review_required": False,
            "marked_bubble_labels": ["A", bubble_label],
            "resolved_bubble_labels": [bubble_label],
            "correct_bubble_label": bubble_label,
            "correct_choice_key": f"choice-{bubble_label.lower()}",
            "marked_choice_keys": [f"choice-{bubble_label.lower()}"],
            "override": {
                "original_status": "multiple_marked",
                "resolved_bubble_label": bubble_label,
            },
        },
    }


def _load_read_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_results_db import read_current_final_mc_results_from_db
    except (ModuleNotFoundError, ImportError):
        test_case.fail(
            "Add `auto_grader.mc_results_db.read_current_final_mc_results_from_db(...)` "
            "so callers can read one authoritative current-final MC truth surface "
            "from persisted machine outcomes plus review resolutions."
        )
    return read_current_final_mc_results_from_db


class McFinalResultsDbContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC final-results DB contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the MC final-results DB contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "MC final-results DB contract tests require psycopg in the active environment."
            )

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_final_results_db_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema_name))
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
                f"MC final-results DB contract test setup failed: {exc}"
            ) from exc

        initialize_schema(self.connection)
        self._seed_exam_instance()

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

    def _seed_exam_instance(self) -> None:
        self.connection.execute(
            "INSERT INTO students (student_key) VALUES ('student-test-001')"
        )
        student_row = self.connection.execute(
            "SELECT id FROM students WHERE student_key = 'student-test-001'"
        ).fetchone()

        self.connection.execute(
            "INSERT INTO template_versions (slug, version, source_yaml) "
            "VALUES ('test-template', 1, 'slug: test')"
        )
        tv_row = self.connection.execute(
            "SELECT id FROM template_versions WHERE slug = 'test-template'"
        ).fetchone()

        self.connection.execute(
            "INSERT INTO exam_definitions (slug, version, title, template_version_id) "
            "VALUES ('test-exam', 1, 'Test Exam', %s)",
            (tv_row["id"],),
        )
        ed_row = self.connection.execute(
            "SELECT id FROM exam_definitions WHERE slug = 'test-exam'"
        ).fetchone()

        self.connection.execute(
            "INSERT INTO exam_instances "
            "(exam_definition_id, student_id, attempt_number, opaque_instance_code) "
            "VALUES (%s, %s, 1, 'TEST-INSTANCE-001')",
            (ed_row["id"], student_row["id"]),
        )
        exam_instance_row = self.connection.execute(
            "SELECT id FROM exam_instances WHERE opaque_instance_code = 'TEST-INSTANCE-001'"
        ).fetchone()
        self.exam_instance_id = exam_instance_row["id"]

    def _persist_manifest(self, manifest: dict) -> int:
        from auto_grader.mc_scan_db import persist_scan_session_to_db

        result = persist_scan_session_to_db(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        return result["mc_scan_session_id"]

    def test_read_current_final_results_fails_when_exam_instance_has_no_scan_session(self) -> None:
        read_current = _load_read_module(self)

        with self.assertRaisesRegex(LookupError, r"exam_instance_id.*TEST-INSTANCE-001|exam_instance_id.*no MC scan session|No MC scan session"):
            read_current(
                exam_instance_id=self.exam_instance_id,
                connection=self.connection,
            )

    def test_read_current_final_results_overlays_current_session_review_resolution(self) -> None:
        read_current = _load_read_module(self)
        from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db

        manifest = _build_manifest(
            scan_results=[
                _build_matched_scan_result(
                    scan_id="page-1.png",
                    page_number=1,
                    fallback_page_code="PAGE-1-CODE",
                    scored_questions={
                        "mc-1": _build_question(
                            question_id="mc-1",
                            status="multiple_marked",
                            is_correct=False,
                            review_required=True,
                            marked_bubble_labels=["A", "B"],
                            resolved_bubble_labels=["A", "B"],
                            marked_choice_keys=["choice-a", "choice-b"],
                        ),
                    },
                ),
                _build_unmatched_scan_result(scan_id="page-2.png"),
            ]
        )
        session_id = self._persist_manifest(manifest)
        persist_mc_review_resolutions_to_db(
            mc_scan_session_id=session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct_for("mc-1"),
            connection=self.connection,
        )

        result = read_current(
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertEqual(result["exam_instance_id"], self.exam_instance_id)
        self.assertEqual(result["mc_scan_session_id"], session_id)
        self.assertEqual(result["session_ordinal"], 1)
        self.assertEqual(result["summary"]["matched"], 1)
        self.assertEqual(result["summary"]["unmatched"], 1)
        self.assertEqual(result["summary"]["unresolved_review_required"], 0)
        self.assertEqual(result["review_required_question_ids"], [])

        pages_by_scan_id = {
            page["scan_id"]: page for page in result["scan_pages"]
        }
        self.assertEqual(pages_by_scan_id["page-2.png"]["status"], "unmatched")
        self.assertEqual(
            pages_by_scan_id["page-2.png"]["failure_reason"],
            "No QR code detected",
        )

        question = result["question_results"]["mc-1"]
        self.assertEqual(question["status"], "correct")
        self.assertTrue(question["is_correct"])
        self.assertFalse(question["review_required"])
        self.assertEqual(question["source"], "review_resolution")
        self.assertEqual(question["machine_status"], "multiple_marked")
        self.assertTrue(question["machine_review_required"])
        self.assertEqual(question["final_resolved_bubble_labels"], ["B"])
        self.assertEqual(question["resolution"]["resolved_bubble_label"], "B")
        self.assertEqual(question["scan_id"], "page-1.png")
        self.assertEqual(question["page_number"], 1)

    def test_read_current_final_results_keeps_latest_session_and_does_not_reuse_stale_resolution(self) -> None:
        read_current = _load_read_module(self)
        from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db

        first_session_id = self._persist_manifest(
            _build_manifest(
                scan_results=[
                    _build_matched_scan_result(
                        scan_id="page-1.png",
                        page_number=1,
                        fallback_page_code="PAGE-1-CODE",
                        scored_questions={
                            "mc-1": _build_question(
                                question_id="mc-1",
                                status="multiple_marked",
                                is_correct=False,
                                review_required=True,
                                marked_bubble_labels=["A", "B"],
                                resolved_bubble_labels=["A", "B"],
                                marked_choice_keys=["choice-a", "choice-b"],
                            ),
                        },
                    ),
                ]
            )
        )
        persist_mc_review_resolutions_to_db(
            mc_scan_session_id=first_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct_for("mc-1"),
            connection=self.connection,
        )

        second_session_id = self._persist_manifest(
            _build_manifest(
                scan_results=[
                    _build_matched_scan_result(
                        scan_id="page-1.png",
                        page_number=1,
                        fallback_page_code="PAGE-1-CODE",
                        scored_questions={
                            "mc-1": _build_question(
                                question_id="mc-1",
                                status="multiple_marked",
                                is_correct=False,
                                review_required=True,
                                marked_bubble_labels=["B", "C"],
                                resolved_bubble_labels=["B", "C"],
                                marked_choice_keys=["choice-b", "choice-c"],
                            ),
                        },
                    ),
                ]
            )
        )

        result = read_current(
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertEqual(result["mc_scan_session_id"], second_session_id)
        self.assertEqual(result["session_ordinal"], 2)
        self.assertEqual(result["review_required_question_ids"], ["mc-1"])

        question = result["question_results"]["mc-1"]
        self.assertEqual(question["status"], "multiple_marked")
        self.assertFalse(question["is_correct"])
        self.assertTrue(question["review_required"])
        self.assertEqual(question["source"], "machine")
        self.assertEqual(question["machine_status"], "multiple_marked")
        self.assertIsNone(question["resolution"])
        self.assertEqual(question["marked_bubble_labels"], ["B", "C"])
        self.assertEqual(question["final_resolved_bubble_labels"], ["B", "C"])

