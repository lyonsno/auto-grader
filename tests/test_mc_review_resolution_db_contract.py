"""Postgres-backed contract tests for durable MC review resolutions.

The review-override surface already exists in-memory via
``auto_grader.mc_review_override``. This suite pins the next step: recording
those human resolutions durably in the DB without mutating away the original
machine-scored outcome.
"""

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


def _build_review_required_manifest() -> dict:
    checksum = hashlib.sha256(b"page-1.png").hexdigest()
    return {
        "opaque_instance_code": "TEST-INSTANCE-001",
        "expected_page_codes": ["PAGE-1-CODE"],
        "scan_results": [
            {
                "scan_id": "page-1.png",
                "checksum": checksum,
                "status": "matched",
                "failure_reason": None,
                "page_number": 1,
                "fallback_page_code": "PAGE-1-CODE",
                "scored_questions": {
                    "mc-1": {
                        "question_id": "mc-1",
                        "status": "multiple_marked",
                        "is_correct": False,
                        "review_required": True,
                        "marked_bubble_labels": ["A", "B"],
                        "resolved_bubble_labels": ["A", "B"],
                        "correct_bubble_label": "B",
                        "correct_choice_key": "choice-b",
                        "marked_choice_keys": ["choice-a", "choice-b"],
                    },
                },
            },
        ],
        "summary": {
            "total_scans": 1,
            "matched": 1,
            "unmatched": 0,
            "ambiguous": 0,
            "review_required": 1,
        },
    }


def _resolved_correct() -> dict[str, dict]:
    return {
        "mc-1": {
            "question_id": "mc-1",
            "status": "correct",
            "is_correct": True,
            "review_required": False,
            "marked_bubble_labels": ["A", "B"],
            "resolved_bubble_labels": ["B"],
            "correct_bubble_label": "B",
            "correct_choice_key": "choice-b",
            "marked_choice_keys": ["choice-b"],
            "override": {
                "original_status": "multiple_marked",
                "resolved_bubble_label": "B",
            },
        },
    }


def _resolved_blank() -> dict[str, dict]:
    return {
        "mc-1": {
            "question_id": "mc-1",
            "status": "blank",
            "is_correct": False,
            "review_required": False,
            "marked_bubble_labels": ["A", "B"],
            "resolved_bubble_labels": [],
            "correct_bubble_label": "B",
            "correct_choice_key": "choice-b",
            "marked_choice_keys": [],
            "override": {
                "original_status": "multiple_marked",
                "resolved_bubble_label": None,
            },
        },
    }


def _load_persist_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db
    except (ModuleNotFoundError, ImportError):
        test_case.fail(
            "Add `auto_grader.mc_review_db.persist_mc_review_resolutions_to_db(...)` "
            "so human MC review decisions can be stored durably in Postgres."
        )
    return persist_mc_review_resolutions_to_db


class McReviewResolutionDbContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC review-resolution DB contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the MC review-resolution DB contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "MC review-resolution DB contract tests require psycopg in the active environment."
            )

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_review_db_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
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
                f"MC review-resolution DB contract test setup failed: {exc}"
            ) from exc

        initialize_schema(self.connection)
        self._seed_exam_instance()
        self._seed_scan_session()

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
        self.student_id = student_row["id"]

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
            (ed_row["id"], self.student_id),
        )
        ei_row = self.connection.execute(
            "SELECT id FROM exam_instances WHERE opaque_instance_code = 'TEST-INSTANCE-001'"
        ).fetchone()
        self.exam_instance_id = ei_row["id"]

    def _seed_scan_session(self) -> None:
        from auto_grader.mc_scan_db import persist_scan_session_to_db

        manifest = _build_review_required_manifest()
        result = persist_scan_session_to_db(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        self.mc_scan_session_id = result["mc_scan_session_id"]
        page_row = self.connection.execute(
            "SELECT id FROM mc_scan_pages WHERE mc_scan_session_id = %s AND scan_id = 'page-1.png'",
            (self.mc_scan_session_id,),
        ).fetchone()
        self.mc_scan_page_id = page_row["id"]
        outcome_row = self.connection.execute(
            "SELECT id FROM mc_question_outcomes WHERE mc_scan_page_id = %s AND question_id = 'mc-1'",
            (self.mc_scan_page_id,),
        ).fetchone()
        self.mc_question_outcome_id = outcome_row["id"]

    def test_persist_review_resolution_creates_row_and_audit_event(self) -> None:
        persist = _load_persist_module(self)

        result = persist(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct(),
            connection=self.connection,
        )

        self.assertEqual(result["created"], 1)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["unchanged"], 0)

        resolution = self.connection.execute(
            "SELECT * FROM mc_review_resolutions WHERE mc_question_outcome_id = %s",
            (self.mc_question_outcome_id,),
        ).fetchone()
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["original_status"], "multiple_marked")
        self.assertEqual(resolution["resolved_bubble_label"], "B")
        self.assertEqual(resolution["final_status"], "correct")
        self.assertTrue(resolution["final_is_correct"])

        machine_outcome = self.connection.execute(
            "SELECT status, is_correct, review_required FROM mc_question_outcomes WHERE id = %s",
            (self.mc_question_outcome_id,),
        ).fetchone()
        self.assertEqual(machine_outcome["status"], "multiple_marked")
        self.assertFalse(machine_outcome["is_correct"])
        self.assertTrue(machine_outcome["review_required"])

        audit_events = self.connection.execute(
            "SELECT * FROM audit_events WHERE entity_type = 'mc_review_resolution'",
        ).fetchall()
        self.assertEqual(len(audit_events), 1)
        self.assertEqual(audit_events[0]["event_type"], "created")

    def test_same_review_resolution_is_idempotent(self) -> None:
        persist = _load_persist_module(self)

        first = persist(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct(),
            connection=self.connection,
        )
        second = persist(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct(),
            connection=self.connection,
        )

        self.assertEqual(first["created"], 1)
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["updated"], 0)
        self.assertEqual(second["unchanged"], 1)

        resolution_count = self.connection.execute(
            "SELECT count(*) AS n FROM mc_review_resolutions",
        ).fetchone()["n"]
        self.assertEqual(resolution_count, 1)

        audit_count = self.connection.execute(
            "SELECT count(*) AS n FROM audit_events WHERE entity_type = 'mc_review_resolution'",
        ).fetchone()["n"]
        self.assertEqual(
            audit_count,
            1,
            "Re-applying the same human decision should not duplicate audit history.",
        )

    def test_changed_review_resolution_updates_current_row_and_appends_audit_history(self) -> None:
        persist = _load_persist_module(self)

        first = persist(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct(),
            connection=self.connection,
        )
        second = persist(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_blank(),
            connection=self.connection,
        )

        self.assertEqual(first["created"], 1)
        self.assertEqual(second["updated"], 1)

        resolution = self.connection.execute(
            "SELECT * FROM mc_review_resolutions WHERE mc_question_outcome_id = %s",
            (self.mc_question_outcome_id,),
        ).fetchone()
        self.assertEqual(resolution["resolved_bubble_label"], None)
        self.assertEqual(resolution["final_status"], "blank")
        self.assertFalse(resolution["final_is_correct"])

        audit_events = self.connection.execute(
            "SELECT event_type FROM audit_events "
            "WHERE entity_type = 'mc_review_resolution' "
            "ORDER BY id",
        ).fetchall()
        self.assertEqual(
            [row["event_type"] for row in audit_events],
            ["created", "updated"],
        )


if __name__ == "__main__":
    unittest.main()
