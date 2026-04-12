"""Contract tests for persisting MC scan session results into the database.

The DB persistence layer takes a session manifest (as produced by
``mc_scan_session.persist_scan_session``) plus an exam_instance_id and writes:

- One ``mc_scan_sessions`` row per session (idempotent on manifest fingerprint)
- One ``mc_scan_pages`` row per scan in the session
- One ``mc_question_outcomes`` row per scored question on matched pages
- Divergence detection when a new session for the same exam instance carries
  different results than a prior session

Re-running the same manifest for the same exam instance is idempotent.
A new manifest for the same exam instance creates a superseding session
with linkage back to the prior one.
"""

from __future__ import annotations

import hashlib
import json
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


def _build_matched_manifest(
    *,
    opaque_instance_code: str = "TEST-INSTANCE-001",
    scan_id: str = "page-1.png",
    checksum: str | None = None,
    question_status: str = "correct",
    review_required: bool = False,
) -> dict:
    """Build a minimal but realistic session manifest for testing."""
    if checksum is None:
        checksum = hashlib.sha256(scan_id.encode()).hexdigest()
    return {
        "opaque_instance_code": opaque_instance_code,
        "expected_page_codes": ["PAGE-1-CODE"],
        "scan_results": [
            {
                "scan_id": scan_id,
                "checksum": checksum,
                "status": "matched",
                "failure_reason": None,
                "page_number": 1,
                "fallback_page_code": "PAGE-1-CODE",
                "scored_questions": {
                    "mc-1": {
                        "question_id": "mc-1",
                        "status": question_status,
                        "is_correct": question_status == "correct",
                        "review_required": review_required,
                        "marked_bubble_labels": ["B"],
                        "correct_bubble_label": "B",
                        "resolved_bubble_labels": ["B"],
                        "marked_choice_keys": ["choice-b"],
                        "correct_choice_key": "choice-b",
                    },
                },
            },
        ],
        "summary": {
            "total_scans": 1,
            "matched": 1,
            "unmatched": 0,
            "ambiguous": 0,
            "review_required": 1 if review_required else 0,
        },
    }


def _build_unmatched_manifest(
    *,
    opaque_instance_code: str = "TEST-INSTANCE-001",
    scan_id: str = "blank.png",
    checksum: str | None = None,
) -> dict:
    """Build a manifest with one unmatched scan."""
    if checksum is None:
        checksum = hashlib.sha256(scan_id.encode()).hexdigest()
    return {
        "opaque_instance_code": opaque_instance_code,
        "expected_page_codes": ["PAGE-1-CODE"],
        "scan_results": [
            {
                "scan_id": scan_id,
                "checksum": checksum,
                "status": "unmatched",
                "failure_reason": "No QR code detected",
            },
        ],
        "summary": {
            "total_scans": 1,
            "matched": 0,
            "unmatched": 1,
            "ambiguous": 0,
            "review_required": 0,
        },
    }


def _manifest_fingerprint(manifest: dict) -> str:
    """Compute the content fingerprint of a manifest (same logic the module should use)."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_persist_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_scan_db import persist_scan_session_to_db
    except (ModuleNotFoundError, ImportError):
        test_case.fail(
            "Add `auto_grader.mc_scan_db.persist_scan_session_to_db(...)` so MC "
            "scan session results can be persisted into the database."
        )
    return persist_scan_session_to_db


class McScanDbContractTests(unittest.TestCase):
    """Postgres-backed contract tests for MC scan session DB persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC scan DB contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the MC scan DB contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "MC scan DB contract tests require psycopg in the active environment."
            )

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_scan_db_{uuid.uuid4().hex}"
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
                f"MC scan DB contract test setup failed: {exc}"
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
        """Create the prerequisite rows so we have an exam_instance_id to use."""
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

    # ------------------------------------------------------------------
    # Core persistence contract
    # ------------------------------------------------------------------

    def test_persist_creates_session_row(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_matched_manifest()

        result = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertIn("mc_scan_session_id", result)
        row = self.connection.execute(
            "SELECT * FROM mc_scan_sessions WHERE id = %s",
            (result["mc_scan_session_id"],),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["exam_instance_id"], self.exam_instance_id)
        self.assertEqual(row["manifest_fingerprint"], _manifest_fingerprint(manifest))
        self.assertEqual(row["session_ordinal"], 1)
        self.assertIsNone(row["supersedes_session_id"])

    def test_persist_creates_scan_page_rows(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_matched_manifest()

        result = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT * FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (result["mc_scan_session_id"],),
        ).fetchall()
        self.assertEqual(len(pages), 1)
        page = pages[0]
        self.assertEqual(page["scan_id"], "page-1.png")
        self.assertEqual(page["status"], "matched")
        self.assertIsNone(page["failure_reason"])
        self.assertEqual(page["page_number"], 1)
        self.assertEqual(page["fallback_page_code"], "PAGE-1-CODE")

    def test_persist_creates_question_outcome_rows(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_matched_manifest()

        result = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT id FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (result["mc_scan_session_id"],),
        ).fetchall()
        page_id = pages[0]["id"]

        outcomes = self.connection.execute(
            "SELECT * FROM mc_question_outcomes WHERE mc_scan_page_id = %s",
            (page_id,),
        ).fetchall()
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertEqual(outcome["question_id"], "mc-1")
        self.assertEqual(outcome["status"], "correct")
        self.assertTrue(outcome["is_correct"])
        self.assertFalse(outcome["review_required"])
        self.assertEqual(outcome["correct_bubble_label"], "B")

    def test_persist_records_unmatched_scans_without_question_outcomes(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_unmatched_manifest()

        result = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT * FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (result["mc_scan_session_id"],),
        ).fetchall()
        self.assertEqual(len(pages), 1)
        page = pages[0]
        self.assertEqual(page["status"], "unmatched")
        self.assertEqual(page["failure_reason"], "No QR code detected")
        self.assertIsNone(page["page_number"])

        outcomes = self.connection.execute(
            "SELECT * FROM mc_question_outcomes WHERE mc_scan_page_id = %s",
            (page["id"],),
        ).fetchall()
        self.assertEqual(len(outcomes), 0)

    # ------------------------------------------------------------------
    # Idempotency contract
    # ------------------------------------------------------------------

    def test_same_manifest_is_idempotent(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_matched_manifest()

        result_1 = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        result_2 = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        # Same session ID returned — no duplication.
        self.assertEqual(
            result_1["mc_scan_session_id"],
            result_2["mc_scan_session_id"],
        )

        # Only one session row exists.
        count = self.connection.execute(
            "SELECT count(*) AS n FROM mc_scan_sessions "
            "WHERE exam_instance_id = %s",
            (self.exam_instance_id,),
        ).fetchone()["n"]
        self.assertEqual(count, 1)

    # ------------------------------------------------------------------
    # Supersession / divergence contract
    # ------------------------------------------------------------------

    def test_different_manifest_creates_superseding_session(self) -> None:
        persist = _load_persist_module(self)
        manifest_1 = _build_matched_manifest(question_status="correct")
        manifest_2 = _build_matched_manifest(question_status="incorrect")

        result_1 = persist(
            manifest=manifest_1,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        result_2 = persist(
            manifest=manifest_2,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertNotEqual(
            result_1["mc_scan_session_id"],
            result_2["mc_scan_session_id"],
        )

        session_2 = self.connection.execute(
            "SELECT * FROM mc_scan_sessions WHERE id = %s",
            (result_2["mc_scan_session_id"],),
        ).fetchone()
        self.assertEqual(session_2["session_ordinal"], 2)
        self.assertEqual(
            session_2["supersedes_session_id"],
            result_1["mc_scan_session_id"],
        )

    def test_superseding_session_flags_divergence_on_scan_pages(self) -> None:
        persist = _load_persist_module(self)
        # Same scan_id and checksum but different outcomes.
        manifest_1 = _build_matched_manifest(question_status="correct")
        manifest_2 = _build_matched_manifest(question_status="incorrect")

        persist(
            manifest=manifest_1,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        result_2 = persist(
            manifest=manifest_2,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT * FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (result_2["mc_scan_session_id"],),
        ).fetchall()
        self.assertEqual(len(pages), 1)
        self.assertTrue(
            pages[0]["divergence_detected"],
            "When the same scan appears in a superseding session with different "
            "outcomes, divergence_detected must be True.",
        )

    def test_superseding_session_no_divergence_when_outcomes_match(self) -> None:
        persist = _load_persist_module(self)
        # Different manifest fingerprint (e.g. extra scan) but same scan outcomes
        # for the overlapping scan.
        manifest_1 = _build_matched_manifest()
        # Build a manifest with a second scan added — different fingerprint,
        # but the original scan's outcomes are identical.
        manifest_2 = _build_matched_manifest()
        manifest_2["scan_results"].append({
            "scan_id": "extra-blank.png",
            "checksum": hashlib.sha256(b"extra").hexdigest(),
            "status": "unmatched",
            "failure_reason": "No QR code detected",
        })
        manifest_2["summary"]["total_scans"] = 2
        manifest_2["summary"]["unmatched"] = 1

        persist(
            manifest=manifest_1,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        result_2 = persist(
            manifest=manifest_2,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT * FROM mc_scan_pages "
            "WHERE mc_scan_session_id = %s AND scan_id = 'page-1.png'",
            (result_2["mc_scan_session_id"],),
        ).fetchall()
        self.assertEqual(len(pages), 1)
        self.assertFalse(
            pages[0]["divergence_detected"],
            "When a superseding session carries the same outcomes for an "
            "overlapping scan, divergence_detected must be False.",
        )

    # ------------------------------------------------------------------
    # Review-required flag propagation
    # ------------------------------------------------------------------

    def test_review_required_flag_persisted_on_question_outcomes(self) -> None:
        persist = _load_persist_module(self)
        manifest = _build_matched_manifest(
            question_status="multiple_marked",
            review_required=True,
        )

        result = persist(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        pages = self.connection.execute(
            "SELECT id FROM mc_scan_pages WHERE mc_scan_session_id = %s",
            (result["mc_scan_session_id"],),
        ).fetchall()
        outcomes = self.connection.execute(
            "SELECT * FROM mc_question_outcomes WHERE mc_scan_page_id = %s",
            (pages[0]["id"],),
        ).fetchall()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["review_required"])
        self.assertEqual(outcomes[0]["status"], "multiple_marked")


    # ------------------------------------------------------------------
    # Atomicity contract
    # ------------------------------------------------------------------

    def test_partial_failure_does_not_leave_orphan_session_row(self) -> None:
        persist = _load_persist_module(self)
        # Build a manifest with a question outcome that will violate a DB
        # constraint (question_id is blank, which violates nonblank CHECK).
        manifest = _build_matched_manifest()
        manifest["scan_results"][0]["scored_questions"]["mc-1"]["question_id"] = ""
        # Also put an empty string as the dict key so the module uses it.
        manifest["scan_results"][0]["scored_questions"] = {
            "": manifest["scan_results"][0]["scored_questions"]["mc-1"]
        }

        with self.assertRaises(Exception):
            persist(
                manifest=manifest,
                exam_instance_id=self.exam_instance_id,
                connection=self.connection,
            )

        # No session row should exist — the transaction must have rolled back.
        count = self.connection.execute(
            "SELECT count(*) AS n FROM mc_scan_sessions "
            "WHERE exam_instance_id = %s",
            (self.exam_instance_id,),
        ).fetchone()["n"]
        self.assertEqual(
            count,
            0,
            "A failed persist must not leave orphan session rows in the database. "
            "All mutations must be atomic.",
        )


if __name__ == "__main__":
    unittest.main()
