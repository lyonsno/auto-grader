"""Contract tests for the professor-facing MC workflow entrypoint.

Pure-logic tests run without a database. DB-backed integration tests require
TEST_DATABASE_URL pointing at a disposable Postgres instance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
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


# ---------------------------------------------------------------------------
# Synthetic current_results fixture for pure tests
# ---------------------------------------------------------------------------

def _synthetic_current_results() -> dict:
    """Mimics the shape returned by read_current_final_mc_results_from_db."""
    return {
        "exam_instance_id": 1,
        "mc_scan_session_id": 10,
        "session_ordinal": 1,
        "scan_pages": [
            {
                "scan_id": "page-1.png",
                "checksum": hashlib.sha256(b"page-1.png").hexdigest(),
                "status": "matched",
                "failure_reason": None,
                "page_number": 1,
                "fallback_page_code": "PAGE-1-CODE",
                "divergence_detected": False,
                "question_ids": ["mc-1", "mc-2", "mc-3"],
            },
        ],
        "question_results": {
            "mc-1": {
                "question_id": "mc-1",
                "scan_id": "page-1.png",
                "page_number": 1,
                "status": "multiple_marked",
                "is_correct": False,
                "review_required": True,
                "source": "machine",
                "machine_status": "multiple_marked",
                "machine_is_correct": False,
                "machine_review_required": True,
                "marked_bubble_labels": ["A", "B"],
                "machine_resolved_bubble_labels": ["A", "B"],
                "final_resolved_bubble_labels": ["A", "B"],
                "correct_bubble_label": "B",
                "correct_choice_key": "choice-b",
                "resolution": None,
            },
            "mc-2": {
                "question_id": "mc-2",
                "scan_id": "page-1.png",
                "page_number": 1,
                "status": "correct",
                "is_correct": True,
                "review_required": False,
                "source": "machine",
                "machine_status": "correct",
                "machine_is_correct": True,
                "machine_review_required": False,
                "marked_bubble_labels": ["C"],
                "machine_resolved_bubble_labels": ["C"],
                "final_resolved_bubble_labels": ["C"],
                "correct_bubble_label": "C",
                "correct_choice_key": "choice-c",
                "resolution": None,
            },
            "mc-3": {
                "question_id": "mc-3",
                "scan_id": "page-1.png",
                "page_number": 1,
                "status": "ambiguous_mark",
                "is_correct": False,
                "review_required": True,
                "source": "machine",
                "machine_status": "ambiguous_mark",
                "machine_is_correct": False,
                "machine_review_required": True,
                "marked_bubble_labels": ["A"],
                "machine_resolved_bubble_labels": ["A"],
                "final_resolved_bubble_labels": ["A"],
                "correct_bubble_label": "A",
                "correct_choice_key": "choice-a",
                "resolution": None,
            },
        },
        "review_required_question_ids": ["mc-1", "mc-3"],
        "summary": {
            "matched": 1,
            "unmatched": 0,
            "ambiguous": 0,
            "unresolved_review_required": 2,
            "correct": 1,
            "incorrect": 0,
            "blank": 0,
        },
    }


# ---------------------------------------------------------------------------
# Pure logic tests (no DB)
# ---------------------------------------------------------------------------

class BuildReviewResolutionsTests(unittest.TestCase):

    def _load(self):
        from auto_grader.mc_workflow import build_review_resolutions_from_simple_map
        return build_review_resolutions_from_simple_map

    def test_correct_resolution_produces_expected_shape(self) -> None:
        build = self._load()
        result = build(
            simple_resolutions={"mc-1": "B"},
            current_results=_synthetic_current_results(),
        )

        self.assertIn("page-1.png", result)
        self.assertIn("mc-1", result["page-1.png"])

        resolved = result["page-1.png"]["mc-1"]
        self.assertEqual(resolved["status"], "correct")
        self.assertTrue(resolved["is_correct"])
        self.assertFalse(resolved["review_required"])
        self.assertEqual(resolved["override"]["original_status"], "multiple_marked")
        self.assertEqual(resolved["override"]["resolved_bubble_label"], "B")

    def test_incorrect_resolution(self) -> None:
        build = self._load()
        result = build(
            simple_resolutions={"mc-1": "A"},
            current_results=_synthetic_current_results(),
        )
        resolved = result["page-1.png"]["mc-1"]
        self.assertEqual(resolved["status"], "incorrect")
        self.assertFalse(resolved["is_correct"])

    def test_blank_resolution(self) -> None:
        build = self._load()
        result = build(
            simple_resolutions={"mc-3": None},
            current_results=_synthetic_current_results(),
        )
        resolved = result["page-1.png"]["mc-3"]
        self.assertEqual(resolved["status"], "blank")
        self.assertFalse(resolved["is_correct"])
        self.assertIsNone(resolved["override"]["resolved_bubble_label"])

    def test_multiple_resolutions_grouped_by_scan_id(self) -> None:
        build = self._load()
        result = build(
            simple_resolutions={"mc-1": "B", "mc-3": None},
            current_results=_synthetic_current_results(),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result["page-1.png"]), 2)

    def test_unknown_question_raises_key_error(self) -> None:
        build = self._load()
        with self.assertRaises(KeyError):
            build(
                simple_resolutions={"mc-999": "A"},
                current_results=_synthetic_current_results(),
            )


class RenderResultsCsvTests(unittest.TestCase):

    def _load(self):
        from auto_grader.mc_workflow import render_results_csv
        return render_results_csv

    def test_csv_has_expected_columns(self) -> None:
        render_csv = self._load()
        from auto_grader.mc_results_demo_export import build_mc_results_demo_export

        # Build a synthetic export dict directly instead of going through the DB.
        # The render_results_csv function works on the export dict, not the DB.
        export = {
            "exam_instance_id": 1,
            "mc_scan_session_id": 10,
            "session_ordinal": 1,
            "summary": {"correct": 1, "incorrect": 0, "blank": 0, "question_count": 1},
            "questions": [
                {
                    "question_id": "mc-1",
                    "page_number": 1,
                    "scan_id": "page-1.png",
                    "status": "correct",
                    "is_correct": True,
                    "review_required": False,
                    "source": "machine",
                    "machine_status": "correct",
                    "resolved_bubble_label": "B",
                    "final_resolved_bubble_labels": ["B"],
                },
            ],
            "review_queue": [],
        }

        csv_text = render_csv(export)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertIn("question_id", reader.fieldnames)
        self.assertIn("status", reader.fieldnames)
        self.assertIn("is_correct", reader.fieldnames)
        self.assertIn("resolved_bubble_label", reader.fieldnames)
        self.assertIn("source", reader.fieldnames)
        self.assertEqual(rows[0]["question_id"], "mc-1")
        self.assertEqual(rows[0]["is_correct"], "True")


# ---------------------------------------------------------------------------
# DB-backed integration tests
# ---------------------------------------------------------------------------

def _build_manifest() -> dict:
    scan_results = [
        {
            "scan_id": "page-1.png",
            "checksum": hashlib.sha256(b"page-1.png").hexdigest(),
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
                "mc-2": {
                    "question_id": "mc-2",
                    "status": "correct",
                    "is_correct": True,
                    "review_required": False,
                    "marked_bubble_labels": ["C"],
                    "resolved_bubble_labels": ["C"],
                    "correct_bubble_label": "C",
                    "correct_choice_key": "choice-c",
                    "marked_choice_keys": ["choice-c"],
                },
            },
        },
        {
            "scan_id": "page-2.png",
            "checksum": hashlib.sha256(b"page-2.png").hexdigest(),
            "status": "unmatched",
            "failure_reason": "No QR code detected",
        },
    ]
    return {
        "opaque_instance_code": "TEST-INSTANCE-001",
        "expected_page_codes": ["PAGE-1-CODE", "PAGE-2-CODE"],
        "scan_results": scan_results,
        "summary": {
            "total_scans": 2,
            "matched": 1,
            "unmatched": 1,
            "ambiguous": 0,
            "review_required": 1,
        },
    }


class McWorkflowDbTests(unittest.TestCase):
    """DB-backed integration tests for the workflow entrypoint functions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC workflow contract tests."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url, label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        if psycopg is None:
            raise AssertionError("MC workflow contract tests require psycopg.")

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_workflow_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema_name))
                )
            self._schema_created = True
            self.connection = psycopg.connect(
                self.database_url, autocommit=True, row_factory=dict_row,
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

    def test_get_review_queue_after_ingest(self) -> None:
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip
        from auto_grader.mc_workflow import get_review_queue

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        queue = get_review_queue(
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        self.assertIn("review_queue", queue)
        self.assertEqual(len(queue["review_queue"]), 1)
        self.assertEqual(queue["review_queue"][0]["question_id"], "mc-1")
        self.assertEqual(queue["review_queue"][0]["machine_status"], "multiple_marked")

    def test_resolve_and_persist_updates_truth(self) -> None:
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip
        from auto_grader.mc_workflow import resolve_and_persist

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        result = resolve_and_persist(
            exam_instance_id=self.exam_instance_id,
            simple_resolutions={"mc-1": "B"},
            connection=self.connection,
        )

        self.assertEqual(result["review_persist"]["created"], 1)
        current = result["current_results"]
        self.assertEqual(current["question_results"]["mc-1"]["source"], "review_resolution")
        self.assertEqual(current["question_results"]["mc-1"]["status"], "correct")
        self.assertEqual(current["summary"]["unresolved_review_required"], 0)

    def test_resolve_idempotent(self) -> None:
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip
        from auto_grader.mc_workflow import resolve_and_persist

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        resolve_and_persist(
            exam_instance_id=self.exam_instance_id,
            simple_resolutions={"mc-1": "B"},
            connection=self.connection,
        )
        result = resolve_and_persist(
            exam_instance_id=self.exam_instance_id,
            simple_resolutions={"mc-1": "B"},
            connection=self.connection,
        )
        self.assertEqual(result["review_persist"]["unchanged"], 1)
        self.assertEqual(result["review_persist"]["created"], 0)

    def test_export_results_json_has_expected_keys(self) -> None:
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip
        from auto_grader.mc_workflow import export_results

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        result = export_results(
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        self.assertIn("questions", result)
        self.assertIn("summary", result)
        self.assertIn("review_queue", result)
        self.assertEqual(len(result["questions"]), 2)

    def test_workflow_script_review_subcommand(self) -> None:
        """The review subcommand should show the review queue after ingest."""
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        script_path = Path("scripts/mc_workflow.py")
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/mc_workflow.py` with subcommands for the professor-facing MC workflow.",
        )

        with tempfile.TemporaryDirectory(prefix="mc-workflow-") as tempdir:
            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "review",
                    "--exam-instance-id",
                    str(self.exam_instance_id),
                    "--database-url",
                    self.database_url,
                    "--schema-name",
                    self.schema_name,
                    "--output-dir",
                    tempdir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"review subcommand failed. stderr: {result.stderr}",
            )

            review_json = json.loads(
                (Path(tempdir) / "review-queue.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(review_json), 1)
            self.assertEqual(review_json[0]["question_id"], "mc-1")

    def test_workflow_script_resolve_subcommand(self) -> None:
        """The resolve subcommand should persist resolutions and update truth."""
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        script_path = Path("scripts/mc_workflow.py")

        with tempfile.TemporaryDirectory(prefix="mc-workflow-") as tempdir:
            resolutions_path = Path(tempdir) / "resolutions.json"
            resolutions_path.write_text(
                json.dumps({"mc-1": "B"}) + "\n", encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "resolve",
                    "--exam-instance-id",
                    str(self.exam_instance_id),
                    "--resolutions-json",
                    str(resolutions_path),
                    "--database-url",
                    self.database_url,
                    "--schema-name",
                    self.schema_name,
                    "--output-dir",
                    tempdir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"resolve subcommand failed. stderr: {result.stderr}",
            )

            bundle = json.loads(
                (Path(tempdir) / "resolve-result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(bundle["review_persist"]["created"], 1)
            self.assertEqual(
                bundle["current_results"]["question_results"]["mc-1"]["source"],
                "review_resolution",
            )

    def test_workflow_script_export_subcommand_csv(self) -> None:
        """The export subcommand with --format csv should produce a valid CSV."""
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip

        run_mc_db_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        script_path = Path("scripts/mc_workflow.py")

        with tempfile.TemporaryDirectory(prefix="mc-workflow-") as tempdir:
            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "export",
                    "--exam-instance-id",
                    str(self.exam_instance_id),
                    "--format",
                    "csv",
                    "--database-url",
                    self.database_url,
                    "--schema-name",
                    self.schema_name,
                    "--output-dir",
                    tempdir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"export csv subcommand failed. stderr: {result.stderr}",
            )

            csv_path = Path(tempdir) / "mc-results.csv"
            self.assertTrue(csv_path.exists())
            csv_text = csv_path.read_text(encoding="utf-8")
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            question_ids = {row["question_id"] for row in rows}
            self.assertEqual(question_ids, {"mc-1", "mc-2"})


if __name__ == "__main__":
    unittest.main()
