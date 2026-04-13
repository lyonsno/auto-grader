"""Contracts for the DB-backed MC demo export surface."""

from __future__ import annotations

import hashlib
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


def _build_manifest() -> dict:
    return {
        "opaque_instance_code": "TEST-INSTANCE-001",
        "expected_page_codes": ["PAGE-1-CODE", "PAGE-2-CODE"],
        "scan_results": [
            {
                "scan_id": "page-1.png",
                "checksum": hashlib.sha256(b"page-1.png").hexdigest(),
                "status": "matched",
                "failure_reason": None,
                "page_number": 1,
                "fallback_page_code": "PAGE-1-CODE",
                "scored_questions": {
                    "mc-1": _build_question(
                        question_id="mc-1",
                        status="multiple_marked",
                        is_correct=False,
                        review_required=True,
                        marked_bubble_labels=["A", "B"],
                        resolved_bubble_labels=["A", "B"],
                        marked_choice_keys=["choice-a", "choice-b"],
                    ),
                    "mc-2": _build_question(
                        question_id="mc-2",
                        status="incorrect",
                        is_correct=False,
                        review_required=False,
                        marked_bubble_labels=["A"],
                        resolved_bubble_labels=["A"],
                        correct_bubble_label="B",
                        correct_choice_key="choice-b",
                        marked_choice_keys=["choice-a"],
                    ),
                },
            },
            {
                "scan_id": "page-2.png",
                "checksum": hashlib.sha256(b"page-2.png").hexdigest(),
                "status": "unmatched",
                "failure_reason": "No QR code detected",
            },
        ],
        "summary": {
            "total_scans": 2,
            "matched": 1,
            "unmatched": 1,
            "ambiguous": 0,
            "review_required": 1,
        },
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


def _load_export_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_results_demo_export import build_mc_results_demo_export
    except (ModuleNotFoundError, ImportError):
        test_case.fail(
            "Add `auto_grader.mc_results_demo_export.build_mc_results_demo_export(...)` "
            "so the landed DB truth can be exported as one compact demo surface."
        )
    return build_mc_results_demo_export


class McResultsDemoExportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC results demo export contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the MC results demo export contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "MC results demo export contract tests require psycopg in the active environment."
            )

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_results_demo_export_{uuid.uuid4().hex}"
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
                f"MC results demo export contract test setup failed: {exc}"
            ) from exc

        initialize_schema(self.connection)
        self._seed_exam_instance()
        self._seed_current_state()

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

    def _seed_current_state(self) -> None:
        from auto_grader.mc_review_db import persist_mc_review_resolutions_to_db
        from auto_grader.mc_scan_db import persist_scan_session_to_db

        result = persist_scan_session_to_db(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        self.mc_scan_session_id = result["mc_scan_session_id"]
        persist_mc_review_resolutions_to_db(
            mc_scan_session_id=self.mc_scan_session_id,
            scan_id="page-1.png",
            resolved_questions=_resolved_correct_for("mc-1"),
            connection=self.connection,
        )

    def test_build_mc_results_demo_export_compacts_current_db_truth_for_demo(self) -> None:
        build_export = _load_export_module(self)

        exported = build_export(
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertEqual(exported["exam_instance_id"], self.exam_instance_id)
        self.assertEqual(exported["mc_scan_session_id"], self.mc_scan_session_id)
        self.assertEqual(exported["summary"]["matched"], 1)
        self.assertEqual(exported["summary"]["unmatched"], 1)
        self.assertEqual(exported["summary"]["unresolved_review_required"], 0)
        self.assertEqual(exported["summary"]["resolved_by_review"], 1)
        self.assertEqual(exported["summary"]["question_count"], 2)

        self.assertEqual(
            [item["question_id"] for item in exported["questions"]],
            ["mc-1", "mc-2"],
            "The demo export should provide one stable, question-ordered surface "
            "instead of forcing callers to walk a question-results mapping by hand.",
        )
        self.assertEqual(exported["questions"][0]["source"], "review_resolution")
        self.assertEqual(exported["questions"][0]["status"], "correct")
        self.assertEqual(exported["questions"][0]["resolved_bubble_label"], "B")
        self.assertEqual(exported["questions"][0]["machine_status"], "multiple_marked")
        self.assertEqual(exported["questions"][1]["source"], "machine")
        self.assertEqual(exported["review_queue"], [])

    def test_export_mc_results_demo_script_writes_json_and_human_summary(self) -> None:
        script_path = Path("scripts/export_mc_results_demo.py")
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/export_mc_results_demo.py` so the DB-backed MC truth can "
            "be exported through one small runnable demo surface.",
        )

        with tempfile.TemporaryDirectory(prefix="mc-results-demo-export-") as tempdir:
            output_dir = Path(tempdir) / "out"

            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "--exam-instance-id",
                    str(self.exam_instance_id),
                    "--database-url",
                    self.database_url,
                    "--schema-name",
                    self.schema_name,
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"The demo export script should succeed on a valid DB-backed exam instance. "
                f"stderr was: {result.stderr}",
            )
            self.assertIn("mc-results-summary.txt", result.stdout)
            self.assertTrue((output_dir / "mc-results-export.json").exists())
            self.assertTrue((output_dir / "mc-results-summary.txt").exists())

            summary_text = (output_dir / "mc-results-summary.txt").read_text(encoding="utf-8")
            self.assertIn("exam_instance_id:", summary_text)
            self.assertIn("resolved_by_review: 1", summary_text)
            self.assertIn("mc-1", summary_text)
            self.assertIn("multiple_marked -> correct", summary_text)


if __name__ == "__main__":
    unittest.main()
