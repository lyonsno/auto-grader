"""Postgres-backed contract tests for the DB-backed MC/OpenCV round trip.

This suite pins the missing integration seam above the landed DB primitives:
persist machine results, optionally persist human review resolutions, and then
return the authoritative current-final MC truth without forcing callers to
stitch those steps together by hand.
"""

from __future__ import annotations

import hashlib
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
    scan_results = [
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
                    correct_bubble_label="B",
                    correct_choice_key="choice-b",
                    marked_choice_keys=["choice-a", "choice-b"],
                ),
                "mc-2": _build_question(
                    question_id="mc-2",
                    status="correct",
                    is_correct=True,
                    review_required=False,
                    marked_bubble_labels=["C"],
                    resolved_bubble_labels=["C"],
                    correct_bubble_label="C",
                    correct_choice_key="choice-c",
                    marked_choice_keys=["choice-c"],
                ),
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


def _review_resolutions() -> dict[str, dict[str, dict]]:
    return {
        "page-1.png": {
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
        },
    }


def _load_round_trip_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_db_round_trip import run_mc_db_round_trip
    except (ModuleNotFoundError, ImportError):
        test_case.fail(
            "Add `auto_grader.mc_db_round_trip.run_mc_db_round_trip(...)` so callers "
            "can drive the DB-backed MC/OpenCV round trip without hand-stitching "
            "machine persistence, review persistence, and final-result reads."
        )
    return run_mc_db_round_trip


class McDbRoundTripContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest(
                "Set TEST_DATABASE_URL to run MC DB round-trip contract tests "
                "against an explicit disposable Postgres instance."
            )
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url,
                label="TEST_DATABASE_URL",
            )
        except ValueError as exc:
            raise AssertionError(
                f"{exc} for the MC DB round-trip contract suite."
            ) from exc
        if psycopg is None:
            raise AssertionError(
                "MC DB round-trip contract tests require psycopg in the active environment."
            )

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_mc_db_round_trip_{uuid.uuid4().hex}"
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
                f"MC DB round-trip contract test setup failed: {exc}"
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

    def test_run_mc_db_round_trip_persists_machine_and_review_then_reads_final_truth(self) -> None:
        run_round_trip = _load_round_trip_module(self)

        result = run_round_trip(
            manifest=_build_manifest(),
            exam_instance_id=self.exam_instance_id,
            review_resolutions_by_scan_id=_review_resolutions(),
            connection=self.connection,
        )

        self.assertTrue(result["machine_persist"]["created"])
        self.assertEqual(result["review_resolution_persist"]["created"], 1)
        self.assertEqual(result["review_resolution_persist"]["updated"], 0)
        self.assertEqual(result["review_resolution_persist"]["unchanged"], 0)
        self.assertEqual(result["review_resolution_persist"]["scan_ids"], ["page-1.png"])

        current = result["current_results"]
        self.assertEqual(current["exam_instance_id"], self.exam_instance_id)
        self.assertEqual(current["summary"]["matched"], 1)
        self.assertEqual(current["summary"]["unmatched"], 1)
        self.assertEqual(current["summary"]["unresolved_review_required"], 0)
        self.assertEqual(current["question_results"]["mc-1"]["source"], "review_resolution")
        self.assertEqual(current["question_results"]["mc-1"]["status"], "correct")
        self.assertEqual(current["question_results"]["mc-2"]["source"], "machine")
        self.assertEqual(current["question_results"]["mc-2"]["status"], "correct")

    def test_run_mc_db_round_trip_script_writes_round_trip_bundle(self) -> None:
        script_path = Path("scripts/run_mc_db_round_trip.py")
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/run_mc_db_round_trip.py` so the DB-backed MC/OpenCV "
            "round trip can run through one small helper surface.",
        )

        with tempfile.TemporaryDirectory(prefix="mc-db-round-trip-") as tempdir:
            tempdir_path = Path(tempdir)
            manifest_path = tempdir_path / "manifest.json"
            reviews_path = tempdir_path / "review-resolutions.json"
            output_dir = tempdir_path / "out"

            manifest_path.write_text(
                json.dumps(_build_manifest(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            reviews_path.write_text(
                json.dumps(_review_resolutions(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "--manifest-json",
                    str(manifest_path),
                    "--exam-instance-id",
                    str(self.exam_instance_id),
                    "--review-resolutions-json",
                    str(reviews_path),
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
                "The round-trip helper script should succeed on a valid manifest "
                f"and review-resolution bundle. stderr was: {result.stderr}",
            )
            self.assertIn("mc-db-round-trip.json", result.stdout)

            bundle = json.loads(
                (output_dir / "mc-db-round-trip.json").read_text(encoding="utf-8")
            )
            self.assertEqual(bundle["review_resolution_persist"]["created"], 1)
            self.assertEqual(
                bundle["current_results"]["question_results"]["mc-1"]["source"],
                "review_resolution",
            )


if __name__ == "__main__":
    unittest.main()
