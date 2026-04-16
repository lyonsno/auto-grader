"""Contract tests for the professor-facing MC workflow GUI."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


def _load_gui_module(test_case: unittest.TestCase):
    try:
        from auto_grader import mc_workflow_gui
    except ImportError:
        test_case.fail(
            "Add `auto_grader.mc_workflow_gui` so the landed professor workflow "
            "can be exercised through a thin local GUI instead of only through terminal commands."
        )
    return mc_workflow_gui


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class McWorkflowGuiContractTests(unittest.TestCase):
    def test_render_page_exposes_ingest_review_resolve_export_surface(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "schema_name": "demo_schema",
                "exam_instance_id": "123",
                "artifact_json": "/tmp/artifact.json",
                "scan_dir": "/tmp/scans",
                "output_dir": "/tmp/out",
            },
            grading_targets=[
                {
                    "exam_instance_id": 123,
                    "label": "Fall Quiz 2 - Noah",
                }
            ],
            assessment_definitions=[
                {
                    "exam_definition_id": 8,
                    "label": "Chemistry Quiz 2",
                }
            ],
        )

        html = gui.render_page(state)

        self.assertIn("Professor MC Workflow", html)
        self.assertIn('action="/ingest"', html)
        self.assertIn('action="/review"', html)
        self.assertIn('action="/resolve"', html)
        self.assertIn('action="/export"', html)
        self.assertIn("exam_instance_id", html)
        self.assertIn("artifact_json", html)
        self.assertIn("scan_dir", html)
        self.assertIn("data-busy-label", html)
        self.assertIn("workflow-spinner", html)
        self.assertIn("<details", html)
        self.assertIn("Advanced Settings", html)
        self.assertIn("Selected Exam", html)
        self.assertNotIn("Exam Instance ID", html)
        self.assertIn('<select name="exam_instance_id"', html)
        self.assertIn("Fall Quiz 2 - Noah", html)
        self.assertIn('action="/create-target"', html)
        self.assertIn("Grade Scans", html)
        self.assertIn("Need a Different Exam?", html)
        self.assertIn("Create new exam", html)
        self.assertIn("Create from assessment", html)
        self.assertIn("Assessment Template", html)
        self.assertIn("Chemistry Quiz 2", html)
        self.assertIn("Choose the exam and scanned pages you want to grade.", html)
        self.assertIn("If the exam you want is not listed, create a new one here.", html)
        self.assertIn("When you're ready, ingest the scans to load results and any questions that need review.", html)
        self.assertNotIn("exam record", html.lower())
        ingest_start = html.index('<form method="post" action="/ingest"')
        ingest_end = html.index("</form>", ingest_start)
        create_start = html.index('<form method="post" action="/create-target"')
        self.assertGreater(create_start, ingest_end)

    def test_render_page_promotes_summary_into_stat_cards_and_result_sections(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "exam_instance_id": "123",
                "artifact_json": "/tmp/artifact.json",
                "scan_dir": "/tmp/scans",
                "output_dir": "/tmp/out",
            },
            ingest_result={
                "mc_scan_session_id": 44,
                "manifest_path": "/tmp/out/session_manifest.json",
                "output_dir": "/tmp/out",
            },
            summary={
                "matched": 5,
                "incorrect": 15,
                "correct": 4,
                "unresolved_review_required": 0,
            },
            export_paths={
                "json": "/tmp/out/mc-results.json",
                "csv": "/tmp/out/mc-results.csv",
            },
        )

        html = gui.render_page(state)

        self.assertIn("stat-card", html)
        self.assertIn("Matched Pages", html)
        self.assertIn("Questions Needing Review", html)
        self.assertIn("Questions Correct", html)
        self.assertIn("Questions Incorrect", html)
        self.assertIn("Showing results for scan set: scans.", html)
        self.assertIn("Result Details", html)
        self.assertIn("Saved Files", html)
        self.assertIn("Detailed Grading Record", html)
        self.assertIn("Results Folder", html)
        self.assertIn("Saved Exports", html)
        self.assertIn("Spreadsheet", html)
        self.assertIn("Detailed Results", html)
        self.assertIn("path-chip", html)
        self.assertIn('action="/open-path"', html)
        self.assertIn('action="/reveal-path"', html)
        self.assertIn("Open", html)
        self.assertIn("Show in Finder", html)
        self.assertNotIn("Workflow Artifacts", html)
        self.assertNotIn(">Manifest<", html)
        self.assertNotIn(">Export Files<", html)
        self.assertNotIn(">Matched<", html)
        self.assertNotIn(">Needs Review<", html)

    def test_render_page_hides_operator_fields_inside_settings_disclosure(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "schema_name": "demo_schema",
                "exam_instance_id": "123",
                "artifact_json": "/tmp/artifact.json",
                "scan_dir": "/tmp/scans",
                "output_dir": "/tmp/out",
            }
        )

        html = gui.render_page(state)

        self.assertIn("Selected Exam", html)
        self.assertNotIn("Exam Instance ID", html)
        self.assertIn("Scanned Pages Folder", html)
        self.assertIn("Advanced Settings", html)
        self.assertIn("Database URL", html)
        self.assertIn("Schema Name", html)
        self.assertIn("Answer Key File", html)
        self.assertIn("Save Results To", html)
        self.assertNotIn("Different exam? Create a separate exam record.", html)

    def test_render_page_promotes_target_creation_when_no_exam_records_exist(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "schema_name": "demo_schema",
                "exam_instance_id": "",
                "artifact_json": "/tmp/artifact.json",
                "scan_dir": "/tmp/scans",
                "output_dir": "/tmp/out",
            },
            assessment_definitions=[
                {
                    "exam_definition_id": 8,
                    "label": "Chemistry Quiz 2",
                }
            ],
        )

        html = gui.render_page(state)

        self.assertIn("Create New Exam", html)
        self.assertIn("No exams are available yet.", html)
        self.assertIn("Create one so new scans have somewhere to land.", html)
        self.assertIn("Create from assessment", html)
        self.assertIn("Assessment Template", html)
        self.assertNotIn("Need a Different Exam?", html)

    def test_render_page_explains_empty_summary_and_review_queue(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "schema_name": "demo_schema",
                "exam_instance_id": "123",
                "artifact_json": "/tmp/artifact.json",
                "scan_dir": "/tmp/scans",
                "output_dir": "/tmp/out",
            }
        )

        html = gui.render_page(state)

        self.assertIn("Ingest a set of scans to see the current grading summary.", html)
        self.assertIn("Only questions that need a human decision appear here.", html)
        self.assertIn("Choose a resolution for each flagged question, then persist it to finalize the result.", html)

    def test_wsgi_app_ingest_action_surfaces_review_queue_without_redefining_backend(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        ingest_result = {
            "exam_instance_id": 123,
            "mc_scan_session_id": 44,
            "manifest_path": "/tmp/out/session_manifest.json",
            "summary": {"matched": 1, "unresolved_review_required": 1},
            "review_queue": [
                {
                    "question_id": "mc-1",
                    "machine_status": "multiple_marked",
                    "scan_id": "page-1.png",
                    "page_number": 1,
                    "marked_bubble_labels": ["A", "B"],
                }
            ],
        }

        def start_response(status, headers):
            captured["status"] = status
            captured["headers"] = headers

        captured: dict[str, object] = {}
        body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "schema_name=demo_schema&"
            "exam_instance_id=123&"
            "artifact_json=%2Ftmp%2Fartifact.json&"
            "scan_dir=%2Ftmp%2Fscans&"
            "output_dir=%2Ftmp%2Fout"
        ).encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/ingest",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }

        fake_connection = mock.Mock()

        with mock.patch.object(gui, "_connect", return_value=fake_connection), mock.patch.object(
            gui,
            "ingest_and_persist_from_scan_dir",
            return_value=ingest_result,
        ) as mock_ingest:
            chunks = list(app(environ, start_response))

        html = b"".join(chunks).decode("utf-8")
        self.assertEqual(captured["status"], "200 OK")
        self.assertIn("mc-1", html)
        self.assertIn("multiple_marked", html)
        self.assertIn("/tmp/out/session_manifest.json", html)
        mock_ingest.assert_called_once_with(
            artifact_json_path="/tmp/artifact.json",
            scan_dir="/tmp/scans",
            exam_instance_id=123,
            output_dir="/tmp/out",
            connection=fake_connection,
        )

    def test_wsgi_app_can_open_known_artifact_paths(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp(
            initial_state=gui.GuiState(
                config={"output_dir": "/tmp/out"},
                ingest_result={
                    "mc_scan_session_id": 44,
                    "manifest_path": "/tmp/out/session_manifest.json",
                    "output_dir": "/tmp/out",
                },
                export_paths={
                    "json": "/tmp/out/mc-results.json",
                },
            )
        )

        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status
            captured["headers"] = headers

        body = "path=%2Ftmp%2Fout%2Fmc-results.json".encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/reveal-path",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }

        with mock.patch.object(gui.subprocess, "run") as mock_run:
            chunks = list(app(environ, start_response))

        html = b"".join(chunks).decode("utf-8")
        self.assertEqual(captured["status"], "200 OK")
        self.assertIn("Opened in Finder.", html)
        mock_run.assert_called_once_with(["open", "-R", "/tmp/out/mc-results.json"], check=True)

    def test_wsgi_app_can_create_grading_target_and_select_it(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp(
            initial_state=gui.GuiState(
                config={
                    "database_url": "postgresql:///postgres",
                    "schema_name": "demo_schema",
                }
            )
        )

        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status
            captured["headers"] = headers

        body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "schema_name=demo_schema&"
            "new_exam_definition_id=8&"
            "new_target_name=Fall+Quiz+2+-+Noah"
        ).encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/create-target",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }

        fake_connection = mock.Mock()
        created_target = {
            "exam_instance_id": 321,
            "label": "Chemistry Quiz 2 - Fall Quiz 2 - Noah",
            "target_name": "Fall Quiz 2 - Noah",
            "exam_title": "Chemistry Quiz 2",
        }
        refreshed_targets = [
            created_target,
        ]
        assessment_definitions = [
            {"exam_definition_id": 8, "label": "Chemistry Quiz 2"},
        ]

        with mock.patch.object(gui, "_connect", return_value=fake_connection), mock.patch.object(
            gui,
            "create_grading_target",
            return_value=created_target,
        ) as mock_create, mock.patch.object(
            gui,
            "list_grading_targets",
            return_value=refreshed_targets,
        ), mock.patch.object(
            gui,
            "list_assessment_definitions",
            return_value=assessment_definitions,
        ):
            chunks = list(app(environ, start_response))

        html = b"".join(chunks).decode("utf-8")
        self.assertEqual(captured["status"], "200 OK")
        self.assertIn("Created exam target.", html)
        self.assertIn("Chemistry Quiz 2 - Fall Quiz 2 - Noah", html)
        self.assertEqual(app.state.config["exam_instance_id"], "321")
        mock_create.assert_called_once_with(
            exam_definition_id=8,
            target_name="Fall Quiz 2 - Noah",
            connection=fake_connection,
        )

    def test_launch_script_help_exposes_local_gui_entrypoint(self) -> None:
        script_path = _repo_root() / "scripts" / "launch_mc_workflow_gui.py"
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/launch_mc_workflow_gui.py` so the professor GUI can be launched "
            "as a first-class local entrypoint rather than only by importing modules.",
        )
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--host", result.stdout)
        self.assertIn("--port", result.stdout)
        self.assertIn("--open-browser", result.stdout)


if __name__ == "__main__":
    unittest.main()
