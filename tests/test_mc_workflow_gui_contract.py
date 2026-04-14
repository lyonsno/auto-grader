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
            }
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
        self.assertIn("Workflow Settings", html)
        self.assertIn("Choose the exam record and scanned pages you want to grade.", html)
        self.assertIn("When you're ready, ingest the scans to load results and any questions that need review.", html)

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
        self.assertIn("Workflow Artifacts", html)
        self.assertIn("Export Files", html)
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

        self.assertIn("Exam Instance ID", html)
        self.assertIn("Scan Directory", html)
        self.assertIn("Workflow Settings", html)
        self.assertIn("Database URL", html)
        self.assertIn("Schema Name", html)
        self.assertIn("Artifact JSON", html)
        self.assertIn("Output Directory", html)

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
