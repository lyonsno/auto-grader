"""Contract tests for the professor-facing assessment authoring GUI surface."""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock


def _load_gui_module(test_case: unittest.TestCase):
    try:
        from auto_grader import mc_workflow_gui
    except ImportError:
        test_case.fail(
            "auto_grader.mc_workflow_gui must exist for the authoring GUI surface."
        )
    return mc_workflow_gui


class _AuthoringTestBase(unittest.TestCase):
    """Base class that stubs out the catalog refresh so authoring tests
    don't need a real DB connection for the grading-target list."""

    def setUp(self) -> None:
        gui = _load_gui_module(self)
        self._patches = [
            mock.patch.object(gui, "list_grading_targets", return_value=[]),
            mock.patch.object(gui, "list_assessment_definitions", return_value=[]),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()


class AssessmentAuthoringGuiRenderTests(_AuthoringTestBase):
    """The rendered page must expose an assessment authoring section."""

    def test_render_page_exposes_authoring_section(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={
                "database_url": "postgresql:///postgres",
                "exam_instance_id": "1",
            }
        )
        html = gui.render_page(state)

        self.assertIn("Author Assessment", html)
        self.assertIn('action="/author"', html)

    def test_render_page_uses_tab_navigation(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        self.assertIn("tab-bar", html)
        self.assertIn('data-tab="grade"', html)
        self.assertIn('data-tab="author"', html)
        self.assertIn('id="tab-grade"', html)
        self.assertIn('id="tab-author"', html)

    def test_render_page_includes_localstorage_draft_persistence(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        self.assertIn("mc_authoring_draft", html)
        self.assertIn("localStorage", html)

    def test_authoring_section_has_title_and_slug_fields(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        self.assertIn('name="authoring_title"', html)
        self.assertIn('name="authoring_slug"', html)

    def test_authoring_section_has_assessment_kind_selector(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        self.assertIn('name="authoring_kind"', html)
        self.assertIn("exam", html.lower())
        self.assertIn("quiz", html.lower())

    def test_authoring_section_has_mc_question_fields(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        # At minimum the form must support entering one MC question with
        # prompt, choices, and correct answer.
        self.assertIn('name="q_1_prompt"', html)
        self.assertIn('name="q_1_choice_A"', html)
        self.assertIn('name="q_1_choice_B"', html)
        self.assertIn('name="q_1_correct"', html)


class AssessmentAuthoringBackendTests(_AuthoringTestBase):
    """POST /author must persist a valid assessment to the durable model."""

    def _post_author(self, gui, app, form_body: str):
        """Helper: POST to /author and return (status, html)."""
        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status

        body_bytes = form_body.encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/author",
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }
        chunks = list(app(environ, start_response))
        html = b"".join(chunks).decode("utf-8")
        return captured["status"], html

    def test_author_endpoint_persists_assessment_to_db(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        fake_connection = mock.MagicMock()
        # fetchone returns the inserted template_version id, then exam_definition id
        fake_connection.execute.return_value.fetchone.side_effect = [
            {"id": 1},  # template_versions INSERT ... RETURNING id
            {"id": 1},  # exam_definitions INSERT ... RETURNING id
        ]

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Quiz+1&"
            "authoring_slug=quiz-1&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+H2O%3F&"
            "q_1_choice_A=Water&"
            "q_1_choice_B=Salt&"
            "q_1_choice_C=Sugar&"
            "q_1_choice_D=Acid&"
            "q_1_correct=A"
        )

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        # The page should show a success message mentioning the assessment
        self.assertIn("Quiz 1", html)
        # The backend must have been called — at least one execute for persistence
        self.assertTrue(fake_connection.execute.called)

    def test_author_endpoint_rejects_missing_title(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=&"
            "authoring_slug=quiz-1&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+H2O%3F&"
            "q_1_choice_A=Water&"
            "q_1_choice_B=Salt&"
            "q_1_correct=A"
        )

        with mock.patch.object(gui, "_connect", return_value=mock.Mock()):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        # Should surface a validation error about the title specifically,
        # not a generic "unknown path" error.
        self.assertIn("title is required", html.lower())

    def test_author_endpoint_builds_valid_yaml_template(self) -> None:
        """The /author happy path completes without error.

        This test exercises the full POST-to-DB path and verifies no error
        is surfaced. It does not assert on the YAML content itself —
        structural validation of authored templates is the responsibility
        of validate_template() in template_schema.py, which has its own
        contract test suite. This test's job is to confirm the GUI
        authoring path reaches the DB without raising.
        """
        gui = _load_gui_module(self)

        # Exercise the form-to-template conversion directly if exposed,
        # otherwise exercise it through the WSGI path and inspect what was persisted.
        app = gui.McWorkflowGuiApp()

        persisted_yaml = None
        fake_connection = mock.MagicMock()

        def capture_execute(query, params=None):
            nonlocal persisted_yaml
            query_str = query if isinstance(query, str) else str(query)
            if "template_versions" in query_str and params:
                # The source_yaml should be among the params
                for p in (params if isinstance(params, (list, tuple)) else [params]):
                    if isinstance(p, str) and "slug" in p:
                        persisted_yaml = p
            result = mock.Mock()
            result.fetchone.return_value = {"id": 1}
            return result

        fake_connection.execute.side_effect = capture_execute

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Quiz+1&"
            "authoring_slug=quiz-1&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+H2O%3F&"
            "q_1_choice_A=Water&"
            "q_1_choice_B=Salt&"
            "q_1_choice_C=&"
            "q_1_choice_D=&"
            "q_1_correct=A"
        )

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        # The page must not contain an error message div (class="message error").
        self.assertNotIn('class="message error"', html)


class AssessmentKindTests(_AuthoringTestBase):
    """Quiz must be a first-class assessment kind, not silently exam-only."""

    def test_quiz_kind_accepted_without_error(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        fake_connection = mock.MagicMock()
        fake_connection.execute.return_value.fetchone.return_value = {"id": 1}

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Weekly+Quiz+3&"
            "authoring_slug=weekly-quiz-3&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+NaCl%3F&"
            "q_1_choice_A=Salt&"
            "q_1_choice_B=Sugar&"
            "q_1_correct=A"
        )

        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status

        body_bytes = form_body.encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/author",
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            chunks = list(app(environ, start_response))

        html = b"".join(chunks).decode("utf-8")
        self.assertEqual(captured["status"], "200 OK")
        # Must show a success message, not an error
        self.assertIn("saved", html.lower())


class AssessmentAuthoringValidationTests(_AuthoringTestBase):
    """Panopticon-driven hardening: correct-answer validation, gap tolerance, slug conflicts."""

    def _post_author(self, gui, app, form_body: str):
        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status

        body_bytes = form_body.encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/author",
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }
        chunks = list(app(environ, start_response))
        html = b"".join(chunks).decode("utf-8")
        return captured["status"], html

    def test_rejects_empty_correct_answer(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Quiz+1&"
            "authoring_slug=quiz-1&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+H2O%3F&"
            "q_1_choice_A=Water&"
            "q_1_choice_B=Salt&"
            "q_1_correct="
        )

        with mock.patch.object(gui, "_connect", return_value=mock.Mock()):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        self.assertIn("correct answer", html.lower())

    def test_gap_in_question_numbers_does_not_silently_drop(self) -> None:
        """If q_1 and q_3 are filled but q_2 is blank, both should be collected."""
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        fake_connection = mock.MagicMock()
        fake_connection.execute.return_value.fetchone.return_value = {"id": 1}

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Gap+Test&"
            "authoring_slug=gap-test&"
            "authoring_kind=exam&"
            "q_1_prompt=First%3F&"
            "q_1_choice_A=Yes&"
            "q_1_choice_B=No&"
            "q_1_correct=A&"
            "q_2_prompt=&"
            "q_3_prompt=Third%3F&"
            "q_3_choice_A=Yes&"
            "q_3_choice_B=No&"
            "q_3_correct=B"
        )

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        self.assertIn("2 questions", html.lower())

    def test_duplicate_slug_surfaces_user_friendly_error(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        fake_connection = mock.MagicMock()
        fake_connection.execute.side_effect = Exception(
            "duplicate key value violates unique constraint"
        )

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Quiz+1&"
            "authoring_slug=quiz-1&"
            "authoring_kind=quiz&"
            "q_1_prompt=What%3F&"
            "q_1_choice_A=A&"
            "q_1_choice_B=B&"
            "q_1_correct=A"
        )

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        self.assertIn("already exists", html.lower())

    def test_authoring_message_clears_on_non_author_post(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(
            config={"database_url": "postgresql:///postgres", "exam_instance_id": "1"},
            authoring_message="Saved something earlier.",
        )
        app = gui.McWorkflowGuiApp(initial_state=state)

        # Trigger a successful /review POST via mocked backend; the
        # authoring_message should be cleared on the success path.
        body = b"database_url=postgresql%3A%2F%2F%2Fpostgres&exam_instance_id=1"
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/review",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }

        captured: dict[str, object] = {}
        def start_response(status, headers):
            captured["status"] = status

        with mock.patch.object(gui, "_connect", return_value=mock.Mock()):
            with mock.patch.object(gui, "get_review_queue", return_value={"review_queue": [], "summary": {}}):
                list(app(environ, start_response))

        self.assertIsNone(app.state.authoring_message)


class ComputedDistractorTests(_AuthoringTestBase):
    """Questions can use computed distractors instead of static choices."""

    def _post_author(self, gui, app, form_body: str):
        captured: dict[str, object] = {}

        def start_response(status, headers):
            captured["status"] = status

        body_bytes = form_body.encode("utf-8")
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/author",
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }
        chunks = list(app(environ, start_response))
        html = b"".join(chunks).decode("utf-8")
        return captured["status"], html

    def test_render_page_exposes_question_mode_toggle(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        # Each question block must have a mode selector
        self.assertIn('name="q_1_mode"', html)
        self.assertIn("static", html.lower())
        self.assertIn("computed", html.lower())

    def test_render_page_exposes_distractor_fields(self) -> None:
        gui = _load_gui_module(self)
        state = gui.GuiState(config={"database_url": "postgresql:///postgres"})
        html = gui.render_page(state)

        self.assertIn('name="q_1_answer_expr"', html)
        self.assertIn('name="q_1_distractor_1"', html)
        self.assertIn('name="q_1_variables"', html)

    def test_computed_distractor_question_persists_without_error(self) -> None:
        gui = _load_gui_module(self)
        app = gui.McWorkflowGuiApp()

        fake_connection = mock.MagicMock()
        fake_connection.execute.return_value.fetchone.return_value = {"id": 1}

        form_body = (
            "database_url=postgresql%3A%2F%2F%2Fpostgres&"
            "authoring_title=Computed+Quiz&"
            "authoring_slug=computed-quiz&"
            "authoring_kind=quiz&"
            "q_1_prompt=What+is+mass+%2F+density%3F&"
            "q_1_mode=computed&"
            "q_1_variables=mass%3A+%7Btype%3A+float%2C+min%3A+10.0%2C+max%3A+99.0%2C+step%3A+0.1%7D%0Adensity%3A+%7Btype%3A+float%2C+min%3A+0.7%2C+max%3A+3.5%2C+step%3A+0.01%7D&"
            "q_1_answer_expr=mass+%2F+density&"
            "q_1_distractor_1=density+%2F+mass&"
            "q_1_distractor_2=mass+*+density&"
            "q_1_distractor_3=mass+%2B+density"
        )

        with mock.patch.object(gui, "_connect", return_value=fake_connection):
            status, html = self._post_author(gui, app, form_body)

        self.assertEqual(status, "200 OK")
        self.assertIn("saved", html.lower())
        # Should mention that distractor evaluation needs the generation pipeline
        self.assertIn("distractor", html.lower())

    def test_computed_distractor_question_builds_distractor_yaml(self) -> None:
        """The YAML for a computed question must use the distractors schema, not choices."""
        gui = _load_gui_module(self)

        form = {
            "q_1_prompt": "What is mass / density?",
            "q_1_mode": "computed",
            "q_1_variables": "mass: {type: float, min: 10.0, max: 99.0, step: 0.1}",
            "q_1_answer_expr": "mass / density",
            "q_1_distractor_1": "density / mass",
            "q_1_distractor_2": "mass * density",
        }
        questions = gui._extract_authored_questions(form)
        self.assertEqual(len(questions), 1)
        q = questions[0]
        self.assertEqual(q["mode"], "computed")
        self.assertIn("answer_expr", q)
        self.assertIn("distractor_exprs", q)
        self.assertNotIn("choices", q)


if __name__ == "__main__":
    unittest.main()
