"""Thin local web GUI for the professor-facing MC workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import parse_qs
import webbrowser
from wsgiref.simple_server import WSGIServer

from psycopg import sql

from auto_grader.db import create_connection
from auto_grader.mc_workflow import (
    create_grading_target,
    export_results,
    get_review_queue,
    ingest_and_persist_from_scan_dir,
    list_assessment_definitions,
    list_grading_targets,
    render_results_csv,
    resolve_and_persist,
)


_CONFIG_FIELDS = (
    "database_url",
    "schema_name",
    "exam_instance_id",
    "artifact_json",
    "scan_dir",
    "output_dir",
)
_RESOLUTION_CHOICES = ("", "__BLANK__", "A", "B", "C", "D", "E", "F")
_ASSESSMENT_KINDS = ("exam", "quiz")
# A–E only: the professor's exams and quizzes use 5-choice MC.
# _RESOLUTION_CHOICES includes F for grading externally-created assessments;
# the authoring surface intentionally does not offer F.
_CHOICE_LABELS = ("A", "B", "C", "D", "E")
_INITIAL_QUESTION_COUNT = 5


@dataclass
class GuiState:
    config: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    error: str | None = None
    ingest_result: dict[str, Any] | None = None
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    export_paths: dict[str, str] = field(default_factory=dict)
    grading_targets: list[dict[str, Any]] = field(default_factory=list)
    assessment_definitions: list[dict[str, Any]] = field(default_factory=list)
    authoring_message: str | None = None
    active_tab: str = "grade"
    authoring_form: dict[str, str] = field(default_factory=dict)


class McWorkflowGuiApp:
    def __init__(self, *, initial_state: GuiState | None = None) -> None:
        self.state = initial_state or GuiState()

    def __call__(self, environ: dict[str, Any], start_response) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        if method == "POST":
            form = _parse_post_body(environ)
            self._update_config(form)
            if path == "/author":
                self.state.active_tab = "author"
                self.state.authoring_form = {
                    k: v for k, v in form.items()
                    if k.startswith(("authoring_", "q_"))
                }
            else:
                self.state.active_tab = "grade"
            try:
                if path == "/ingest":
                    self._handle_ingest()
                elif path == "/review":
                    self._handle_review()
                elif path == "/resolve":
                    self._handle_resolve(form)
                elif path == "/export":
                    self._handle_export()
                elif path == "/create-target":
                    self._handle_create_target(form)
                elif path == "/open-path":
                    self._handle_open_path(form, reveal=False)
                elif path == "/reveal-path":
                    self._handle_open_path(form, reveal=True)
                elif path == "/author":
                    self._handle_author(form)
                elif path == "/browse-dir":
                    self._handle_browse_dir(form)
                elif path == "/browse-file":
                    self._handle_browse_file(form)
                else:
                    raise ValueError(f"Unknown action path: {path}")
                self._refresh_catalog()
                if path != "/author":
                    self.state.authoring_message = None
                if path == "/author":
                    self.state.authoring_form = {}
                self.state.error = None
            except Exception as exc:
                self.state.error = str(exc)
                self.state.message = None
                self.state.authoring_message = None
        else:
            self._refresh_catalog_if_possible()

        html = render_page(self.state).encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [html]

    def _update_config(self, form: dict[str, str]) -> None:
        for key in _CONFIG_FIELDS:
            if key in form:
                self.state.config[key] = form[key]

    def _handle_ingest(self) -> None:
        config = self._require_config("database_url", "exam_instance_id", "artifact_json", "scan_dir", "output_dir")
        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            result = ingest_and_persist_from_scan_dir(
                artifact_json_path=config["artifact_json"],
                scan_dir=config["scan_dir"],
                exam_instance_id=_require_int(config["exam_instance_id"], "exam_instance_id"),
                output_dir=config["output_dir"],
                connection=connection,
            )
        finally:
            connection.close()

        self.state.ingest_result = result
        self.state.review_queue = list(result["review_queue"])
        self.state.summary = dict(result["summary"])
        self.state.message = "Ingest completed."
        self.state.export_paths = {}

    def _handle_review(self) -> None:
        config = self._require_config("database_url", "exam_instance_id")
        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            queue = get_review_queue(
                exam_instance_id=_require_int(config["exam_instance_id"], "exam_instance_id"),
                connection=connection,
            )
        finally:
            connection.close()

        self.state.review_queue = list(queue["review_queue"])
        self.state.summary = dict(queue["summary"])
        self.state.message = "Review queue refreshed."

    def _handle_resolve(self, form: dict[str, str]) -> None:
        config = self._require_config("database_url", "exam_instance_id")
        simple_resolutions = _extract_simple_resolutions(form)
        if not simple_resolutions:
            raise ValueError("Select at least one resolution before submitting.")

        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            result = resolve_and_persist(
                exam_instance_id=_require_int(config["exam_instance_id"], "exam_instance_id"),
                simple_resolutions=simple_resolutions,
                connection=connection,
            )
            queue = get_review_queue(
                exam_instance_id=_require_int(config["exam_instance_id"], "exam_instance_id"),
                connection=connection,
            )
        finally:
            connection.close()

        self.state.review_queue = list(queue["review_queue"])
        self.state.summary = dict(queue["summary"])
        self.state.message = (
            f"Persisted resolutions. created={result['review_persist']['created']} "
            f"updated={result['review_persist']['updated']} "
            f"unchanged={result['review_persist']['unchanged']}"
        )

    def _handle_export(self) -> None:
        config = self._require_config("database_url", "exam_instance_id", "output_dir")
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            exported = export_results(
                exam_instance_id=_require_int(config["exam_instance_id"], "exam_instance_id"),
                connection=connection,
            )
        finally:
            connection.close()

        json_path = output_dir / "mc-results.json"
        csv_path = output_dir / "mc-results.csv"
        txt_path = output_dir / "mc-results-summary.txt"
        json_path.write_text(json.dumps(exported, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        csv_path.write_text(render_results_csv(exported), encoding="utf-8")
        txt_path.write_text(_render_summary_text(exported), encoding="utf-8")

        self.state.summary = dict(exported["summary"])
        self.state.review_queue = list(exported["review_queue"])
        self.state.export_paths = {
            "json": str(json_path),
            "csv": str(csv_path),
            "txt": str(txt_path),
        }
        self.state.message = "Export completed."

    def _handle_create_target(self, form: dict[str, str]) -> None:
        config = self._require_config("database_url")
        exam_definition_id = _require_int(
            form.get("new_exam_definition_id", ""),
            "new_exam_definition_id",
        )
        target_name = form.get("new_target_name", "").strip()
        if target_name == "":
            raise ValueError("new_target_name must be provided")

        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            created = create_grading_target(
                exam_definition_id=exam_definition_id,
                target_name=target_name,
                connection=connection,
            )
        finally:
            connection.close()

        self.state.config["exam_instance_id"] = str(created["exam_instance_id"])
        self.state.message = "Created exam target."

    def _handle_author(self, form: dict[str, str]) -> None:
        title = form.get("authoring_title", "").strip()
        slug = form.get("authoring_slug", "").strip()
        kind = form.get("authoring_kind", "exam").strip()

        if not title:
            raise ValueError("Title is required.")
        if not slug:
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if kind not in _ASSESSMENT_KINDS:
            raise ValueError(f"Unsupported assessment kind: {kind}")

        questions = _extract_authored_questions(form)
        if not questions:
            raise ValueError("At least one question with a prompt is required.")

        has_computed = False
        for q in questions:
            if q["mode"] == "static":
                if q["correct"] not in _CHOICE_LABELS:
                    raise ValueError(f"Question {q['form_slot']}: correct answer must be one of {', '.join(_CHOICE_LABELS)}.")
            elif q["mode"] == "computed":
                has_computed = True
                if not q.get("answer_expr"):
                    raise ValueError(f"Question {q['form_slot']}: answer expression is required for computed distractors.")
                if not q.get("distractor_exprs"):
                    raise ValueError(f"Question {q['form_slot']}: at least one distractor expression is required.")

        source_yaml = _build_template_yaml(slug=slug, title=title, kind=kind, questions=questions)

        # database_url is optional here — _connect falls through to the
        # DATABASE_URL env var when not provided, which is the common case
        # on the Author tab where the Workflow Settings disclosure may
        # never have been opened.
        config = self.state.config
        connection = _connect(
            database_url=config.get("database_url") or None,
            schema_name=config.get("schema_name") or None,
        )
        # Intentionally write-once: version is always 1. The schema enforces
        # UNIQUE(slug, version), so a second authoring attempt with the same
        # slug hits the duplicate-slug error below rather than silently
        # creating version 2. Re-versioning is a future concern — for now
        # the professor edits by creating a new slug or deleting the old one.
        try:
            with connection.transaction():
                tv_id = connection.execute(
                    "INSERT INTO template_versions (slug, version, source_yaml) "
                    "VALUES (%s, 1, %s) RETURNING id",
                    (slug, source_yaml),
                ).fetchone()["id"]
                connection.execute(
                    "INSERT INTO exam_definitions (slug, version, title, template_version_id) "
                    "VALUES (%s, 1, %s, %s) RETURNING id",
                    (slug, title, tv_id),
                ).fetchone()["id"]
        except Exception as exc:
            exc_str = str(exc)
            if "unique" in exc_str.lower() or "duplicate" in exc_str.lower():
                raise ValueError(
                    f"An assessment with slug \u201c{slug}\u201d already exists. "
                    "Choose a different title or slug."
                ) from exc
            raise
        finally:
            connection.close()

        msg = f"Saved assessment \u201c{title}\u201d ({kind}, {len(questions)} questions)."
        if has_computed:
            msg += " Computed-distractor questions require the generation pipeline to produce printable answer sheets."
        self.state.authoring_message = msg
        self.state.message = None

    _FIELD_LABELS: dict[str, str] = {
        "database_url": "Database URL",
        "schema_name": "Schema Name",
        "exam_instance_id": "Selected Exam",
        "artifact_json": "Answer Key File",
        "scan_dir": "Scanned Pages Folder",
        "output_dir": "Save Results To",
    }

    def _require_config(self, *keys: str) -> dict[str, str]:
        missing = [self._FIELD_LABELS.get(k, k) for k in keys if self.state.config.get(k, "").strip() == ""]
        if missing:
            raise ValueError(f"Please fill in: {', '.join(missing)}")
        return self.state.config

    def _handle_open_path(self, form: dict[str, str], *, reveal: bool) -> None:
        path = form.get("path", "").strip()
        if path == "":
            raise ValueError("No path was selected.")
        if path not in _collect_openable_paths(self.state):
            raise ValueError("Selected path is not available in the current workflow state.")

        # This professor-facing GUI currently targets the local Mac workflow only, so
        # Finder/open integration is intentional rather than an accidental dependency.
        command = ["open", "-R", path] if reveal else ["open", path]
        subprocess.run(command, check=True)
        self.state.message = "Opened in Finder." if reveal else "Opened result."

    def _handle_browse_dir(self, form: dict[str, str]) -> None:
        target_field = form.get("target_field", "").strip()
        if not target_field:
            raise ValueError("No target field specified for browse.")
        selected = _native_pick_directory()
        if selected:
            self.state.config[target_field] = selected

    def _handle_browse_file(self, form: dict[str, str]) -> None:
        target_field = form.get("target_field", "").strip()
        if not target_field:
            raise ValueError("No target field specified for browse.")
        selected = _native_pick_file()
        if selected:
            self.state.config[target_field] = selected

    def _refresh_catalog_if_possible(self) -> None:
        if self.state.config.get("database_url", "").strip() == "":
            return
        try:
            self._refresh_catalog()
        except Exception:
            pass

    def _refresh_catalog(self) -> None:
        if self.state.config.get("database_url", "").strip() == "":
            self.state.grading_targets = []
            self.state.assessment_definitions = []
            return
        config = self._require_config("database_url")
        connection = _connect(
            database_url=config["database_url"] or None,
            schema_name=config.get("schema_name") or None,
        )
        try:
            self.state.grading_targets = list_grading_targets(connection=connection)
            self.state.assessment_definitions = list_assessment_definitions(connection=connection)
        finally:
            connection.close()


def render_page(state: GuiState) -> str:
    config = {key: state.config.get(key, "") for key in _CONFIG_FIELDS}
    queue_rows = "".join(_render_review_row(item) for item in state.review_queue)
    export_rows = _render_key_value_rows(
        (
            (_export_label(kind), path)
            for kind, path in sorted(state.export_paths.items())
        )
    )
    summary_intro = _render_summary_intro(state.summary, config.get("scan_dir", ""))
    ingest_rows = _render_key_value_rows(
        (
            ("Detailed Grading Record", state.ingest_result.get("manifest_path")),
            ("Results Folder", state.ingest_result.get("output_dir")),
        )
        if state.ingest_result
        else ()
    )
    stat_cards = _render_stat_cards(state.summary)
    detail_rows = _render_summary_detail_rows(state.summary)

    authoring_msg = ""
    if state.authoring_message:
        authoring_msg = f'<div class="message ok">{escape(state.authoring_message)}</div>'
    af = state.authoring_form
    authoring_questions = _render_authoring_question_fields(af)
    grade_active = "active" if state.active_tab != "author" else ""
    author_active = "active" if state.active_tab == "author" else ""
    a_title = escape(af.get("authoring_title", ""))
    a_slug = escape(af.get("authoring_slug", ""))
    a_kind = af.get("authoring_kind", "exam")
    a_kind_exam_sel = " selected" if a_kind != "quiz" else ""
    a_kind_quiz_sel = " selected" if a_kind == "quiz" else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Professor MC Workflow</title>
  <style>
    :root {{
      --ink: #1f1a17;
      --paper: #f6f3ee;
      --card: #fffdf9;
      --line: #ddd3c5;
      --mist: #e9e2d6;
      --accent: #204e57;
      --accent-soft: #2d6772;
      --secondary: #7a5c37;
      --ok-bg: #e6f4ea;
      --ok-line: #b8d8bf;
      --error-bg: #fdeceb;
      --error-line: #efb8b4;
    }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: var(--paper); color: var(--ink); }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; box-shadow: 0 4px 12px rgba(54, 42, 28, 0.06); }}
    label {{ display: block; font-size: 0.92rem; margin-bottom: 10px; }}
    input, select, button {{ width: 100%; padding: 10px 12px; margin-top: 4px; border-radius: 10px; border: 1px solid #c8bcae; box-sizing: border-box; }}
    button {{ background: var(--accent); color: white; font-weight: 600; cursor: pointer; transition: opacity 120ms ease, transform 120ms ease; }}
    button:hover {{ background: var(--accent-soft); }}
    button.secondary {{ background: var(--secondary); }}
    button[disabled] {{ opacity: 0.72; cursor: wait; transform: none; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eadfce; vertical-align: top; }}
    th {{ font-size: 0.84rem; letter-spacing: 0.02em; color: #5b5148; }}
    .message {{ padding: 12px; border-radius: 10px; margin-bottom: 16px; }}
    .ok {{ background: var(--ok-bg); border: 1px solid var(--ok-line); }}
    .error {{ background: var(--error-bg); border: 1px solid var(--error-line); }}
    .wide {{ grid-column: 1 / -1; }}
    code {{ font-family: ui-monospace, monospace; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .stat-card {{ background: linear-gradient(180deg, #fffdfa 0%, #f8f2e8 100%); border: 1px solid var(--mist); border-radius: 12px; padding: 12px; }}
    .stat-label {{ display: block; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #62574d; margin-bottom: 6px; }}
    .stat-value {{ display: block; font-size: 1.8rem; font-weight: 700; line-height: 1; }}
    .stacked-section + .stacked-section {{ margin-top: 16px; }}
    .section-title {{ margin: 0 0 8px 0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em; color: #62574d; }}
    .support-copy {{ margin: 0 0 12px 0; color: #5d5349; line-height: 1.45; }}
    .action-copy {{ margin: 0 0 8px 0; color: #5d5349; line-height: 1.4; font-size: 0.92rem; }}
    .detail-list {{ list-style: none; margin: 0; padding: 0; }}
    .detail-list li {{ display: flex; justify-content: space-between; gap: 16px; padding: 7px 0; border-bottom: 1px solid #efe6d9; }}
    .detail-list li:last-child {{ border-bottom: none; }}
    .detail-label {{ color: #5d5349; }}
    .detail-value {{ font-weight: 600; text-align: right; overflow-wrap: anywhere; }}
    .path-value {{ display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }}
    .path-chip {{ display: inline-block; max-width: min(100%, 420px); padding: 6px 10px; border-radius: 999px; background: #f3eee6; border: 1px solid #e3d7c7; color: #3d342c; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 0.86rem; line-height: 1.35; text-align: left; overflow-wrap: anywhere; }}
    .path-actions {{ display: inline-flex; gap: 6px; flex-wrap: wrap; }}
    .path-actions form {{ margin: 0; }}
    .path-button {{ width: auto; min-width: 0; margin-top: 0; padding: 6px 10px; border-radius: 999px; font-size: 0.78rem; line-height: 1.1; }}
    details.settings {{ margin: 12px 0 14px; border-top: 1px solid #eadfce; padding-top: 14px; }}
    details.settings summary {{ cursor: pointer; font-weight: 600; color: #4b433a; }}
    details.settings[open] summary {{ margin-bottom: 12px; }}
    .flow-block {{ border: 1px solid #eadfce; border-radius: 12px; padding: 14px; background: #fdfaf5; }}
    .flow-block + .flow-block {{ margin-top: 16px; }}
    .flow-heading {{ margin: 0 0 8px 0; font-size: 1rem; font-weight: 700; color: #3f372f; }}
    .flow-block details.settings {{ margin-bottom: 0; }}
    .secondary-flow details.settings {{ margin: 0; border-top: none; padding-top: 0; }}
    .busy-banner {{ display: none; align-items: center; gap: 10px; padding: 10px 12px; margin-bottom: 12px; border-radius: 10px; background: #edf5f6; border: 1px solid #c8dfe3; color: #234b55; font-weight: 600; }}
    .workflow-spinner {{ width: 16px; height: 16px; border-radius: 999px; border: 2px solid #b5cfd5; border-top-color: var(--accent); animation: workflow-spin 0.8s linear infinite; }}
    body.busy .busy-banner {{ display: flex; }}
    @keyframes workflow-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .field-hint {{ display: block; font-size: 0.78rem; color: #9a8e82; font-weight: 400; margin-top: 2px; margin-bottom: 2px; font-family: ui-monospace, monospace; }}
    .tab-bar {{ display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid var(--line); }}
    .tab-bar button {{ all: unset; border-bottom: 3px solid transparent; margin-bottom: -2px; padding: 10px 20px; font-size: 1rem; font-weight: 600; color: #7a6f63; cursor: pointer; transition: color 120ms ease, border-color 120ms ease; }}
    .tab-bar button:hover {{ color: var(--accent); }}
    .tab-bar button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", () => {{
      const body = document.body;
      for (const form of document.querySelectorAll("form[data-busy-label]")) {{
        form.addEventListener("submit", () => {{
          body.classList.add("busy");
          const banner = document.getElementById("busy-banner");
          if (banner) {{
            const label = form.getAttribute("data-busy-label") || "Working...";
            banner.querySelector("[data-busy-text]").textContent = label;
          }}
          for (const button of document.querySelectorAll("button:not(.tab-bar button)")) {{
            button.disabled = true;
          }}
        }});
      }}
      window.doBrowse = (action, targetField) => {{
        const form = document.createElement("form");
        form.method = "post";
        form.action = action;
        form.style.display = "none";
        // Copy all config hidden fields from the main ingest form
        const mainForm = document.querySelector('form[action="/ingest"]');
        if (mainForm) {{
          for (const input of mainForm.querySelectorAll('input[type="hidden"], input[type="text"], select')) {{
            const h = document.createElement("input");
            h.type = "hidden";
            h.name = input.name;
            h.value = input.value;
            form.appendChild(h);
          }}
        }}
        const tf = document.createElement("input");
        tf.type = "hidden";
        tf.name = "target_field";
        tf.value = targetField;
        form.appendChild(tf);
        document.body.appendChild(form);
        form.submit();
      }};
      window.toggleQuestionMode = (sel, idx) => {{
        const isComputed = sel.value === "computed";
        const sp = document.getElementById("q_" + idx + "_static_panel");
        const cp = document.getElementById("q_" + idx + "_computed_panel");
        if (sp) sp.style.display = isComputed ? "none" : "";
        if (cp) cp.style.display = isComputed ? "" : "none";
      }};
      for (const btn of document.querySelectorAll(".tab-bar button")) {{
        btn.addEventListener("click", () => {{
          const target = btn.getAttribute("data-tab");
          for (const b of document.querySelectorAll(".tab-bar button")) b.classList.remove("active");
          for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
          btn.classList.add("active");
          const panel = document.getElementById("tab-" + target);
          if (panel) panel.classList.add("active");
        }});
      }}

      // --- localStorage draft persistence for the Author tab ---
      const DRAFT_KEY = "mc_authoring_draft";
      const authorForm = document.querySelector('form[action="/author"]');
      if (authorForm) {{
        // Restore saved draft into any field that is still at its default.
        try {{
          const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "{{}}");
          for (const [name, val] of Object.entries(saved)) {{
            const el = authorForm.elements[name];
            if (!el) continue;
            if (el.tagName === "TEXTAREA") {{
              if (!el.value) el.value = val;
            }} else if (el.tagName === "SELECT") {{
              el.value = val;
              // fire mode toggle if this is a question mode selector
              if (name.match(/^q_\d+_mode$/) && typeof el.onchange === "function") {{
                el.onchange.call(el);
              }}
            }} else if (el.type === "text") {{
              if (!el.value) el.value = val;
            }}
          }}
          // re-trigger mode toggles so panels match restored selects
          for (let i = 1; i <= 5; i++) {{
            const sel = authorForm.elements["q_" + i + "_mode"];
            if (sel) window.toggleQuestionMode(sel, i);
          }}
          // Clear stale error banners only inside the Author tab — a
          // restored draft means the author tab is a fresh start, but
          // Grade tab errors should remain visible.
          for (const msg of document.querySelectorAll("#tab-author .message.error")) {{
            msg.remove();
          }}
        }} catch (e) {{ /* ignore corrupt localStorage */ }}

        // Save draft on every change.
        authorForm.addEventListener("input", () => {{
          const data = {{}};
          for (const el of authorForm.elements) {{
            if (!el.name || el.type === "hidden") continue;
            data[el.name] = el.value;
          }}
          try {{ localStorage.setItem(DRAFT_KEY, JSON.stringify(data)); }} catch (e) {{}}
        }});
        authorForm.addEventListener("change", () => {{
          const evt = new Event("input");
          authorForm.dispatchEvent(evt);
        }});

        // Clear draft on successful save (page reloads with a success message).
        const okMsg = document.querySelector("#tab-author .message.ok");
        if (okMsg && okMsg.textContent.toLowerCase().includes("saved")) {{
          try {{ localStorage.removeItem(DRAFT_KEY); }} catch (e) {{}}
        }}
      }}
    }});
  </script>
</head>
<body>
<main>
  <h1>Professor MC Workflow</h1>
  {_render_message_blocks(state)}
  <div class="busy-banner" id="busy-banner" aria-live="polite">
    <span class="workflow-spinner" id="workflow-spinner" aria-hidden="true"></span>
    <span data-busy-text>Working...</span>
  </div>
  <div class="tab-bar">
    <button class="{grade_active}" data-tab="grade">Grade</button>
    <button class="{author_active}" data-tab="author">Author</button>
  </div>
  <div id="tab-grade" class="tab-panel {grade_active}">
  <div class="grid">
    <section class="card">
      <h2>Configuration</h2>
      <div class="flow-block primary-flow">
        <h3 class="flow-heading">Grade Scans</h3>
        <p class="support-copy">Choose the exam and scanned pages you want to grade.</p>
        <form method="post" action="/ingest" data-busy-label="Ingesting scans...">
          {_render_exam_selector(state, config)}
          {_render_browse_input("scan_dir", "Scanned Pages Folder", config["scan_dir"], browse_type="dir", config=config)}
          {_render_browse_input("artifact_json", "Answer Key File", config["artifact_json"], browse_type="file", config=config)}
          {_render_browse_input("output_dir", "Save Results To", config["output_dir"], browse_type="dir", config=config)}
          <details class="settings">
            <summary>Advanced Settings</summary>
            {_render_text_input("database_url", "Database URL", config["database_url"])}
            {_render_text_input("schema_name", "Schema Name", config["schema_name"])}
          </details>
          <p class="action-copy">When you're ready, ingest the scans to load results and any questions that need review.</p>
          <button type="submit">Ingest Scans</button>
        </form>
      </div>
      {_render_create_target_affordance(state, config)}
    </section>

    <section class="card">
      <h2>Current Summary</h2>
      <p class="support-copy">{escape(summary_intro)}</p>
      <div class="stat-grid">{stat_cards}</div>
      <div class="stacked-section">
        <h3 class="section-title">Result Details</h3>
        <ul class="detail-list">{detail_rows or '<li><span class="detail-label">Status</span><span class="detail-value">No workflow result yet.</span></li>'}</ul>
      </div>
      <div class="stacked-section">
        <h3 class="section-title">Saved Files</h3>
        <p class="support-copy">These files keep the record and results from this grading run.</p>
        <ul class="detail-list">{ingest_rows or '<li><span class="detail-label">Saved Files</span><span class="detail-value">No files are available yet.</span></li>'}</ul>
      </div>
      <form method="post" action="/review" data-busy-label="Refreshing review queue...">
        {_render_hidden_config(config)}
        <button class="secondary" type="submit">Refresh Review Queue</button>
      </form>
      <form method="post" action="/export" style="margin-top: 12px;" data-busy-label="Exporting final results...">
        {_render_hidden_config(config)}
        <button class="secondary" type="submit">Export Final Results</button>
      </form>
      <div class="stacked-section">
        <h3 class="section-title">Saved Exports</h3>
        <ul class="detail-list">{export_rows or '<li><span class="detail-label">Saved Exports</span><span class="detail-value">No exports are available yet.</span></li>'}</ul>
      </div>
    </section>

    <section class="card wide">
      <h2>Review Queue</h2>
      <p class="support-copy">Only questions that need a human decision appear here. Choose a resolution for each flagged question, then persist it to finalize the result.</p>
      <form method="post" action="/resolve" data-busy-label="Persisting selected resolutions...">
        {_render_hidden_config(config)}
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Machine Status</th>
              <th>Scan</th>
              <th>Marked</th>
              <th>Resolution</th>
            </tr>
          </thead>
          <tbody>
            {queue_rows or '<tr><td colspan="5">No questions currently require review.</td></tr>'}
          </tbody>
        </table>
        <div style="margin-top: 12px;">
          <button type="submit">Persist Selected Resolutions</button>
        </div>
      </form>
    </section>
  </div>
  </div>
  <div id="tab-author" class="tab-panel {author_active}">
    <section class="card wide">
      <h2>Author Assessment</h2>
      <p class="support-copy">Create a new exam or quiz definition. The authored assessment is saved to the database so it can be used as a grading target.</p>
      {authoring_msg}
      <form method="post" action="/author" data-busy-label="Saving assessment...">
        {_render_hidden_config(config)}
        <div class="grid" style="margin-bottom: 16px;">
          <label>Title<span class="field-hint">e.g. CHM 141 Quiz 3</span><input type="text" name="authoring_title" value="{a_title}"></label>
          <label>Slug<span class="field-hint">auto-derived from title if blank</span><input type="text" name="authoring_slug" value="{a_slug}"></label>
        </div>
        <label style="margin-bottom: 16px;">Assessment Kind
          <select name="authoring_kind">
            <option value="exam"{a_kind_exam_sel}>Exam</option>
            <option value="quiz"{a_kind_quiz_sel}>Quiz</option>
          </select>
        </label>
        <h3 class="section-title">Multiple-Choice Questions</h3>
        <details style="margin-bottom: 16px; border: 1px solid var(--mist); border-radius: 10px; padding: 12px; background: linear-gradient(180deg, #fffdfa 0%, #f8f2e8 100%);">
          <summary style="cursor: pointer; font-weight: 600; color: #4b433a; font-size: 0.92rem;">Example: computed-distractor question</summary>
          <div style="margin-top: 10px; font-size: 0.88rem; line-height: 1.6; color: #4b433a;">
            <p style="margin: 0 0 8px;">A computed-distractor question lets the system generate different numbers for each student. You write the prompt with variable placeholders, declare the variables, give the correct answer as an expression, and list wrong-answer expressions as distractors.</p>
            <div style="background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; font-family: ui-monospace, monospace; font-size: 0.82rem; line-height: 1.7;">
              <div><strong>Prompt:</strong> Mercury has a density of {{{{density}}}} g/cm\u00b3. What is the volume of {{{{mass}}}} g of Hg?</div>
              <div style="margin-top: 6px;"><strong>Variables (YAML):</strong></div>
              <div style="padding-left: 12px;">mass: {{type: float, min: 50.0, max: 150.0, step: 1.0}}</div>
              <div style="padding-left: 12px;">density: {{type: float, min: 10.0, max: 16.0, step: 0.1}}</div>
              <div style="margin-top: 6px;"><strong>Answer expression:</strong> mass / density</div>
              <div style="margin-top: 6px;"><strong>Distractors:</strong></div>
              <div style="padding-left: 12px;">1. density / mass <span style="color: #9a8e82;">\u2014 inverted division</span></div>
              <div style="padding-left: 12px;">2. mass * density <span style="color: #9a8e82;">\u2014 wrong operation</span></div>
              <div style="padding-left: 12px;">3. mass + density <span style="color: #9a8e82;">\u2014 nonsensical sum</span></div>
            </div>
            <p style="margin: 8px 0 0; color: #7a6f63;">Each student gets different values for <code>mass</code> and <code>density</code>, so both the correct answer and every distractor are unique per exam. Set the mode to <strong>Computed Distractors</strong> on any question below to use this shape.</p>
          </div>
        </details>
        {authoring_questions}
        <div style="margin-top: 16px;">
          <button type="submit">Save Assessment</button>
        </div>
      </form>
    </section>
  </div>
</main>
</body>
</html>"""


def serve_mc_workflow_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    initial_state: GuiState | None = None,
    open_browser: bool = False,
) -> int:
    import socketserver
    from wsgiref.simple_server import WSGIRequestHandler

    class _ThreadedWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
        """Handle each request in a separate thread so blocking ops
        (like native file pickers via osascript) don't freeze the browser."""
        daemon_threads = True

    app = McWorkflowGuiApp(initial_state=initial_state)
    url = f"http://{host}:{port}"
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    server = _ThreadedWSGIServer((host, port), WSGIRequestHandler)
    server.set_app(app)
    server.serve_forever()
    return 0


def _connect(*, database_url: str | None, schema_name: str | None) -> Any:
    connection = create_connection(database_url)
    if schema_name:
        connection.execute(
            sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(_require_schema_identifier(schema_name))
            )
        )
    return connection


def _extract_simple_resolutions(form: dict[str, str]) -> dict[str, str | None]:
    resolutions: dict[str, str | None] = {}
    for key, value in form.items():
        if not key.startswith("resolution__"):
            continue
        question_id = key.removeprefix("resolution__")
        if value == "":
            continue
        resolutions[question_id] = None if value == "__BLANK__" else value
    return resolutions


def _parse_post_body(environ: dict[str, Any]) -> dict[str, str]:
    length = int(environ.get("CONTENT_LENGTH", "0") or "0")
    body = environ["wsgi.input"].read(length).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _render_message_blocks(state: GuiState) -> str:
    blocks: list[str] = []
    if state.message:
        blocks.append(f'<div class="message ok">{escape(state.message)}</div>')
    if state.error:
        blocks.append(f'<div class="message error">{escape(state.error)}</div>')
    return "".join(blocks)


def _render_text_input(name: str, label: str, value: str) -> str:
    return (
        f'<label>{escape(label)}'
        f'<input type="text" name="{escape(name)}" value="{escape(value)}"></label>'
    )


def _render_browse_input(
    name: str, label: str, value: str, *, browse_type: str, config: dict[str, str],
) -> str:
    """Render a text input with a Browse button that pops a native file/dir picker.

    browse_type is either "dir" or "file". The Browse button uses JS to
    submit a standalone form (not nested inside the parent form, which
    browsers reject as invalid HTML).
    """
    action = "/browse-dir" if browse_type == "dir" else "/browse-file"
    hidden = _render_hidden_config(config)
    btn_id = f"browse__{escape(name)}"
    return (
        f'<label>{escape(label)}'
        f'<div style="display:flex;gap:6px;align-items:stretch;">'
        f'<input type="text" name="{escape(name)}" value="{escape(value)}" style="flex:1;">'
        f'<button type="button" id="{btn_id}" class="secondary" '
        f'style="width:auto;padding:8px 14px;white-space:nowrap;margin-top:4px;" '
        f'onclick="doBrowse(\'{action}\', \'{escape(name)}\')">Browse</button>'
        f'</div></label>'
    )


def _render_select_input(name: str, label: str, value: str, options: list[tuple[str, str]]) -> str:
    rendered_options = []
    for option_value, option_label in options:
        selected = ' selected' if option_value == value else ""
        rendered_options.append(
            f'<option value="{escape(option_value)}"{selected}>{escape(option_label)}</option>'
        )
    return (
        f'<label>{escape(label)}'
        f'<select name="{escape(name)}">{"".join(rendered_options)}</select></label>'
    )


def _render_hidden_config(config: dict[str, str]) -> str:
    return "".join(
        f'<input type="hidden" name="{escape(key)}" value="{escape(value)}">'
        for key, value in config.items()
    )


def _render_exam_selector(state: GuiState, config: dict[str, str]) -> str:
    if state.grading_targets:
        selected = config.get("exam_instance_id", "").strip()
        if selected == "":
            selected = str(state.grading_targets[0]["exam_instance_id"])
            config["exam_instance_id"] = selected
        options = [
            (str(target["exam_instance_id"]), target["label"])
            for target in state.grading_targets
        ]
        return _render_select_input("exam_instance_id", "Selected Exam", selected, options)
    return _render_text_input("exam_instance_id", "Selected Exam", config["exam_instance_id"])


def _render_create_target_form(state: GuiState, config: dict[str, str]) -> str:
    if not state.assessment_definitions:
        return (
            '<p class="support-copy">No assessments are available yet. Add one in the authoring workflow before creating a grading target here.</p>'
        )
    definition_options = [
        (str(definition["exam_definition_id"]), definition["label"])
        for definition in state.assessment_definitions
    ]
    first_definition = definition_options[0][0]
    return (
        '<p class="support-copy">Create from assessment.</p>'
        f'<form method="post" action="/create-target" data-busy-label="Creating exam...">{_render_hidden_config(config)}'
        f'{_render_select_input("new_exam_definition_id", "Assessment Template", first_definition, definition_options)}'
        f'{_render_text_input("new_target_name", "Exam Name", "")}'
        '<button class="secondary" type="submit">Create Exam</button>'
        "</form>"
    )


def _render_create_target_affordance(state: GuiState, config: dict[str, str]) -> str:
    if state.grading_targets:
        return (
            '<div class="flow-block secondary-flow">'
            '<h3 class="flow-heading">Need a Different Exam?</h3>'
            '<p class="support-copy">If the exam you want is not listed, create a new one here.</p>'
            '<details class="settings">'
            '<summary>Create new exam</summary>'
            f"{_render_create_target_form(state, config)}"
            "</details>"
            "</div>"
        )
    return (
        '<div class="flow-block secondary-flow">'
        '<h3 class="flow-heading">Create New Exam</h3>'
        '<p class="support-copy">No exams are available yet. Create one so new scans have somewhere to land.</p>'
        f"{_render_create_target_form(state, config)}"
        '</div>'
    )


def _render_review_row(item: dict[str, Any]) -> str:
    options = "".join(
        f'<option value="{escape(choice)}">{escape(_choice_label(choice))}</option>'
        for choice in _RESOLUTION_CHOICES
    )
    marked = ", ".join(item.get("marked_bubble_labels", []))
    return (
        "<tr>"
        f"<td><code>{escape(item['question_id'])}</code></td>"
        f"<td>{escape(item['machine_status'])}</td>"
        f"<td>{escape(item['scan_id'])} / p{escape(str(item['page_number']))}</td>"
        f"<td>{escape(marked)}</td>"
        f'<td><select name="resolution__{escape(item["question_id"])}">{options}</select></td>'
        "</tr>"
    )


def _choice_label(choice: str) -> str:
    if choice == "":
        return "Leave unchanged"
    if choice == "__BLANK__":
        return "Blank"
    return choice


def _require_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _native_pick_directory() -> str:
    """Pop a native macOS directory picker via AppleScript. Returns the selected path or ''."""
    script = (
        'tell application "System Events" to set frontmost of process "osascript" to true\n'
        'POSIX path of (choose folder with prompt "Choose a folder")'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True,
    )
    return result.stdout.strip().rstrip("/") if result.returncode == 0 else ""


def _native_pick_file() -> str:
    """Pop a native macOS file picker via AppleScript. Returns the selected path or ''."""
    script = (
        'tell application "System Events" to set frontmost of process "osascript" to true\n'
        'POSIX path of (choose file with prompt "Choose a file")'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_schema_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            "schema_name must be a simple Postgres identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    return value


_DISTRACTOR_SLOT_COUNT = 4


def _render_authoring_question_fields(saved: dict[str, str] | None = None) -> str:
    """Render the initial set of MC question input blocks for the authoring form.

    If *saved* is provided (from a failed POST round-trip), field values
    are restored so the professor does not lose their work.
    """
    saved = saved or {}
    blocks: list[str] = []
    for idx in range(1, _INITIAL_QUESTION_COUNT + 1):
        def _v(name: str) -> str:
            return escape(saved.get(name, ""))

        mode = saved.get(f"q_{idx}_mode", "static")
        is_computed = mode == "computed"
        static_display = "none" if is_computed else ""
        computed_display = "" if is_computed else "none"

        choice_inputs = "".join(
            f'<label>{label}<input type="text" name="q_{idx}_choice_{label}" value="{_v(f"q_{idx}_choice_{label}")}" placeholder="Choice {label}"></label>'
            for label in _CHOICE_LABELS
        )
        correct_val = saved.get(f"q_{idx}_correct", "")
        correct_options = "".join(
            f'<option value="{label}"{" selected" if label == correct_val else ""}>{label}</option>'
            for label in _CHOICE_LABELS
        )
        distractor_inputs = "".join(
            f'<label>Distractor {di}'
            f'<span class="field-hint">e.g. density / mass</span>'
            f'<input type="text" name="q_{idx}_distractor_{di}" value="{_v(f"q_{idx}_distractor_{di}")}"></label>'
            for di in range(1, _DISTRACTOR_SLOT_COUNT + 1)
        )
        blocks.append(
            f'<div class="card" style="margin-bottom: 12px;">'
            f"<h4 style=\"margin: 0 0 8px;\">Question {idx}</h4>"
            f'<label>Prompt'
            f'<span class="field-hint">Use {{{{var}}}} for variable placeholders</span>'
            f'<input type="text" name="q_{idx}_prompt" value="{_v(f"q_{idx}_prompt")}"></label>'
            f'<label>Mode<select name="q_{idx}_mode" onchange="toggleQuestionMode(this, {idx})">'
            f'<option value="static"{"" if is_computed else " selected"}>Static Choices</option>'
            f'<option value="computed"{" selected" if is_computed else ""}>Computed Distractors</option>'
            f"</select></label>"
            f'<div id="q_{idx}_static_panel" style="display:{static_display};">'
            f"{choice_inputs}"
            f'<label>Correct Answer<select name="q_{idx}_correct"><option value="">—</option>{correct_options}</select></label>'
            f"</div>"
            f'<div id="q_{idx}_computed_panel" style="display:{computed_display};">'
            f'<label>Variables (YAML)'
            f'<span class="field-hint">mass: {{type: float, min: 10.0, max: 99.0, step: 0.1}}</span>'
            f'<textarea name="q_{idx}_variables" rows="3" style="width:100%;font-family:ui-monospace,monospace;font-size:0.88rem;">{_v(f"q_{idx}_variables")}</textarea></label>'
            f'<label>Answer Expression'
            f'<span class="field-hint">e.g. mass / density</span>'
            f'<input type="text" name="q_{idx}_answer_expr" value="{_v(f"q_{idx}_answer_expr")}"></label>'
            f"{distractor_inputs}"
            f"</div>"
            f"</div>"
        )
    return "".join(blocks)


def _extract_authored_questions(form: dict[str, str]) -> list[dict[str, Any]]:
    """Parse numbered MC question fields from the form into a list of question dicts.

    Scans all slots up to _INITIAL_QUESTION_COUNT and collects every slot
    that has a non-empty prompt, regardless of gaps between them. Each
    question carries a ``mode`` field (``"static"`` or ``"computed"``)
    that determines which fields are populated.
    """
    questions: list[dict[str, Any]] = []
    question_number = 0
    for idx in range(1, _INITIAL_QUESTION_COUNT + 1):
        prompt = form.get(f"q_{idx}_prompt", "").strip()
        if not prompt:
            continue
        question_number += 1
        mode = form.get(f"q_{idx}_mode", "static").strip()

        if mode == "computed":
            variables_yaml = form.get(f"q_{idx}_variables", "").strip()
            answer_expr = form.get(f"q_{idx}_answer_expr", "").strip()
            distractor_exprs: list[str] = []
            for di in range(1, _DISTRACTOR_SLOT_COUNT + 1):
                expr = form.get(f"q_{idx}_distractor_{di}", "").strip()
                if expr:
                    distractor_exprs.append(expr)
            questions.append({
                "id": f"mc-{question_number}",
                "form_slot": idx,
                "prompt": prompt,
                "mode": "computed",
                "variables_yaml": variables_yaml,
                "answer_expr": answer_expr,
                "distractor_exprs": distractor_exprs,
            })
        else:
            choices: dict[str, str] = {}
            for label in _CHOICE_LABELS:
                text = form.get(f"q_{idx}_choice_{label}", "").strip()
                if text:
                    choices[label] = text
            correct = form.get(f"q_{idx}_correct", "").strip()
            questions.append({
                "id": f"mc-{question_number}",
                "form_slot": idx,
                "prompt": prompt,
                "mode": "static",
                "choices": choices,
                "correct": correct,
            })
    return questions


def _build_template_yaml(
    *,
    slug: str,
    title: str,
    kind: str,
    questions: list[dict[str, Any]],
) -> str:
    """Build a minimal valid YAML template string from authored question data."""
    import yaml  # PyYAML — declared repo dependency, used throughout template_schema.py

    built_questions: list[dict[str, Any]] = []
    for q in questions:
        base: dict[str, Any] = {
            "id": q["id"],
            "points": 2,
            "answer_type": "multiple_choice",
            "prompt": q["prompt"],
        }
        if q.get("mode") == "computed":
            variables_raw = q.get("variables_yaml", "")
            if variables_raw:
                parsed_vars = yaml.safe_load(variables_raw)
                if isinstance(parsed_vars, dict):
                    base["variables"] = parsed_vars
            base["answer"] = {"expr": q["answer_expr"]}
            base["distractors"] = {
                "common_errors": [{"expr": e} for e in q["distractor_exprs"]],
            }
        else:
            base["choices"] = q["choices"]
            base["correct"] = q["correct"]
        built_questions.append(base)

    template: dict[str, Any] = {
        "slug": slug,
        "title": title,
        "kind": kind,
        "sections": [
            {
                "id": "mc",
                "title": "Multiple Choice",
                "questions": built_questions,
            }
        ],
    }
    return yaml.dump(template, default_flow_style=False, sort_keys=False)


def _render_summary_text(exported: dict[str, Any]) -> str:
    lines = [
        f"exam_instance_id: {exported['exam_instance_id']}",
        f"mc_scan_session_id: {exported['mc_scan_session_id']}",
        f"session_ordinal: {exported['session_ordinal']}",
    ]
    for key, value in sorted(exported["summary"].items()):
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _render_summary_intro(summary: dict[str, Any] | None, scan_dir: str) -> str:
    if not summary:
        return "Ingest a set of scans to see the current grading summary."
    scan_name = Path(scan_dir).name.strip() if scan_dir else ""
    if scan_name:
        return f"Showing results for scan set: {scan_name}."
    return "Showing results for the current scan set."


def _render_stat_cards(summary: dict[str, Any] | None) -> str:
    stats = (
        ("Matched Pages", summary.get("matched") if summary else None),
        ("Questions Correct", summary.get("correct") if summary else None),
        ("Questions Incorrect", summary.get("incorrect") if summary else None),
        ("Questions Needing Review", summary.get("unresolved_review_required") if summary else None),
    )
    return "".join(
        "<div class=\"stat-card\">"
        f"<span class=\"stat-label\">{escape(label)}</span>"
        f"<span class=\"stat-value\">{escape(str(value if value is not None else '—'))}</span>"
        "</div>"
        for label, value in stats
    )


def _render_summary_detail_rows(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    ordered_keys = (
        ("matched", "Matched Pages"),
        ("unmatched", "Unmatched Scans"),
        ("ambiguous", "Ambiguous Scans"),
        ("question_count", "Questions"),
        ("correct", "Questions Correct"),
        ("incorrect", "Questions Incorrect"),
        ("blank", "Blank"),
        ("resolved_by_review", "Resolved by Review"),
        ("unresolved_review_required", "Questions Needing Review"),
    )
    return _render_key_value_rows(
        (label, summary[key]) for key, label in ordered_keys if key in summary
    )


def _render_key_value_rows(items: Any) -> str:
    rows = []
    for label, value in items:
        if value is None:
            continue
        rendered_value = _render_detail_value(label, value)
        rows.append(
            "<li>"
            f"<span class=\"detail-label\">{escape(str(label))}</span>"
            f"{rendered_value}"
            "</li>"
        )
    return "".join(rows)


def _render_detail_value(label: Any, value: Any) -> str:
    value_text = str(value)
    if _looks_like_openable_path(value_text):
        return (
            '<span class="detail-value path-value">'
            f'<span class="path-chip">{escape(value_text)}</span>'
            '<span class="path-actions">'
            f'{_render_path_action("/open-path", value_text, "Open")}'
            f'{_render_path_action("/reveal-path", value_text, "Show in Finder")}'
            "</span>"
            "</span>"
        )
    return f'<span class="detail-value">{escape(value_text)}</span>'


def _render_path_action(action: str, path: str, label: str) -> str:
    return (
        f'<form method="post" action="{escape(action)}" data-busy-label="{escape(label)}...">'
        f'<input type="hidden" name="path" value="{escape(path)}">'
        f'<button class="secondary path-button" type="submit">{escape(label)}</button>'
        "</form>"
    )


def _looks_like_openable_path(value: str) -> bool:
    return value.startswith("/")


def _export_label(kind: str) -> str:
    mapping = {
        "csv": "Spreadsheet",
        "json": "Detailed Results",
        "txt": "Quick Summary",
    }
    return mapping.get(kind.lower(), kind.upper())


def _collect_openable_paths(state: GuiState) -> set[str]:
    paths: set[str] = set()
    for collection in (
        state.export_paths.values(),
        (
            state.ingest_result.get("manifest_path")
            if state.ingest_result
            else None,
            state.ingest_result.get("output_dir")
            if state.ingest_result
            else None,
        ),
    ):
        for value in collection:
            if isinstance(value, str) and value.strip():
                paths.add(value.strip())
    return paths
