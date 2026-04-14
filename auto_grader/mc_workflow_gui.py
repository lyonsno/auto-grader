"""Thin local web GUI for the professor-facing MC workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import io
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs
import webbrowser
from wsgiref.simple_server import make_server

from psycopg import sql

from auto_grader.db import create_connection
from auto_grader.mc_workflow import (
    export_results,
    get_review_queue,
    ingest_and_persist_from_scan_dir,
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


@dataclass
class GuiState:
    config: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    error: str | None = None
    ingest_result: dict[str, Any] | None = None
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    export_paths: dict[str, str] = field(default_factory=dict)


class McWorkflowGuiApp:
    def __init__(self, *, initial_state: GuiState | None = None) -> None:
        self.state = initial_state or GuiState()

    def __call__(self, environ: dict[str, Any], start_response) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        if method == "POST":
            form = _parse_post_body(environ)
            self._update_config(form)
            try:
                if path == "/ingest":
                    self._handle_ingest()
                elif path == "/review":
                    self._handle_review()
                elif path == "/resolve":
                    self._handle_resolve(form)
                elif path == "/export":
                    self._handle_export()
                else:
                    raise ValueError(f"Unknown action path: {path}")
                self.state.error = None
            except Exception as exc:
                self.state.error = str(exc)
                self.state.message = None

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

    def _require_config(self, *keys: str) -> dict[str, str]:
        missing = [key for key in keys if self.state.config.get(key, "").strip() == ""]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        return self.state.config


def render_page(state: GuiState) -> str:
    config = {key: state.config.get(key, "") for key in _CONFIG_FIELDS}
    queue_rows = "".join(_render_review_row(item) for item in state.review_queue)
    export_rows = _render_key_value_rows(
        (
            (kind.upper(), path)
            for kind, path in sorted(state.export_paths.items())
        )
    )
    summary_intro = _render_summary_intro(state.summary, config.get("scan_dir", ""))
    ingest_rows = _render_key_value_rows(
        (
            ("Session ID", state.ingest_result.get("mc_scan_session_id")),
            ("Manifest", state.ingest_result.get("manifest_path")),
            ("Output", state.ingest_result.get("output_dir")),
        )
        if state.ingest_result
        else ()
    )
    stat_cards = _render_stat_cards(state.summary)
    detail_rows = _render_summary_detail_rows(state.summary)

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
    details.settings {{ margin: 12px 0 14px; border-top: 1px solid #eadfce; padding-top: 14px; }}
    details.settings summary {{ cursor: pointer; font-weight: 600; color: #4b433a; }}
    details.settings[open] summary {{ margin-bottom: 12px; }}
    .busy-banner {{ display: none; align-items: center; gap: 10px; padding: 10px 12px; margin-bottom: 12px; border-radius: 10px; background: #edf5f6; border: 1px solid #c8dfe3; color: #234b55; font-weight: 600; }}
    .workflow-spinner {{ width: 16px; height: 16px; border-radius: 999px; border: 2px solid #b5cfd5; border-top-color: var(--accent); animation: workflow-spin 0.8s linear infinite; }}
    body.busy .busy-banner {{ display: flex; }}
    @keyframes workflow-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
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
          for (const button of document.querySelectorAll("button")) {{
            button.disabled = true;
          }}
        }});
      }}
    }});
  </script>
</head>
<body>
<main>
  <h1>Professor MC Workflow</h1>
  <p>Thin local GUI over the landed ingest, review, resolve, and export workflow.</p>
  {_render_message_blocks(state)}
  <div class="busy-banner" id="busy-banner" aria-live="polite">
    <span class="workflow-spinner" id="workflow-spinner" aria-hidden="true"></span>
    <span data-busy-text>Working...</span>
  </div>
  <div class="grid">
    <section class="card">
      <h2>Configuration</h2>
      <p class="support-copy">Choose the exam record and scanned pages you want to grade.</p>
      <form method="post" action="/ingest" data-busy-label="Ingesting scans...">
        {_render_text_input("exam_instance_id", "Exam Instance ID", config["exam_instance_id"])}
        {_render_text_input("scan_dir", "Scan Directory", config["scan_dir"])}
        <details class="settings">
          <summary>Workflow Settings</summary>
          {_render_text_input("database_url", "Database URL", config["database_url"])}
          {_render_text_input("schema_name", "Schema Name", config["schema_name"])}
          {_render_text_input("artifact_json", "Artifact JSON", config["artifact_json"])}
          {_render_text_input("output_dir", "Output Directory", config["output_dir"])}
        </details>
        <p class="action-copy">When you're ready, ingest the scans to load results and any questions that need review.</p>
        <button type="submit">Ingest Scans</button>
      </form>
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
        <h3 class="section-title">Workflow Artifacts</h3>
        <ul class="detail-list">{ingest_rows or '<li><span class="detail-label">Ingest</span><span class="detail-value">No ingest run yet.</span></li>'}</ul>
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
        <h3 class="section-title">Export Files</h3>
        <ul class="detail-list">{export_rows or '<li><span class="detail-label">Export</span><span class="detail-value">No export run yet.</span></li>'}</ul>
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
    app = McWorkflowGuiApp(initial_state=initial_state)
    url = f"http://{host}:{port}"
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    with make_server(host, port, app) as server:
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


def _render_hidden_config(config: dict[str, str]) -> str:
    return "".join(
        f'<input type="hidden" name="{escape(key)}" value="{escape(value)}">'
        for key, value in config.items()
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


def _require_schema_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            "schema_name must be a simple Postgres identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    return value


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
        rows.append(
            "<li>"
            f"<span class=\"detail-label\">{escape(str(label))}</span>"
            f"<span class=\"detail-value\">{escape(str(value))}</span>"
            "</li>"
        )
    return "".join(rows)
