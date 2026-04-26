"""Current model routing defaults for auto-grader inference clients."""

from __future__ import annotations

DEFAULT_GRAPHEUS_BASE_URL = "http://macbook-pro-2.local:8090"
DEFAULT_GRADER_MODEL = "qwen3p5-35B-A3B"


def autograder_grapheus_headers(
    *,
    pathway: str,
    component: str,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    """Headers Grapheus can use to identify auto-grader traffic."""
    headers = {
        "X-AutoGrader-Repo": "auto-grader",
        "X-AutoGrader-Pathway": pathway,
        "X-AutoGrader-Component": component,
    }
    if model:
        headers["X-AutoGrader-Model"] = model
    if run_id:
        headers["X-AutoGrader-Run-ID"] = run_id
    return headers
