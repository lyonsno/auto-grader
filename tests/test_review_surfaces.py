import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
MANIFEST = DOCS / "review_surfaces.toml"


def load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def repo_relative_path(rel_path: str) -> Path:
    path = (REPO_ROOT / rel_path).resolve()
    repo_root = REPO_ROOT.resolve()
    if not path.is_relative_to(repo_root):
        raise ValueError(f"path escapes repo root: {rel_path}")
    return path


def test_review_surface_paths_are_repo_relative() -> None:
    assert repo_relative_path("docs/review-authority-surfaces.md") == (
        DOCS / "review-authority-surfaces.md"
    )
    assert repo_relative_path("auto_grader/vlm_inference.py") == (
        REPO_ROOT / "auto_grader/vlm_inference.py"
    )
    with pytest.raises(ValueError):
        repo_relative_path("../outside.md")


def test_review_surface_manifest_declares_clean_eval_scan_authority() -> None:
    manifest = load_manifest()
    surfaces = manifest["surfaces"]

    clean_eval_scan = surfaces["clean_eval_scan_authority"]
    assert clean_eval_scan["canonical_surface"] == "docs/review-authority-surfaces.md"
    assert clean_eval_scan["authoritative_paths"] == ["auto_grader/vlm_inference.py"]
    assert clean_eval_scan["forbidden_fallback_aliases"] == ["15 blue.pdf"]
    assert clean_eval_scan["default_review_action"] == "waive-contaminated-fallback"


def test_clean_eval_scan_review_surface_doc_states_authority_and_boundary() -> None:
    manifest = load_manifest()
    clean_eval_scan = manifest["surfaces"]["clean_eval_scan_authority"]
    canonical_surface = repo_relative_path(clean_eval_scan["canonical_surface"])
    assert canonical_surface.is_file()

    doc_text = canonical_surface.read_text(encoding="utf-8")
    assert "Clean Eval Scan Authority" in doc_text
    assert "auto_grader/vlm_inference.py" in doc_text
    assert "15 blue_professor_markings_hidden.pdf" in doc_text
    assert "15 blue.pdf" in doc_text
    assert "Refusing contaminated fallback" in doc_text
    assert "separate explicit mode" in doc_text.lower()


def test_agents_documents_make_this_durable_for_review_route() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "Make this durable for review" in agents
    assert "Topothesia review surfaces" in agents
    assert "Prilosec" in agents
    assert "docs/review-authority-surfaces.md" in agents
