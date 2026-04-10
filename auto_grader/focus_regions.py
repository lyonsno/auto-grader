"""Load and save focus-region annotations as YAML.

This module is the one contract shared by ``scripts/smoke_vlm.py``
(runtime consumer) and ``scripts/annotate_focus_regions.py``
(interactive editor). Both paths use the same loader and saver so
interactive and batch use share a single file format.

The YAML shape is::

    regions:
      exam_id/question_id:
        page: int
        x: float          # all coords are normalized [0, 1]
        y: float
        width: float
        height: float
        source: str       # provenance tag
      ...

Keys are flat ``"exam_id/question_id"`` strings (not nested mappings)
so they round-trip through any YAML dumper without reshuffling.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from auto_grader.eval_harness import FocusRegion


def load_focus_regions(path: Path | str) -> dict[tuple[str, str], FocusRegion]:
    """Load a focus-regions YAML file into a dict keyed by
    ``(exam_id, question_id)`` tuples.

    Missing file → empty dict (so a first invocation of the annotation
    tool can start from nothing without the caller having to check).
    Malformed top-level shape → raises ``ValueError`` with a caller-
    helpful message.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(
            f"focus regions file at {path} is not a YAML mapping at top level"
        )
    regions_block = document.get("regions", {})
    if not isinstance(regions_block, dict):
        raise ValueError(
            f"focus regions file at {path} is missing a top-level `regions:` mapping"
        )
    out: dict[tuple[str, str], FocusRegion] = {}
    for flat_key, payload in regions_block.items():
        if not isinstance(flat_key, str) or "/" not in flat_key:
            raise ValueError(
                f"focus regions key {flat_key!r} is not a valid "
                f"'exam_id/question_id' string"
            )
        exam_id, question_id = flat_key.split("/", 1)
        if not isinstance(payload, dict):
            raise ValueError(
                f"focus regions entry for {flat_key!r} is not a mapping"
            )
        out[(exam_id, question_id)] = FocusRegion(
            page=int(payload["page"]),
            x=float(payload["x"]),
            y=float(payload["y"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
            source=str(payload.get("source", "unknown")),
        )
    return out


def save_focus_regions(
    path: Path | str,
    regions: dict[tuple[str, str], FocusRegion],
    *,
    header_comment: str | None = None,
) -> None:
    """Write a focus-regions dict back to YAML.

    Entries are sorted by ``(exam_id, question_id)`` so diffs across
    runs stay stable. ``header_comment`` is rendered as a leading
    comment block so the file explains itself when opened by hand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, dict[str, object]]] = {"regions": {}}
    for (exam_id, question_id), region in sorted(regions.items()):
        flat_key = f"{exam_id}/{question_id}"
        payload["regions"][flat_key] = {
            "page": int(region.page),
            "x": float(region.x),
            "y": float(region.y),
            "width": float(region.width),
            "height": float(region.height),
            "source": region.source,
        }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    with open(path, "w") as handle:
        if header_comment is not None:
            for line in header_comment.splitlines():
                handle.write(f"# {line}\n" if line else "#\n")
            handle.write("\n")
        handle.write(body)


DEFAULT_FOCUS_REGIONS_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "focus_regions.yaml"
)
