"""Interactive focus-region annotation tool.

Walks a list of targets on a single PDF page, letting the operator
click-drag a bounding box for each, and writes the result back into a
focus-regions YAML file.

Shape: one invocation handles one (PDF, page) pair and however many
targets you want to draw on it. Agents loop the tool over multiple
pages by calling it repeatedly with different --page and --targets.

Usage:
    python scripts/annotate_focus_regions.py \\
        --pdf ~/dev/auto-grader-assets/scans/"27 blue 2023.pdf" \\
        --page 1 \\
        --targets 27-blue-2023/fr-3,27-blue-2023/fr-5b \\
        [--config eval/focus_regions.yaml]

Keys while the window is open:
    click-drag      draw new box (replaces any existing box for this target)
    Enter / Space   accept current box, advance to next target
    s / Right       skip this target (keep existing box unchanged)
    r / Backspace   clear current drawing, re-drag
    q / Esc         save and exit immediately

On exit the config file is updated in place. Targets not in --targets
are left untouched in the file.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

import fitz

# scripts/ is not on sys.path by default; add the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from auto_grader.eval_harness import FocusRegion  # noqa: E402
from auto_grader.focus_regions import (  # noqa: E402
    DEFAULT_FOCUS_REGIONS_PATH,
    load_focus_regions,
    save_focus_regions,
)


_MAX_DISPLAY_WIDTH = 1400
_MAX_DISPLAY_HEIGHT = 1600
_DPI = 160


def _parse_targets(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "/" not in token:
            raise SystemExit(
                f"--targets entry {token!r} is not in 'exam_id/question_id' form"
            )
        exam_id, question_id = token.split("/", 1)
        out.append((exam_id.strip(), question_id.strip()))
    if not out:
        raise SystemExit("--targets was empty")
    return out


def _rasterize_page(pdf_path: Path, page_number: int) -> tuple[bytes, int, int]:
    """Rasterize a PDF page (1-indexed) to PNG bytes and return
    (png_bytes, width_px, height_px) at the configured DPI."""
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise SystemExit(
                f"--page {page_number} out of range for {pdf_path} "
                f"(page count = {document.page_count})"
            )
        page = document[page_number - 1]
        pix = page.get_pixmap(dpi=_DPI)
        return pix.tobytes("png"), pix.width, pix.height


def _compute_display_scale(page_width: int, page_height: int) -> float:
    scale_w = _MAX_DISPLAY_WIDTH / page_width
    scale_h = _MAX_DISPLAY_HEIGHT / page_height
    return min(1.0, scale_w, scale_h)


class AnnotationApp:
    def __init__(
        self,
        *,
        page_png: bytes,
        page_width_px: int,
        page_height_px: int,
        pdf_page_number: int,
        targets: list[tuple[str, str]],
        existing: dict[tuple[str, str], FocusRegion],
    ) -> None:
        self.page_width_px = page_width_px
        self.page_height_px = page_height_px
        self.pdf_page_number = pdf_page_number
        self.targets = targets
        self.existing = dict(existing)  # copy; will be mutated on accept
        self.result: dict[tuple[str, str], FocusRegion] = dict(existing)
        self.current_index = 0
        self.scale = _compute_display_scale(page_width_px, page_height_px)
        self.display_width = int(round(page_width_px * self.scale))
        self.display_height = int(round(page_height_px * self.scale))

        self.root = tk.Tk()
        self.root.title("focus region annotator")
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            font=("TkFixedFont", 13),
            padx=10,
            pady=6,
        )
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(
            self.root,
            width=self.display_width,
            height=self.display_height,
            bg="#202020",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP)

        # tk.PhotoImage handles PNG natively in modern Tk; if the page
        # is huge we scale it down to the display resolution via fitz
        # before handing it off, since PhotoImage doesn't scale.
        self.tk_image = self._build_display_image(page_png)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Rectangle state.
        self.drag_start: tuple[int, int] | None = None
        self.current_rect_id: int | None = None
        self.existing_rect_id: int | None = None
        self.current_normalized: tuple[float, float, float, float] | None = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Return>", self._on_accept)
        self.root.bind("<space>", self._on_accept)
        self.root.bind("<s>", self._on_skip)
        self.root.bind("<Right>", self._on_skip)
        self.root.bind("<r>", self._on_reset_current)
        self.root.bind("<BackSpace>", self._on_reset_current)
        self.root.bind("<q>", self._on_quit)
        self.root.bind("<Escape>", self._on_quit)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._load_current_target()

    def _build_display_image(self, page_png: bytes) -> tk.PhotoImage:
        if self.scale >= 1.0:
            return tk.PhotoImage(data=page_png)
        # Re-rasterize at the display scale directly so we don't have to
        # rely on PhotoImage's subsample (which only does integer ratios).
        with fitz.open(stream=page_png, filetype="png") as doc:
            matrix = fitz.Matrix(self.scale, self.scale)
            pix = doc[0].get_pixmap(matrix=matrix)
            return tk.PhotoImage(data=pix.tobytes("png"))

    def _clear_rects(self) -> None:
        if self.current_rect_id is not None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
        if self.existing_rect_id is not None:
            self.canvas.delete(self.existing_rect_id)
            self.existing_rect_id = None
        self.current_normalized = None

    def _load_current_target(self) -> None:
        if self.current_index >= len(self.targets):
            self._finish_and_close()
            return
        self._clear_rects()
        key = self.targets[self.current_index]
        label = f"{key[0]}/{key[1]}"
        progress = f"item {self.current_index + 1} of {len(self.targets)}"
        existing = self.existing.get(key)
        if existing is not None and existing.page == self.pdf_page_number:
            # Draw the existing box in a muted color so the operator can
            # see what's there and adjust rather than start from scratch.
            x0 = int(round(existing.x * self.display_width))
            y0 = int(round(existing.y * self.display_height))
            x1 = int(round((existing.x + existing.width) * self.display_width))
            y1 = int(round((existing.y + existing.height) * self.display_height))
            self.existing_rect_id = self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#888888",
                width=2,
                dash=(4, 4),
            )
            existing_msg = f" (existing: {existing.source})"
        elif existing is not None:
            existing_msg = f" (existing on page {existing.page}, not shown)"
        else:
            existing_msg = " (no existing box)"
        self.status_var.set(
            f"{progress}  ·  {label}  (page {self.pdf_page_number}){existing_msg}   "
            f"[drag to draw · enter=accept · s=skip · r=reset · q=quit]"
        )

    def _on_press(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)
        if self.current_rect_id is not None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
        self.current_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#ff6a3a",
            width=2,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.current_rect_id is None:
            return
        x0, y0 = self.drag_start
        x1 = max(0, min(self.display_width, event.x))
        y1 = max(0, min(self.display_height, event.y))
        self.canvas.coords(self.current_rect_id, x0, y0, x1, y1)

    def _on_release(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1 = max(0, min(self.display_width, event.x))
        y1 = max(0, min(self.display_height, event.y))
        self.drag_start = None
        # Normalize order so x0<x1 and y0<y1.
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        if self.current_rect_id is not None:
            self.canvas.coords(self.current_rect_id, left, top, right, bottom)
        if right - left < 3 or bottom - top < 3:
            # Treat a tap as "no box drawn" — ignore.
            if self.current_rect_id is not None:
                self.canvas.delete(self.current_rect_id)
                self.current_rect_id = None
            self.current_normalized = None
            return
        self.current_normalized = (
            left / self.display_width,
            top / self.display_height,
            (right - left) / self.display_width,
            (bottom - top) / self.display_height,
        )

    def _on_accept(self, _event=None) -> None:
        if self.current_normalized is None:
            # Nothing drawn — treat Enter as skip so operators can skim
            # forward through items that already have good boxes.
            self._on_skip()
            return
        key = self.targets[self.current_index]
        x, y, w, h = self.current_normalized
        self.result[key] = FocusRegion(
            page=self.pdf_page_number,
            x=x,
            y=y,
            width=w,
            height=h,
            source="operator_annotated",
        )
        self.current_index += 1
        self._load_current_target()

    def _on_skip(self, _event=None) -> None:
        self.current_index += 1
        self._load_current_target()

    def _on_reset_current(self, _event=None) -> None:
        if self.current_rect_id is not None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
        self.current_normalized = None

    def _on_quit(self, _event=None) -> None:
        self._finish_and_close()

    def _finish_and_close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> dict[tuple[str, str], FocusRegion]:
        self.root.mainloop()
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive annotator for eval focus regions"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the exam PDF file to annotate",
    )
    parser.add_argument(
        "--page",
        required=True,
        type=int,
        help="1-indexed page number within the PDF to annotate",
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated list of exam_id/question_id targets to prompt for",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Focus-regions YAML file to read from and write back to. "
            f"Default: {DEFAULT_FOCUS_REGIONS_PATH}"
        ),
    )
    args = parser.parse_args()

    config_path = args.config or DEFAULT_FOCUS_REGIONS_PATH
    targets = _parse_targets(args.targets)
    page_png, width_px, height_px = _rasterize_page(args.pdf, args.page)
    existing = load_focus_regions(config_path)

    print(
        f"annotating {len(targets)} target(s) on page {args.page} of "
        f"{args.pdf.name}",
        file=sys.stderr,
    )
    print(f"config: {config_path}", file=sys.stderr)

    app = AnnotationApp(
        page_png=page_png,
        page_width_px=width_px,
        page_height_px=height_px,
        pdf_page_number=args.page,
        targets=targets,
        existing=existing,
    )
    updated = app.run()

    # Merge result back into the on-disk config. Non-targeted entries
    # are preserved unchanged (the app never touches keys outside
    # `targets`, but we reload to be defensive against concurrent
    # edits).
    current = load_focus_regions(config_path)
    for key in targets:
        if key in updated:
            current[key] = updated[key]
        # If the operator skipped a target and it wasn't in existing,
        # it stays absent. If it was in existing, it stays as-is.
    save_focus_regions(
        config_path,
        current,
        header_comment=(
            "Canonical focus-region annotations for the eval harness.\n"
            "\n"
            "Read by scripts/smoke_vlm.py (default path; overridable with\n"
            "--focus-regions PATH for concurrent sessions).\n"
            "Written by scripts/annotate_focus_regions.py.\n"
            "\n"
            "Keys: 'exam_id/question_id' flat strings. Values mirror FocusRegion."
        ),
    )
    print(f"wrote updated focus regions to {config_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
