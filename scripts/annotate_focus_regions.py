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
    d               ditto: copy the immediately previous target's in-session
                    accepted box, mark current accepted, advance. Useful for
                    multi-part questions (fr-7a/b/c/d/e) that share one
                    physical answer region on the page. Strict semantics:
                    skipping a target breaks the ditto chain.
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

from auto_grader.eval_harness import FocusRegion, load_ground_truth  # noqa: E402
from auto_grader.focus_regions import (  # noqa: E402
    DEFAULT_FOCUS_REGIONS_PATH,
    load_focus_regions,
    save_focus_regions,
)

_GROUND_TRUTH_PATH = _REPO_ROOT / "eval" / "ground_truth.yaml"


_MAX_DISPLAY_WIDTH = 1400
_MAX_DISPLAY_HEIGHT = 1600


def _ditto_box_for(
    current_index: int,
    session_accepted: dict,
):
    """Return the box to copy when the operator hits ditto at the
    current target, or None if no previous operator-accepted box is
    available.

    Strict semantics: copies the IMMEDIATELY previous target's
    in-session accepted box (index == current - 1) or nothing. Does
    not walk backward past gaps — skipping a target breaks the ditto
    chain, which is the right behavior because otherwise ditto would
    silently copy coordinates from an unrelated earlier target.
    """
    if current_index <= 0:
        return None
    return session_accepted.get(current_index - 1)


def _load_exam_question_sequence(exam_id: str) -> list[str]:
    """Return the canonical question_id sequence for an exam from the
    ground-truth YAML. Order is preserved as it appears in the file."""
    gt = load_ground_truth(_GROUND_TRUTH_PATH)
    return [item.question_id for item in gt if item.exam_id == exam_id]


def _parse_targets(raw: str) -> list[tuple[str, str]]:
    """Parse a --targets string into an ordered list of (exam_id, question_id)
    tuples. Accepts explicit entries and dotdot ranges:

        15-blue/fr-1                single entry
        15-blue/fr-1..fr-3          range, expanded using ground-truth order
        15-blue/fr-1,15-blue/fr-3   comma-separated, same exam or different
        15-blue/fr-1..fr-2,15-blue/fr-12a   mix of ranges and explicits

    Ranges expand against the canonical question_id sequence from the
    ground-truth YAML, so sub-parts (fr-5a, fr-5b, fr-5c) are included
    when a range straddles them. Both endpoints must exist in the
    exam's sequence; unknown endpoints are a SystemExit with a
    caller-helpful message.
    """
    out: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "/" not in token:
            raise SystemExit(
                f"--targets entry {token!r} is not in 'exam_id/question_id' form"
            )
        exam_id, rest = token.split("/", 1)
        exam_id = exam_id.strip()
        rest = rest.strip()
        if ".." in rest:
            first, last = rest.split("..", 1)
            first = first.strip()
            last = last.strip()
            sequence = _load_exam_question_sequence(exam_id)
            if not sequence:
                raise SystemExit(
                    f"--targets exam_id {exam_id!r} has no ground-truth entries"
                )
            if first not in sequence:
                raise SystemExit(
                    f"--targets range start {exam_id}/{first} not found in "
                    f"ground truth for exam {exam_id}"
                )
            if last not in sequence:
                raise SystemExit(
                    f"--targets range end {exam_id}/{last} not found in "
                    f"ground truth for exam {exam_id}"
                )
            start_idx = sequence.index(first)
            end_idx = sequence.index(last)
            if end_idx < start_idx:
                raise SystemExit(
                    f"--targets range {exam_id}/{first}..{last} is reversed "
                    f"(end comes before start in ground-truth order)"
                )
            for qid in sequence[start_idx : end_idx + 1]:
                out.append((exam_id, qid))
        else:
            out.append((exam_id, rest))
    if not out:
        raise SystemExit("--targets was empty")
    return out


def _compute_display_dpi(
    page_pt_width: float,
    page_pt_height: float,
    *,
    max_width: int = _MAX_DISPLAY_WIDTH,
    max_height: int = _MAX_DISPLAY_HEIGHT,
) -> float:
    """Return the DPI at which a PDF page of the given point dimensions
    rasterizes to the largest pixel size that still fits inside the
    max display bounds. Pure function of point dimensions and bounds.

    The returned DPI, when passed to `page.get_pixmap(dpi=...)`,
    produces a pixmap whose width and height are exactly the display
    dimensions the caller will use to create the canvas. No re-scale
    step is needed downstream.
    """
    # page_pt_* is in PDF points (72 per inch). pixel = point × dpi / 72.
    dpi_w = max_width * 72.0 / page_pt_width
    dpi_h = max_height * 72.0 / page_pt_height
    return min(dpi_w, dpi_h)


def _normalize_drag_box(
    drag_start: tuple[int, int],
    release: tuple[int, int],
    *,
    display_width: int,
    display_height: int,
    min_pixel_size: int = 3,
):
    """Pure click-to-normalized coordinate math.

    Takes two display-space points (in pixels), clamps them to the
    display bounds, normalizes their order, and returns a tuple
    ``(x, y, width, height)`` in normalized ``[0, 1]`` coordinates —
    *or* ``None`` if the drag is smaller than ``min_pixel_size`` in
    either dimension (treated as a tap and ignored).

    This function is the hard contract for what the annotator saves:
    every stored focus region was produced by exactly this math,
    against a ``(display_width, display_height)`` equal to the actual
    pixel dimensions of the rendered page image. If that invariant
    ever slips, the saved coordinates become garbage — which is why
    we factor this out as a pure function and test it directly.
    """
    start_x, start_y = drag_start
    release_x, release_y = release
    start_x = max(0, min(display_width, start_x))
    start_y = max(0, min(display_height, start_y))
    release_x = max(0, min(display_width, release_x))
    release_y = max(0, min(display_height, release_y))
    left, right = min(start_x, release_x), max(start_x, release_x)
    top, bottom = min(start_y, release_y), max(start_y, release_y)
    if right - left < min_pixel_size or bottom - top < min_pixel_size:
        return None
    return (
        left / display_width,
        top / display_height,
        (right - left) / display_width,
        (bottom - top) / display_height,
    )


def _rasterize_page_for_display(
    pdf_path: Path,
    page_number: int,
) -> tuple[bytes, int, int]:
    """Rasterize a PDF page (1-indexed) at the display DPI computed
    from the page's native point dimensions. Returns
    (png_bytes, width_px, height_px). The width and height are the
    *actual* pixel dimensions of the returned PNG — the canvas should
    be sized to these values exactly, with no further scaling.
    """
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise SystemExit(
                f"--page {page_number} out of range for {pdf_path} "
                f"(page count = {document.page_count})"
            )
        page = document[page_number - 1]
        rect = page.rect
        # fitz wants an integer DPI; floor instead of round to make
        # sure we do not exceed the max display bounds on either axis
        # due to rounding up.
        dpi = int(_compute_display_dpi(rect.width, rect.height))
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png"), pix.width, pix.height


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
        self.pdf_page_number = pdf_page_number
        self.targets = targets
        self.existing = dict(existing)  # copy; will be mutated on accept
        self.result: dict[tuple[str, str], FocusRegion] = dict(existing)
        # Session-scoped map of target index → box accepted in this
        # session. Distinct from `existing` (which is the loaded config)
        # and from `result` (which is the merged output). The ditto
        # affordance reads only from this map, never from `existing`,
        # so it cannot silently copy stale coordinates.
        self.session_accepted: dict[int, FocusRegion] = {}
        self.current_index = 0
        # The page image was already rasterized at the exact display
        # dimensions by `_rasterize_page_for_display` — no scale step
        # here, no re-rasterize. The canvas size IS the image size.
        # This is the contract that makes click → normalized coord
        # math correct: every stored coord is `click / display_dim`,
        # and `display_dim` equals the actual PNG pixel dimensions.
        self.display_width = page_width_px
        self.display_height = page_height_px

        self.root = tk.Tk()
        self.root.title("focus region annotator")
        # macOS foreground dance. Without this, second-and-subsequent
        # invocations from the same shell session can land behind the
        # current window and the operator thinks the tool is hanging.
        # Briefly pin topmost, lift, force focus, then clear topmost so
        # the window doesn't stay stuck above everything.
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
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

        # The page_png was rasterized at the exact display dimensions
        # upstream, so tk.PhotoImage takes it directly at native size.
        # No scaling required and no reinterpret-PNG-as-document path.
        self.tk_image = tk.PhotoImage(data=page_png)
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
        self.root.bind("<d>", self._on_ditto)
        self.root.bind("<q>", self._on_quit)
        self.root.bind("<Escape>", self._on_quit)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._load_current_target()

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
            f"[drag=draw · enter=accept · d=ditto prev · s=skip · r=reset · q=quit]"
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
        release = (event.x, event.y)
        drag_start = self.drag_start
        self.drag_start = None
        normalized = _normalize_drag_box(
            drag_start,
            release,
            display_width=self.display_width,
            display_height=self.display_height,
        )
        if normalized is None:
            if self.current_rect_id is not None:
                self.canvas.delete(self.current_rect_id)
                self.current_rect_id = None
            self.current_normalized = None
            return
        # Reflect the clamped, normalized coords back to the canvas
        # rectangle so the visible selection matches what was stored.
        x_norm, y_norm, w_norm, h_norm = normalized
        left = int(round(x_norm * self.display_width))
        top = int(round(y_norm * self.display_height))
        right = int(round((x_norm + w_norm) * self.display_width))
        bottom = int(round((y_norm + h_norm) * self.display_height))
        if self.current_rect_id is not None:
            self.canvas.coords(self.current_rect_id, left, top, right, bottom)
        self.current_normalized = normalized

    def _on_accept(self, _event=None) -> None:
        if self.current_normalized is None:
            # Nothing drawn — treat Enter as skip so operators can skim
            # forward through items that already have good boxes.
            self._on_skip()
            return
        key = self.targets[self.current_index]
        x, y, w, h = self.current_normalized
        region = FocusRegion(
            page=self.pdf_page_number,
            x=x,
            y=y,
            width=w,
            height=h,
            source="operator_annotated",
        )
        self.result[key] = region
        self.session_accepted[self.current_index] = region
        self.current_index += 1
        self._load_current_target()

    def _on_skip(self, _event=None) -> None:
        self.current_index += 1
        self._load_current_target()

    def _on_ditto(self, _event=None) -> None:
        previous = _ditto_box_for(self.current_index, self.session_accepted)
        if previous is None:
            # Nothing to copy. Don't advance, don't beep — just leave
            # the operator on the current target so they can draw.
            return
        key = self.targets[self.current_index]
        region = FocusRegion(
            page=self.pdf_page_number,
            x=previous.x,
            y=previous.y,
            width=previous.width,
            height=previous.height,
            source="operator_annotated",
        )
        self.result[key] = region
        self.session_accepted[self.current_index] = region
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
    page_png, width_px, height_px = _rasterize_page_for_display(args.pdf, args.page)
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
