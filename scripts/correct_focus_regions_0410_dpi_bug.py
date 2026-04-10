"""One-shot recovery script for the 2026-04-10 annotator DPI bug.

Background
----------
The original `scripts/annotate_focus_regions.py` used a broken display
image pipeline:

1. Rasterized the PDF page at DPI=160, producing e.g. a 1357x1753 PNG.
2. Declared a canvas of size (display_w, display_h) = round(source × scale)
   where scale = min(1400/source_w, 1600/source_h, 1.0).
   For 15-blue pages: canvas = 1239 x 1600.
3. "Rescaled" the PNG for display via
   `fitz.open(stream=png, filetype='png').get_pixmap(matrix=scale)`.
   This path reinterprets the PNG as a POINT-based document rather than
   a pixel image, so the returned pixmap is ~45% of the intended canvas
   size — for 15-blue, 558x720 instead of 1239x1600.
4. Painted that small PNG at (0, 0) on the canvas via
   `create_image(0, 0, anchor=tk.NW)`. The canvas was the declared size,
   but the image only covered the top-left 558x720 of it.
5. Normalized operator clicks against the CANVAS dimensions (1239x1600)
   instead of the IMAGE dimensions (558x720). Every stored
   `FocusRegion` has normalized coordinates compressed toward the
   top-left corner of the page by the ratio image/canvas ≈ 0.45.

Reconstruction
--------------
For each stored operator_annotated entry, recover the true normalized
coordinates by multiplying each axis by (canvas_dim / image_dim) and
clamping to [0, 1]. This inverts the stale normalization exactly:

    true_x      = min(1, stored_x      * canvas_w / image_w)
    true_y      = min(1, stored_y      * canvas_h / image_h)
    true_x_end  = min(1, (stored_x + stored_width)  * canvas_w / image_w)
    true_y_end  = min(1, (stored_y + stored_height) * canvas_h / image_h)
    true_width  = true_x_end - true_x
    true_height = true_y_end - true_y

Case analysis:
- Stored box fully inside legal terrain
  (stored_x + stored_width ≤ image_w/canvas_w): correction is a clean
  scalar multiplication. The operator clicked on real image pixels
  and the corrected normalized coord is exactly where they clicked in
  page coordinates.
- Stored box extends past the legal terrain boundary: the operator
  dragged past the visible image into empty canvas. The only thing
  past the image on that axis is the edge of the page itself, so
  clamping the multiplied coord to 1.0 faithfully captures operator
  intent ("include everything up to the page edge on this axis").
- There is no information-theoretic gap. Every stored box fully
  determines its corrected coordinates given the canvas/image
  dimensions.

Constants (15-blue source)
--------------------------
- PDF page rasterized at 160 DPI → 1357x1753 px
- `_compute_display_scale(1357, 1753)` = 0.913
- Declared canvas = (1239, 1600)
- Actual rendered image from the buggy path = (558, 720)
- Correction factor x = 1239 / 558 ≈ 2.2204
- Correction factor y = 1600 / 720 ≈ 2.2222

Scope
-----
This script ONLY corrects entries in `eval/focus_regions.yaml` whose
provenance is `operator_annotated`. Mock entries from the old
_TRICKY_FOCUS_REGION_MOCKS migration are left alone — they were never
affected by this bug. Only the annotated 15-blue boxes from the
2026-04-10 session need correction.

This is a ONE-SHOT recovery tool. After running it once and fixing
the annotator, this bug cannot recur, and the script should not be
run again. It is committed for provenance so the git history records
exactly how the data was recovered.
"""

from __future__ import annotations

from pathlib import Path

from auto_grader.eval_harness import FocusRegion
from auto_grader.focus_regions import load_focus_regions, save_focus_regions

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "eval" / "focus_regions.yaml"
)

# Per-exam correction factors. Derived from the specific (source-DPI,
# max-display, actual-rendered) combination that existed at the time
# of the 2026-04-10 annotation session. 15-blue is the only exam that
# was actually annotated in that session; other exams still carry
# mock provenance and are not touched by this script.
_CORRECTION_FACTORS: dict[str, tuple[float, float]] = {
    "15-blue": (1239.0 / 558.0, 1600.0 / 720.0),
}


def _corrected_region(region: FocusRegion, fx: float, fy: float) -> FocusRegion:
    true_x = min(1.0, region.x * fx)
    true_y = min(1.0, region.y * fy)
    true_x_end = min(1.0, (region.x + region.width) * fx)
    true_y_end = min(1.0, (region.y + region.height) * fy)
    return FocusRegion(
        page=region.page,
        x=true_x,
        y=true_y,
        width=max(0.0, true_x_end - true_x),
        height=max(0.0, true_y_end - true_y),
        source=region.source,
    )


def main() -> None:
    regions = load_focus_regions(_DEFAULT_CONFIG)
    updated: dict[tuple[str, str], FocusRegion] = {}
    corrections = 0
    skipped_not_operator = 0
    skipped_no_factor = 0

    for key, region in regions.items():
        exam_id = key[0]
        if region.source != "operator_annotated":
            updated[key] = region
            skipped_not_operator += 1
            continue
        if exam_id not in _CORRECTION_FACTORS:
            updated[key] = region
            skipped_no_factor += 1
            continue
        fx, fy = _CORRECTION_FACTORS[exam_id]
        corrected = _corrected_region(region, fx, fy)
        updated[key] = corrected
        corrections += 1
        print(
            f"{exam_id}/{key[1]}: "
            f"x={region.x:.4f}→{corrected.x:.4f} "
            f"y={region.y:.4f}→{corrected.y:.4f} "
            f"w={region.width:.4f}→{corrected.width:.4f} "
            f"h={region.height:.4f}→{corrected.height:.4f}"
        )

    save_focus_regions(_DEFAULT_CONFIG, updated)
    print()
    print(f"corrections applied:          {corrections}")
    print(f"skipped (not operator):       {skipped_not_operator}")
    print(f"skipped (no factor for exam): {skipped_no_factor}")
    print(f"wrote: {_DEFAULT_CONFIG}")


if __name__ == "__main__":
    main()
