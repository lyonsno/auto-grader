# Focus Preview

This note records the current contract and rendering constraints for the
Project Paint Dry focus-preview surface. It is intentionally internal and
pragmatic: the goal is to preserve the lessons from the prototype lane without
pretending the detector path is already finished.

## FocusRegion contract

`FocusRegion` is a display seam, not a detector result.

- It is the best available crop hint for the item currently being graded.
- It may come from item-level ground truth metadata, template question/part
  metadata, or a temporary mock map used to prototype the display path.
- `resolve_focus_region(...)` prefers item metadata and otherwise falls back to
  template metadata by `question_id`.
- The `source` field is the provenance hook. Consumers should use it to explain
  where a box came from, not to infer detector quality.

What this seam is not:

- It is not an OCR claim.
- It is not proof that the crop was discovered automatically.
- It is not yet a stable public file format beyond the current normalized box
  shape (`page`, `x`, `y`, `width`, `height`, `source`).

## Rendering constraints

The focus preview sits under `status + live` and above `history` as a companion
surface. The renderer is intentionally conservative because this panel shares a
high-FPS terminal UI with the rest of Paint Dry.

Current constraints:

- Decode and rasterize the preview once per item. Do not rebuild the PNG or the
  full preview field on every animation tick.
- Keep preview aspect honest. If the crop should be tighter, tighten the crop;
  do not distort it into a friendlier terminal rectangle.
- Budget preview size from both terminal width and source detail. Small crops
  and large detailed crops should not receive the same raster budget.
- Keep the previous preview visible while the next item is loading. The panel
  should not collapse and reappear between items.
- Pending transitions may use a faint overlay field, but the steady-state image
  should be background-first. If the image starts reading like a full ASCII
  texture, the renderer has drifted too far toward glyph art.
- Compress harsh paper whites for dark-theme harmony, but preserve thin dark
  strokes during downsampling so handwritten structure survives.

## Current prototype status

The focus preview is real enough to smoke:

- `smoke_vlm.py` emits preview events.
- `NarratorSink` carries those events through the live Paint Dry channel.
- `narrator_reader.py` renders the preview panel in the live UI.
- The full `TRICKY+` set currently has mock fallback boxes so the path can be
  exercised before a detector exists.

What is still provisional:

- The boxes are still mock coordinates unless real metadata is present.
- There is no OCR- or OpenCV-based detector yet.
- Visual tuning is still smoke-driven, so the renderer should be treated as a
  settling internal surface rather than a finished presentation layer.
