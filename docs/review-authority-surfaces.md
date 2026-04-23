# Review Authority Surfaces

This document extends Topothesia beyond public-doc routing and into review
authority for `auto-grader`.

When the human says `Make this durable for review`, choose the narrowest durable
control that fits the situation:

- Update `docs/review_surfaces.toml` and this document when the issue is about
  authority, canonical-vs-fallback interpretation, allowed divergence, or other
  review-routing semantics.
- Update the repo-local Prilosec registry when the issue is a recurring
  acknowledged false positive or accepted finding family that future reviews
  should suppress or demote.
- Update only the review artifact when the issue is specific to one reviewed
  commit and does not need a standing repo-level rule.

Topothesia review surfaces are for authority and routing, not generic
suppression.

## Clean Eval Scan Authority

For grading, narration, smoke, and probe flows, `auto_grader/vlm_inference.py`
is the authoritative implementation surface for exam-id to scan-path routing,
including the contamination boundary enforced by `resolve_scan_pdf_path()`.

Current authoritative clean-scan examples:

- `15-blue` must resolve to `15 blue_professor_markings_hidden.pdf`
- `39-blue-redacted` resolves to
  `39 blue_Redacted_grading_marks_removed.pdf`

The legacy professor-marked filename `15 blue.pdf` is not a compatibility
fallback in the current mode. The authoritative behavior is to fail loudly with
`Refusing contaminated fallback`, not to reopen that file silently.

Consumer surfaces such as `scripts/smoke_vlm.py`, `scripts/smoke_probe.py`, and
`scripts/narrator_reader.py` should inherit this boundary instead of inventing
their own fallback semantics.

Default review posture: treat a "missing compatibility fallback to
`15 blue.pdf`" finding as intentionally waived by design, not as a must-fix
bug. The current grading, narration, and eval mode is only semantically
coherent on clean scans; if we ever need to inspect professor-marked scans,
that must be a separate explicit mode with separate explicit mode semantics.

Promote the finding to material only if one of these becomes true:

- a surface silently accepts a contaminated alias instead of failing loudly
- consumer surfaces contradict the authoritative clean-scan declaration
- the project introduces a professor-marked mode without seating separate
  review-routing semantics for it

Revisit this authority declaration when clean-scan preparation becomes
automatic, or when professor-marked inspection becomes a real first-class
workflow instead of an intentionally unsupported fallback.
