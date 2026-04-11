# First Redelivery Evaluation

Date: 2026-04-10

This note records what actually happened when the first model-edit pathology
redelivery was inspected against the current MC/OpenCV pipeline.

## What Arrived

The loose `edits_from_chatgpt/` directory did not contain the per-variant PNGs
named in the original manifest. It still held contact sheets plus manifests.

A partial redelivery zip then supplied two real full-resolution edited pages:

- `prompt_05_main_fill_plus_tiny_stray_v1.png`
- `prompt_12_ugly_but_intended_v1.png`

Those lived in:

- a local redelivery archive under the operator's `auto-grader-assets/edits_from_chatgpt/`
  drop, specifically `model_edit_pathology_packet_outputs_redelivery.zip` on the
  machine where this evaluation was run

## Contact Sheets Were Not Evaluable

The contact sheets were split into tiles for a quick probe, but that turned out
to be an invalid evaluation surface:

- every split tile failed QR identity
- the failure was `ValueError: No page-identity QR code detected`

That is a packaging/substrate problem, not a useful bubble-readback result.

Conclusion:

- contact-sheet outputs are not acceptable for pipeline evaluation
- we need full-resolution per-variant pages, not collages

## Real Full-Resolution Edited Pages

Two full-resolution pages *were* usable:

### `prompt_05_main_fill_plus_tiny_stray_v1.png`

Observed behavior:

- QR identity survived
- page matching succeeded
- the page produced one real marked bubble
- the scorer returned `incorrect`, not review

Interpretation:

- this does not currently look like a CV robustness failure
- it looks more like the edited page itself contains a substantive wrong-bubble
  mark rather than “main fill plus tiny ignorable stray”

### `prompt_12_ugly_but_intended_v1.png`

Observed behavior:

- QR identity survived
- page matching succeeded
- the page reduced to one real mark on question 4
- the rest of the page stayed blank
- that marked question scored as a single `incorrect` answer

Interpretation:

- this is encouraging
- the page behaved like a real “ugly but clearly intended” answer attempt
- the pipeline did not melt down into page-wide ambiguity

## Net Takeaway

The important result is not “the redelivery passed” or “the redelivery failed.”
It is more specific:

- the contact-sheet format is not an honest evaluation surface
- the current pipeline *can* survive full-resolution model-edited pages
- at least on the two real pages received so far, the outputs were interpretable
  and sensible

## What Is Still Missing

The most valuable full-resolution probes are still absent:

- `prompt_06_main_fill_plus_weak_secondary_v*.png`
- `prompt_08_changed_answer_erasure_residue_v*.png`

Those are the best remaining probes for:

- dominance arbitration
- changed-answer residue
- the real-world boundary between ignorable junk and substantial secondary marks

## Local Evaluation Artifact

The machine-readable evaluation used for this note is local at:

- `/tmp/chatgpt-pathology-redelivery-eval.json` on the machine where this
  evaluation was run
