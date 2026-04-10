# Eval Run Interpretation

This note is for reading prompt-tuning and smoke-eval runs without
re-deriving the current evaluation stance from thread context.

## Run Sets

`--tricky`
- The original 6-item regression sentinel.
- Use it to catch obvious regressions quickly.
- Do not treat it as a representative corpus; it overweights hostile items.

`--tricky-plus`
- The 12-item expanded set used for current prompt tuning.
- Includes the original hostile probes plus clean-correct positives and
  partial-credit calibration items.
- This is the main prompt-comparison surface right now.

## Truth vs History

`professor_score`
- The historical mark written on the page.
- Preserve it for traceability.

`corrected_score`
- Human-investigated correction when the historical mark is believed to be
  wrong.

`truth_score`
- The score the harness should compare against.
- Falls back to `professor_score` when no correction exists.

Important consequence:
- A run can be correct while disagreeing with the historical professor mark.
- Example: `15-blue/fr-5b` and `34-blue/fr-8` are correction-sensitive items.

## Obvious Buckets

`is_obviously_fully_correct`
- High-trust auto-pass bucket.
- Use only when the answer is clearly correct and does not need human rescue.

`is_obviously_wrong`
- High-trust auto-zero bucket.
- Use only when the answer is clearly wrong and no lawful partial-credit path
  remains.

Do not read these as general confidence scores. They are triage buckets.

## Current Optimization Posture

As of 2026-04-10:
- Qwen is the primary prompt-tuning surface.
- Gemma is still useful as a comparison model, but it is slower and noisier.
- Broad prompt surgery should stop unless the run family genuinely shifts.

Recent Qwen `--tricky-plus` family:
- Stable good band: roughly `9-10/12` exact, `11-12/12` within one point.
- The current work is calibration in the partial-credit middle, not obvious
  full-credit or obvious-zero behavior.

## Stable Misses

The recurring `--tricky-plus` misses are now narrow:
- `34-blue/fr-12a`
  Lewis partial-credit rescue remains too harsh.
- `15-blue/fr-10a`
  setup-credit rescue remains too harsh.

These are the main items that still justify targeted prompt calibration.

The following items have been wobble/noise rather than stable failures:
- `15-blue/fr-1`
- `15-blue/fr-3`

Do not broaden the prompt on the basis of those items alone.

## Narrator Reading

The live narrator is for legibility, not authority.

Use it for:
- spotting reasoning loops
- seeing whether the model is stuck, converging, or hallucinating
- understanding what kind of human review might be needed

Do not use it as ground truth for:
- chemistry correctness
- corrected-truth adjudication
- final evaluation metrics

The durable metrics surface is still `predictions.jsonl` scored against
`truth_score`.

## History Direction

Project Paint Dry is moving toward:
- fast live/status rails for motion
- checkpoint-shaped durable history for legibility

Raw thought persistence is still present for comparison, but the direction is
to make durable history increasingly checkpoint-based once that path proves
itself in real runs.
