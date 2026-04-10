# Integration Protocol

Use this when the human asks you to:
- manage the integration
- cut or refresh an integration branch
- pull everything in
- make the smoke surface current

This doc exists so "integration" is a concrete workflow with receipts, not a
memory test.

## Goal

Build or refresh a union smoke surface that is honestly current with respect to
the sibling lanes the human intends to integrate, while preserving the
distinction between:
- committed branch-tip deltas
- dirty local scratch
- integration-local commits that exist only on the union branch

## Terms

`source lanes`
- The sibling implementation lanes feeding the integration branch.
- In the current auto-grader shape, these are usually:
  - `Project Solipsism`
  - `Liquid Varnish Squadron`
  - `Gauge Saints`
  - `focus frame`

`integration lane`
- The union smoke surface the human is actually going to run.

`current`
- Reserved word.
- You may only say the integration lane is current after you perform the full
  parity ritual below and emit the receipt.

## Hard rules

- Do not infer "current" from remembered SHAs.
- Do not infer "current" from `origin/*` alone.
- Do not infer "current" from one sibling lane at a time.
- Do not collapse committed deltas and dirty local scratch into one notion of
  "ahead".
- Do not silently ignore integration-local commits just because they do not
  exist on any sibling branch.

## Integration ritual

Before you claim an integration lane is current, do all of these:

1. Inspect the local sibling worktrees.
   Record the current local tip and worktree status for each source lane.

2. Separate committed deltas from dirty local scratch.
   For each source lane, report both:
   - `committed missing deltas`
   - `dirty scratch left alone`

3. Compare the integration branch against the sibling branch tips.
   Use content-aware comparison, not ancestry alone.
   `git cherry` is useful, but if it still shows commits after parity work you
   must say whether they are:
   - conflict-resolved equivalents
   - truly missing slices
   - or integration-local commits that must be retained

4. Inspect the integration branch's own post-pull commits.
   Some smoked-surface fixes may live only on the integration branch. Do not
   lose them by trying to reconstruct the branch purely from sibling SHAs.

5. Verify by a fresh relevant test or smoke.
   Do not stop at git shape. Run the smallest contracts that materially cover
   the slice you just pulled.

## Response receipt

Whenever you say the integration lane is current, include this exact receipt
shape in your response:

- `Checked lanes:` list the sibling lanes you actually inspected
- `Committed missing deltas:` `none` or an explicit list
- `Dirty scratch left alone:` `none` or an explicit list
- `Integration-local commits retained:` `none` or an explicit list
- `Integration tip:` branch + SHA
- `Verification:` commands run and pass/fail

If any field is uncertain, say so explicitly instead of compressing it away.

## Choosing what to pull

If the human says "pull everything in", default to:
- all committed sibling-lane deltas relevant to the requested integration scope
- while leaving dirty local scratch alone unless the human explicitly asks for it

If the human means exact parity with sibling tips, say that explicitly in the
receipt.

If the lane has an authoritative scope restriction, honor it. Example:
- visual smoke surface
- semantic base + selected polish
- focus-preview runtime only

Do not silently widen or narrow the intended integration scope.

## Conflict handling

When cherry-picks conflict:
- resolve minimally
- preserve integration-local commits unless they are clearly superseded
- rerun the smallest relevant contracts immediately after resolving
- explain in the receipt when a sibling commit is present only as a
  conflict-resolved equivalent

## Stale state

If Epistaxis has duplicate or stale entries for the integration lane:
- reconcile or remove them on contact before doing new integration work

Stale duplicate topo state is a coordination bug, not harmless clutter.

## Current auto-grader note

For this repo, the integration lane is often a smoke branch that merges:
- the semantic source lane
- the visual polish lanes
- the focus-preview lane

That means the answer to "is everything in?" often depends on both:
- sibling branch parity
- and whether the union branch itself carries smoked-surface fixes that do not
  live anywhere else

Treat both as first-class.
