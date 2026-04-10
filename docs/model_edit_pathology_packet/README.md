# Model-Edited Pathology Packet

This packet is for stress-testing the MC/OpenCV lane with model-edited marks that
are more realistic than our procedural probes, while still preserving the real
paper substrate and page geometry.

The goal is not to ask an image model to invent a whole exam page. The goal is
to start from our real rendered pages and ask for small, local mark edits that
simulate plausible student behavior.

## Attachment Sets

Use these exact images when submitting prompts.

### Set A: Single-question morphology probes

Use these when you want tight control over one bubble cluster and do not care
about the rest of the page looking busy.

- [single_question_clean.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/single_question_clean.png)
- [single_question_target_crop.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/single_question_target_crop.png)

### Set B: Full-page local-edit probes

Use these when you want a realistic exam-page context but still want the model
to edit only one local question block.

- [full_page_clean.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/full_page_clean.png)
- [full_page_question_block_crop.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/full_page_question_block_crop.png)

### Human reference only

These are not the first images I would submit to the model. They are here so we
can compare what the model makes against our current decision boundary.

- [boundary_ignored_short_slash.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/boundary_ignored_short_slash.png)
- [boundary_nonignored_compact_scribble.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/boundary_nonignored_compact_scribble.png)
- [dominance_reference_rendered.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/dominance_reference_rendered.png)
- [dominance_reference_normalized.png](/private/tmp/auto-grader-mc-opencv-prereq-0409/docs/model_edit_pathology_packet/assets/dominance_reference_normalized.png)

## Submission Guidance

Use image editing, not fresh image generation.

Ask for 4 variants per prompt.

Do not ask the model to change:

- QR codes
- registration markers
- printed text
- page perspective
- lighting
- paper edges
- bubble outlines
- page layout

Do ask it to change only local graphite or pencil-like marks inside or just
around the target bubbles.

If the tool supports it, add this short steering note before each prompt:

```text
Keep the page geometry, printed text, QR codes, registration markers, bubble outlines, and overall scan appearance unchanged. Only edit the local pencil/graphite marks near the target bubble area.
```

## Prompt Pack

### 1. Tiny incidental dust-like mark

Attachments:

- Set A

```text
Edit this existing answer-sheet image. Keep the page geometry, printed text, QR codes, registration markers, bubble outlines, and overall scan appearance unchanged.

Only change one target bubble area. Add a very small incidental graphite speck or dust-like pencil dot inside one wrong bubble. The mark should look accidental, tiny, and not at all like an attempt to fill the bubble. It should be plausible on a real scanned paper exam. Do not add any other marks anywhere else.

Return 4 variants with slightly different tiny accidental dot placement and darkness.
```

### 2. Small interior check tick that should still be ignorable

Attachments:

- Set A

```text
Edit this existing answer-sheet image. Preserve page geometry and all printed content exactly.

Inside one wrong bubble, add a small pencil check-like tick that lightly brushes the interior. It should look like an incidental pencil flick or a casual hand movement, not like a deliberate filled answer. Keep it sparse, thin, and local. Do not make it dark enough or dense enough to look like a true fill attempt.

Return 4 variants with slightly different check-tick shape, angle, and placement.
```

### 3. Short slash-through style mark

Attachments:

- Set A

```text
Edit this existing answer-sheet image while keeping the page and scan intact.

Inside one wrong bubble, add a short diagonal pencil slash that crosses part of the bubble interior. It should resemble a student making a quick line mark that they might mistakenly think counts as an answer, but it should still be much sparser than a filled bubble. Do not darken or broaden it into a true scribble fill.

Return 4 variants with slightly different slash length, angle, and darkness.
```

### 4. Dense compact scribble that really does look like a fill attempt

Attachments:

- Set A

```text
Edit this existing answer-sheet image while preserving the full page exactly.

Inside one wrong bubble, add a compact dark pencil scribble that clearly looks like a deliberate attempt to fill the bubble, even if it is messy. Keep the scribble mostly contained inside the bubble. It should look like a real student answer attempt, not a tiny accidental mark and not a line-through.

Return 4 variants with different messy-but-deliberate compact scribble patterns.
```

### 5. Strong correct fill plus tiny stray on a wrong bubble

Attachments:

- Set A

```text
Edit this existing answer-sheet image. Keep the whole page, scan, and printed substrate unchanged.

Make one bubble look clearly and deliberately filled with a dark compact pencil mark. In a different wrong bubble, add only a tiny accidental graphite speck or very faint incidental dot. The overall result should read visually as one obvious intended answer plus one tiny accidental secondary mark.

Return 4 variants with different positions and intensity for the tiny accidental secondary mark.
```

### 6. Strong correct fill plus weak secondary trace

Attachments:

- Set A

```text
Edit this existing answer-sheet image while preserving all page geometry and printing.

Make one bubble clearly filled with a dark compact intentional pencil mark. In another wrong bubble, add a weaker secondary mark that is more noticeable than a tiny dot but still visibly less substantial than the main filled bubble. It should look like a stray check, light partial fill, or weak incidental trace, not a second equally serious answer.

Return 4 variants with different weak-secondary shapes and darkness, but always keep the main answer obviously stronger.
```

### 7. Genuine double-answer case

Attachments:

- Set A

```text
Edit this existing answer-sheet image while keeping the page substrate unchanged.

Create a real double-mark scenario: make two different bubbles both look like substantial deliberate student fill attempts. They do not need to be identical, but both should plausibly be interpreted by a human as real answers rather than incidental traces.

Return 4 variants with slightly different relative strength between the two filled bubbles, but both must still look like genuine answer attempts.
```

### 8. Partial erasure residue after a changed answer

Attachments:

- Set A

```text
Edit this existing answer-sheet image. Preserve the page and scan exactly.

Create a changed-answer scenario. One bubble should look like the current intended filled answer. A different wrong bubble should show faint erased graphite residue, partial smudging, or rubbed-out remnants from an earlier answer. The erased bubble should look plausibly left behind by real pencil erasure on scanned paper, not like a clean fresh fill.

Return 4 variants with different amounts of erasure residue and smudging.
```

### 9. Full-page localized realistic mark edit

Attachments:

- Set B

```text
Edit this existing full answer-sheet page. Keep the page layout, text, QR codes, registration markers, bubble outlines, and overall scan geometry unchanged.

Only modify one question block. Add a realistic student mark pathology near the bubbles for that one question: either a tiny incidental dot, a light check tick, a short slash, or a partial erasure residue. Keep everything else on the page untouched. The result should still look like a normal scanned exam page, not a synthetic redraw.

Return 4 variants, one for each of those four pathology types.
```

### 10. Full-page dominance case

Attachments:

- Set B

```text
Edit this existing full answer-sheet page. Preserve all page geometry and printed elements exactly.

In one question block, make one bubble clearly look like the intended filled answer and add a weaker secondary mark on another bubble in the same question. The weaker mark should be plausibly accidental or at least visibly less substantial than the main one. The result should look like a case where a human would usually say there is one real answer and one weaker stray survivor.

Return 4 variants with different secondary-mark styles: tiny dot, check tick, partial erasure residue, and weak partial fill.
```

### 11. Adjacent handwriting intrusion

Attachments:

- Set B

```text
Edit this existing full answer-sheet page while preserving the printed page exactly.

Add a small amount of realistic stray handwritten graphite near one question, such as a short handwritten note fragment, number, or margin-like pencil motion that lightly intrudes toward a bubble area without being a true bubble fill. Keep it plausible for a real student working quickly on paper.

Return 4 variants with different nearby handwritten intrusion shapes and distances from the bubble.
```

### 12. Ugly but still clearly intended answer

Attachments:

- Set B

```text
Edit this existing full answer-sheet page while keeping the scan substrate unchanged.

In one question block, create an ugly but still obviously intended answer mark. It should be messy, uneven, and student-like, but still look to a human like a real attempt to fill one bubble rather than an incidental trace. Do not add a second substantial mark. The point is to create realistic sloppy fills, not adversarial ambiguity.

Return 4 variants with different messy fill styles.
```

## Recommended First Batch

If you only do one round first, use these six prompts:

1. Prompt 2: small interior check tick
2. Prompt 3: short slash-through style mark
3. Prompt 5: strong correct fill plus tiny stray
4. Prompt 6: strong correct fill plus weak secondary trace
5. Prompt 8: partial erasure residue
6. Prompt 12: ugly but clearly intended answer

That set is the best first reality check because it probes the exact classroom
boundary we care about:

- ignorable incidental junk
- weak secondary survivors next to a real answer
- changed-answer residue
- ugly but still gradable intentional fills
