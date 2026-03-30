# Professor Backlog

Items that need professor input, verification, or decision before the system
can act on them. Each item is tagged with priority and what's blocked on it.

## Pending

### Verify transcribed template constants
- **Priority**: Before generation pipeline produces real exams
- **What**: The CHM 141 Fall 2023 exam has been transcribed into
  `templates/chm141-final-fall2023.yaml`. Numeric constants (molar masses,
  Planck's constant, R values, heat of vaporization, specific heat
  capacities) were taken from the source docx but should be spot-checked
  against the professor's reference materials.
- **Blocked**: Generation pipeline can be built and tested against these
  values, but real exam production should wait for verification.

### mc-5: isotope notation question
- **Priority**: Low — does not block any pipeline work
- **What**: Multiple-choice question 5 from the Fall 2023 final uses isotope
  notation symbols (superscript mass number, subscript atomic number) that
  are rendered as images, not representable in plain text. The question is
  omitted from the current template. To include it, we need:
  1. The professor to provide or approve a figure image for the isotope
     notation choices.
  2. The figure to be placed in the template's figures directory.
  3. The question added to the YAML with a `figure:` reference.
- **Blocked**: Nothing — this is a completeness gap, not a functional one.
  The template validates and works without it.

### Structured rubrics for remaining manual-review questions
- **Priority**: Before LLM-assisted grading (post-v1)
- **What**: The template includes structured rubrics
  (`[{criterion, points}]`) for all manual-review questions, but these were
  derived from the answer key, not from the professor's actual grading
  rubric. The professor likely has more detailed rubrics — or can refine
  these — that would make LLM grading instructions more precise.
- **Blocked**: Nothing in v1. These rubrics work for human review now. But
  higher-quality rubrics will directly improve LLM grading accuracy later.

### Additional exam templates
- **Priority**: After generation pipeline is working
- **What**: The system currently has one exam template (Fall 2023 final).
  For the skeletonizer and LLM inferencer helpers (build order steps 7-8),
  we'll want multiple historical versions of the same exam with different
  numbers. The professor's archive of past exams would feed both template
  authoring and the inference pipeline.
- **Blocked**: Nothing immediate. One template is sufficient to build and
  test the full pipeline.

### Completed exam scan corpus (OCR dataset)
- **Priority**: When OCR model evaluation begins
- **What**: The professor has an estimated 10-100 additional completed exam
  scans from the same institutional scanner, same course, same paper format.
  Currently 4 are in the assets repo. The full corpus is a significant
  affordance for OCR work:
  1. **Validation set**: Evaluate OCR model accuracy against real handwriting
     diversity (neat vs. messy, chemical notation, equations, diagrams).
  2. **Fine-tuning dataset**: If a base OCR model underperforms on chemistry
     handwriting, a corpus this size is plausibly sufficient for fine-tuning
     or LoRA adaptation of a small model to the specific handwriting +
     scanning distribution.
  3. **Confidence calibration**: Establish the relationship between model
     confidence scores and actual accuracy on this specific input
     distribution.
- **Action**: When OCR evaluation begins, request the full scan corpus from
  the professor. Ingest into `auto-grader-assets/scans/`.
- **Blocked**: Nothing — the 4 scans already available are enough to start
  model evaluation. The full corpus unlocks fine-tuning and robust
  calibration.
