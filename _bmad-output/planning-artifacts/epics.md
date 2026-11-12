# TERRAPYGE Epics and Stories

Derived from the GitHub issue tracker so the two systems mirror rather than
compete. Each story maps to one issue number.

## Epic 1: Thesis manuscript

Deliver the five-chapter manuscript, its figures, and the defense materials.

### Story 1.1: Revise Chapter 1 objectives, limitations and definitions

Align Chapter 1 with the delivered evaluation: Objective 5 must promise the
blocked cross-validation, and Section 1.5 must stop claiming that spatial
masking alone is the appropriate test. GitHub issue #13.

### Story 1.2: Add evaluation-design background to Chapter 2

Add a Technical Background subsection covering spatial autocorrelation, blocked
and leave-one-region-out cross-validation, transductive versus inductive
evaluation, metric choice under class imbalance, and bootstrap testing. GitHub
issue #14.

### Story 1.3: Write Chapter 5 conclusions and recommendations

Summarise findings, tie conclusions to the Chapter 1 objectives, and state the
contribution as robustness under information loss rather than peak accuracy.
GitHub issue #6.

### Story 1.4: Complete thesis front matter

Abstract, table of contents, lists of figures, tables and notations, and the
approval sheet. GitHub issue #7.

### Story 1.5: Finalise manuscript figures and tables

Produce and curate the figure set into docs/figures/, including the study area
map, slope units, graph schematic, masking curve, cross-validation comparison
and hazard map. GitHub issue #8.

### Story 1.6: Prepare defense presentation

Build the deck, prepare anticipated questions, and rehearse. GitHub issue #9.

## Epic 2: Methodological hardening

Close the methodological gaps surfaced by the Stage B4 analysis.

### Story 2.1: Spatial and block cross-validation

Add scripts/spatial_cv.py with random, blocked and inductive protocols over two
feature modes, plus the induced-subgraph helper. GitHub issue #1.

### Story 2.2: Calibrate the decision threshold for F1 reporting

Replace the fixed 0.5 threshold with a validation-selected, prevalence-matched
threshold and report precision and recall alongside F1. GitHub issue #11.

### Story 2.3: Investigate the XGBoost masking anomaly

Explain why baseline AUC rises as more topographic evidence is withheld, and
either dismiss it as a composition artifact or add a caveat. GitHub issue #10.

### Story 2.4: Extend Stage B4 to all backbones and a buffer sweep

Run all three GNN backbones and a buffer-distance sweep to test how sensitive
the 0.873 figure is to those choices. GitHub issue #5.

## Epic 3: Reproducibility and automation

Make the repository verifiable by someone other than its author.

### Story 3.1: Documentation sync

Align DATA_SCHEMA, REPRODUCIBILITY and the artifact manifest with the physics
pipeline, and reorganise the documentation into docs/. GitHub issue #2.

### Story 3.2: Add CI to run pytest on push and pull request

Add a GitHub Actions workflow with a fast unit-test job and an advisory
full-stack installation job. GitHub issue #12.

### Story 3.3: Run end-to-end pipeline verification

Regenerate every artifact from raw inputs on a clean machine with GRASS GIS, and
compare against the artifact manifest. GitHub issue #4.
