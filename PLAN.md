# Development Plan

## Milestone 1 — Project foundation

Deliverables: modular Python package, isolated dependencies, automated tests, README, working rules, data/output folders, and progress log.

Acceptance criteria:

- The package installs into `.venv`.
- `pytest` discovers and runs tests.
- Setup and reproduction commands are documented.

## Milestone 2 — Real IndPenSim data foundation

Deliverables: verified public dataset acquisition, source inspection, schema adapter, validation, data-quality report, batch/fault metadata, and loader tests.

Acceptance criteria:

- At least five healthy batches and several fault batches load from genuine IndPenSim output.
- Source file hashes, units, timestamps, batch IDs, and fault-active intervals are recorded.
- Labels are available for evaluation but cannot enter the model feature matrix.
- Missing required sensors, timestamps, duplicates, and short/long gaps are tested.

## Milestone 3 — First anomaly detector

Deliverables: causal rolling-window features, complete-batch split, phase-aware preprocessing, healthy-only Isolation Forest training, validation threshold, saved artifact, timeline/event output, chart, and model tests.

Acceptance criteria:

- Train, validation, and test batch IDs are disjoint.
- An unseen batch produces one numerical score and alert decision per eligible window.
- Score direction and threshold behavior are verified.
- A healthy and at least one faulty held-out batch are analyzed by real execution.

## Milestone 4 — Review and strengthen detection (completed 2026-10-25)

Deliverables: broader evaluation by fault type, false-alarm/event metrics, statistical baseline, One-Class SVM comparison, window-size study, and documented model selection.

Acceptance criteria:

- All comparisons use the same untouched test batches.
- Measured results and limitations are reported without extrapolating to contamination.

Outcome: 24 configurations were compared on development data. A three-sample,
core-sensor One-Class SVM was locked before a single independent evaluation on
22 fresh test batches. It improved fault-event recall and reduced healthy false
alerts relative to v1. See `docs/phase2_results.md`.

## Milestone 5 — Explanations and investigation prioritization (not started)

Deliverables: SHAP explanations for the selected score, sensor-level summaries, transparent investigation ranking, JSON/Markdown reports, and tests.

Acceptance criteria:

- Explanations reconcile with the score being explained.
- Recommendations state evidence, suggested checks, uncertainty, and missing context.
- SHAP is never described as proof of physical cause or contamination entry.

