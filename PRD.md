# Product Requirements Document

## MaverickX Post-Batch Anomaly Detection & Investigation Engine

**Version:** 1.0
**Project type:** Standalone machine-learning research prototype
**Primary development language:** Python
**Developer:** Codex, with human review at major milestones
**Website integration:** Out of scope for this build

---

## 1. Product Summary

Build a standalone AI-powered system that analyzes historical bioreactor sensor measurements from a completed fermentation batch, identifies periods of abnormal process behavior, explains why those periods were flagged, and produces a prioritized list of potential investigations.

The system is intended for **post-batch investigation**, especially when contamination has already been confirmed or strongly suspected.

Its purpose is not to declare that a batch is contaminated. Instead, it helps operators answer:

1. When did unusual behavior occur during the batch?
2. Which sensor measurements and trends were most influential?
3. What process systems, equipment, or operational events should be investigated first?
4. What evidence supports each recommendation?
5. What cannot be concluded from the available data?

The tool should produce a clear, reproducible report that can eventually be connected to a website developed separately.

## 2. Problem Statement

Biomanufacturing teams collect large amounts of sensor data during fermentation. When a batch becomes contaminated or otherwise fails, operators may need to manually review historical measurements and process logs to understand what happened.

This process can be time-consuming, especially when several measurements change at once and the underlying cause is unclear.

A tool that detects unusual patterns, highlights contributing features, and organizes possible investigation leads could reduce the time needed to review a failed batch.

**Important scientific limitation:** Unusual sensor behavior does not prove microbial contamination, and SHAP explanations do not establish physical root causes or the point of contamination entry.

## 3. Users and Workflow

**Primary user:** Bioprocess scientist, process engineer, or biomanufacturing operator.

**Typical workflow:**

1. A fermentation batch is completed and selected for investigation.
2. The user provides historical sensor data from that batch.
3. The software validates and processes the data.
4. A trained anomaly detection model identifies unusual periods relative to healthy historical behavior.
5. SHAP identifies features that contributed to anomaly scores.
6. An investigation engine combines SHAP evidence, observed sensor patterns, and any available process logs.
7. The software generates a ranked investigation report.
8. The user reviews the findings and independently verifies possible causes.

The system must also support analysis of healthy batches, so users can compare ordinary and abnormal behavior.

## 4. Project Scope

### In scope

- Acquiring and preparing appropriate public fermentation data.
- Generating labeled fault examples using IndPenSim if necessary.
- Supporting historical CSV batch input.
- Data validation and preprocessing.
- Time-series feature engineering.
- Training and comparing anomaly detection algorithms.
- Identifying historical anomaly periods.
- Generating SHAP explanations.
- Ranking potential investigations.
- Producing human-readable and machine-readable reports.
- Evaluating model performance using actual experimental results.
- Automated testing and reproducible documentation.
- Defining a stable interface for future website integration.

### Out of scope

- Building a website or dashboard.
- Deploying the model to the cloud.
- Building a real-time monitoring pipeline.
- Connecting directly to industrial bioreactor hardware.
- Automatically controlling bioreactor equipment.
- Diagnosing microbial species.
- Claiming that simulated equipment-fault results establish contamination-detection accuracy.
- Claiming that SHAP proves contamination sources.

## 5. Required Input Data

### 5.1 Minimum inputs

Each input batch should include:

- Unique batch identifier.
- Timestamp or elapsed batch time.
- Fermentation stage (if available)
- Dissolved oxygen.
- pH.
- Temperature.
- Pressure.

The dataset may use different original column names and units. Build an adapter that maps source-specific columns into a documented internal schema.

If a dataset lacks a required sensor, provide a clear validation error or use a documented, compatible reduced-sensor model. Do not silently fabricate missing readings.

### 5.2 Optional inputs

Where available, support:

- Aeration rate.
- Agitation speed.
- Substrate and nutrient feed rates.
- Acid and base dosing.
- Heating and cooling settings.
- Sampling events.
- Equipment maintenance or interventions.
- Batch outcome and contamination confirmation.
- Known fault labels and fault-onset times for evaluation.

Optional channels should enhance investigation accuracy without making the system unusable when they are absent.

### 5.3 Data quality

The pipeline must:

- Validate timestamps and column types.
- Identify missing values and inconsistent sampling intervals.
- Preserve measurement units.
- Identify duplicate readings.
- Handle short gaps conservatively.
- Clearly report large gaps or unusable data.
- Avoid deleting real anomalies as presumed measurement noise.
- Keep labels and known fault identities out of the detection model's features.

## 6. Dataset Requirements

### 6.1 Primary benchmark

Start with the open-source IndPenSim simulator:

[https://github.com/Lemniscabio/IndPenSim](https://github.com/Lemniscabio/IndPenSim)

It models industrial penicillin fermentation and supports process and sensor fault scenarios.

Use it to create or obtain:

- Fault-free fermentation batches.
- Aeration faults.
- Vessel pressure faults.
- Substrate feed faults.
- Base-flow faults.
- Coolant-flow faults.
- Temperature sensor faults.
- pH sensor faults.

Handle combined faults separately because they may have more than one accepted investigation.

### 6.2 Dataset acquisition

Codex should attempt dataset acquisition and setup directly.

If the simulator requires GNU Octave or MATLAB, inspect available tools and install or configure an appropriate supported runtime where possible.

Begin with a small pilot dataset. Generate additional batches for credible evaluation after the full pipeline works.

If data acquisition fails, document the exact blocker and provide a clear action for the human developer. Do not substitute invented real-world data or claim benchmark success.

### 6.3 Ground-truth labels

Preserve:

- Batch identifier.
- Fault code.
- Fault type.
- Known fault-active intervals.
- Simulation settings.
- Data-generation version.

Importantly, a batch-level fault code is not equivalent to a time-specific fault label. Use documented fault onset and end times for evaluation.

### 6.4 Training/testing separation

Split complete batches into independent training, validation, and test groups.

A reasonable initial split is approximately:

- 60% training.
- 20% validation.
- 20% testing.

Train healthy-only anomaly detectors using fault-free training batches.

Validation data should be used for tuning model parameters, alert thresholds, investigation-ranking weights, and other decisions.

The test set must remain untouched until final evaluation.

Do not leak batch identifiers, fault codes, future information, or ground-truth labels into model inputs.

Document random seeds and split membership.

## 7. Feature Engineering

The model must analyze patterns across time rather than treating each sensor measurement independently.

For each sensor, investigate rolling-window features such as:

- Mean.
- Standard deviation.
- Slope.
- First-to-last difference.
- Minimum and maximum.
- Change relative to an earlier window.

Start with a small, interpretable feature set, preferably about four features per sensor.

Evaluate several window sizes. For IndPenSim's 12-minute data, a window of approximately 60–72 minutes is a sensible initial candidate, but this must be tested rather than assumed to be optimal.

Account for process phase or elapsed fermentation time so that expected biological changes are not incorrectly treated as anomalies simply because the process progresses.

When reviewing a completed batch, the system may access the full historical record, but each window's detection features should initially depend only on measurements available up to that window's end. This allows meaningful fault-onset and detection-delay evaluation.

## 8. Anomaly Detection

### 8.1 Candidate models

Implement and evaluate at least:

1. Isolation Forest.
2. One-Class SVM.
3. A simple statistical threshold-based baseline.

Isolation Forest is the first recommended prototype, not a predetermined winner.

Select the best model based on held-out performance, false alarms, interpretability, complexity, and practical usability.

A temporal autoencoder is optional and should not delay the core prototype.

### 8.2 Required model outputs

For every analyzed time window, return:

- Window start and end time.
- Anomaly score.
- Alert threshold.
- Whether the window is flagged.
- Model version.

Do not describe an anomaly score as a contamination probability unless a separately validated probability model exists.

### 8.3 Anomaly event grouping

Group consecutive abnormal windows into understandable events.

Each event should include:

- Start time.
- End time.
- Duration.
- Maximum anomaly score.
- Representative anomalous windows.
- Relevant affected sensors.

The final report should focus on a manageable number of meaningful events rather than hundreds of redundant alerts.

## 9. SHAP Explainability

### 9.1 Purpose

For each important anomaly, determine which features contributed most to the model's anomaly score.

Use a SHAP method appropriate for the selected detection algorithm.

Explain the same score quantity that the anomaly detector uses. Document the score's sign and interpretation.

### 9.2 Requirements

- Use a representative healthy training-data background.
- Generate feature-level SHAP attributions.
- Preserve feature names, values, and units.
- Separate positive and negative contributions where relevant.
- Aggregate positive contributions by sensor for easier interpretation.
- Explain why a sensor's behavior was unusual, including direction and timing.
- Check explanation consistency and computational performance.

Do not interpret SHAP importance as proof of equipment failure or contamination entry.

### 9.3 Expected output

For a selected anomalous event, provide:

- Top contributing features.
- Contribution magnitudes.
- Corresponding observed trends.
- Sensor-level contribution summary.
- A concise plain-language interpretation.

## 10. Investigation Prioritization

### 10.1 Objective

Convert model explanations into a ranked shortlist of investigations.

The ranking should combine:

1. SHAP evidence.
2. Actual observed sensor patterns.
3. Available process and operational context.

If process logs are absent, the system must acknowledge that limitation.

### 10.2 Initial investigation categories

Include at least:

- Aeration and oxygen delivery.
- Agitation/mixing.
- Dissolved oxygen probe.
- Acid/base dosing and pH control.
- pH probe.
- Substrate/feed system.
- Cooling/heating system.
- Temperature probe.
- Pressure regulation.
- Pressure probe.
- Sampling, transfers, handling, or maintenance records.
- Broader contamination investigation when unexplained disturbances justify it.

Rules concerning contamination entry should be worded cautiously. A process event near an anomaly is an investigation lead, not evidence that the event introduced contaminants.

### 10.3 Ranking method

Develop a transparent, configurable scoring approach combining SHAP support and sensor-pattern agreement.

Where logs exist, incorporate temporal alignment with recorded events.

Investigate alternative weighting schemes on validation data rather than fixing arbitrary weights permanently.

For every recommendation, return:

- Investigation title.
- Rank.
- Priority score and its meaning.
- Supporting sensor features.
- Actual observed patterns.
- Relevant process-log events, if any.
- Suggested operator verification.
- Uncertainty or missing evidence.

Do not present priority scores as calibrated probabilities of physical root cause.

### 10.4 Verification with known faults

Create a predefined mapping from IndPenSim fault types to acceptable investigation categories.

Evaluate whether the correct investigation appears among the top recommendations.

Avoid tailoring that mapping to the held-out test results.

## 11. Performance Evaluation

### 11.1 Detection metrics

Calculate:

- Fault-event recall.
- Window-level precision and recall where labels permit.
- F2 score.
- False-alarm events per healthy batch-day.
- Detection delay from known fault onset.
- Results by individual fault category.

Measure event-level performance separately from individual window-level performance.

### 11.2 Investigation metrics

Calculate:

- Top-1 investigation accuracy.
- Top-3 investigation accuracy.
- Ranking accuracy by fault type.
- Performance on combined or ambiguous faults, reported separately.

### 11.3 Baseline comparisons

Compare the selected anomaly detector against statistical thresholds and at least one alternative ML method.

Use the same test batches and appropriately tuned decision thresholds.

### 11.4 Prototype targets

Suggested initial development targets:

- At least 90% fault-event recall.
- At least 85% Top-3 investigation accuracy.
- Lower false-alarm burden or better detection performance than the baseline.

These are targets only. Never fabricate performance numbers to meet them.

If the system fails to reach them, report measured results, diagnose limitations, and document improvements.

### 11.5 Scientific honesty

Results from IndPenSim must be labeled as simulation-based.

Do not claim measured microbial contamination detection or contamination source identification unless independently validated against appropriate real contamination records.

Provide the dataset version, test split, model version, parameters, and evaluation procedure for reproducibility.

## 12. Required Final Outputs

The completed project must produce the following.

### A. Trained model

A saved anomaly detection model, including preprocessing configuration and feature definitions.

### B. Batch analysis interface

A clear way to analyze a historical CSV batch using the local project.

It should require minimal manual intervention and have documented instructions.

### C. Machine-readable report

Produce a JSON result containing:

- Batch metadata.
- Data-quality summary.
- Model version.
- Anomaly events and timestamps.
- Anomaly scores.
- SHAP feature explanations.
- Ranked investigations.
- Supporting evidence and limitations.

### D. Human-readable report

Generate an easy-to-read Markdown report containing:

- Executive summary.
- Significant anomalous periods.
- Relevant sensor trends.
- SHAP explanations.
- Top investigation priorities.
- Recommended next checks.
- Important limitations.

Optionally generate charts as PNG files for the report, including a sensor timeline and feature-importance plot.

### E. Benchmark report

Generate a report containing genuine measured results:

- Dataset and split descriptions.
- Model comparison.
- Detection metrics.
- False alarms and delays.
- Investigation-ranking performance.
- Results by fault category.
- Known limitations.

### F. Tests

Implement automated tests for:

- Dataset loading and schema validation.
- Missing-data handling.
- Feature calculation.
- Train/test separation.
- Score direction and thresholds.
- SHAP output structure.
- Investigation ranking.
- Report generation.
- Reproducibility.
- End-to-end batch analysis.

### G. Documentation

Provide a README explaining setup, dataset preparation, model training, evaluation, and report generation in plain language.

Provide a concise file describing the input/output contract for future website integration.

## 13. Suggested Project Structure

Organize the project into clear modules or equivalent folders for:

- Data acquisition and simulation.
- Raw and processed data.
- Preprocessing and validation.
- Feature engineering.
- Model training.
- Anomaly inference.
- SHAP explanations.
- Investigation rules and ranking.
- Evaluation and benchmarking.
- Report generation.
- Tests.
- Documentation.
- Saved models and outputs.

Keep large generated datasets and model artifacts out of Git version control unless there is a specific reason to include them.

The precise implementation structure is Codex's decision, provided it remains maintainable.

## 14. Development Priorities

### Phase 1 — Data foundation

Acquire a small real simulator-generated dataset, inspect its schema, and create validated data-loading and feature-engineering modules.

### Phase 2 — First detection model

Train Isolation Forest on healthy data and produce an anomaly timeline for a faulty batch.

### Phase 3 — Explanation and ranking

Add SHAP, investigation rules, and a readable batch report.

### Phase 4 — Scientific validation

Expand dataset coverage, benchmark models, calculate performance metrics, and improve weaknesses.

### Phase 5 — Standalone delivery

Finalize tests, documentation, structured output, and the future integration contract.

Every phase must leave the project in a runnable state.

## 15. Working Instructions for Codex

The human project owner is not highly familiar with software engineering or machine learning implementation.

Therefore:

- Explain technical terms in plain language.
- Break work into small, understandable chunks.
- Before each chunk, summarize what will be built and why.
- After each chunk, summarize what was completed and what was tested.
- Run tests and fix problems rather than merely describing how to test.
- Record decisions and assumptions in project documentation.
- Prefer reliable, simple implementations over unnecessary complexity.
- Do not ask the user to manually perform tasks that Codex can execute.
- Only request user intervention when access, permissions, installations, or major product decisions require it.
- Do not invent results, datasets, test executions, or scientific validation.
- State clearly when a capability is incomplete or unverified.
- Keep the project focused on the model; do not start website development.

## 16. Definition of Done

The project is complete when:

1. A previously unseen historical fermentation batch can be analyzed without modifying source code.
2. The model identifies and timestamps unusual behavior.
3. The system explains significant anomalies using SHAP.
4. It produces a ranked list of relevant investigations with supporting evidence.
5. It generates both JSON and readable reports.
6. It has been evaluated on held-out labeled fault batches.
7. It publishes real measured metrics and baseline comparisons.
8. Automated tests pass.
9. The system can be reproduced from documented setup instructions.
10. Its input/output interface is ready for a teammate to integrate into a separate website.

**The final product is a validated post-batch investigation prototype, not a proven microbial contamination diagnostic system.**
