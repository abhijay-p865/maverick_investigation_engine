# MaverickX Post-Batch Anomaly Detection

MaverickX is a standalone Python research prototype for reviewing a completed fermentation batch. It compares rolling sensor behavior with healthy historical batches and returns a time-by-time anomaly score. A high score means "unusual compared with the healthy training data"; it is **not** a contamination probability or a root-cause diagnosis.

The current foundation uses real simulator-generated IndPenSim data. It covers simulated process faults, not measured microbial contamination.

## What the working system does

- Downloads and verifies the public 100-batch IndPenSim dataset.
- Converts source-specific column names into a documented internal schema.
- Keeps batch identity and fault timing for evaluation while excluding labels from model inputs.
- Calculates causal rolling-window summaries for the four guaranteed sensors: dissolved oxygen, pH, temperature, and pressure. Extra process channels are preserved for later model comparisons.
- Compares Isolation Forest, One-Class SVM, and a robust statistical baseline,
  using only healthy batches for model fitting.
- Selects window size, sensor set, and threshold on development data while
  prioritizing fault-event recall, detection delay, and false-alert frequency.
- Runs the saved v2.1 detector on a completed batch, explains representative
  anomaly windows with model-agnostic Kernel SHAP, and ranks transparent
  investigation leads using SHAP plus optional process channels.
- Writes a window timeline, structured JSON, readable Markdown, and PNG charts.

The selected phase-2 detector is a One-Class SVM using three-sample rolling
windows and the four core sensors. Its original locked evaluation detected 27
of 34 known simulated fault intervals (79.4%) and produced 0.330 false alert
events per healthy batch-day. A later development-only threshold sweep produced
the v2.1 operational artifact. Its clearly labeled post-hoc test check detected
29 of 34 intervals (85.3%), with 0.470 false alert events per healthy batch-day
and a 7.14-hour mean detected-event delay. These are simulator results, not
contamination-detection claims.

## Setup on this Windows workspace

The normal `python` command is not installed, so this project uses Codex's bundled Python to create a local virtual environment. Once `.venv` exists, all normal project commands use its Python:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev]"
```

## Run the interactive website

The repository now includes a local FastAPI website with MaverickX-inspired
black-and-white styling. It accepts a completed IndPenSim-format CSV, runs the
saved v2.1 detector and Phase 3 workflow, then displays and downloads the
resulting investigation report.

```powershell
.\.venv\Scripts\python.exe -m uvicorn maverickx.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and upload the included example:

```text
runtime_assets/demo/Batch_40_fault_8.csv
```

The small `runtime_assets/` directory contains the frozen model, ten healthy
reference batches needed for SHAP/process comparisons, and one demonstration
batch. Large generated datasets, outputs, the virtual environment, and the
Octave runtime remain intentionally excluded from Git.

## Reproduce the pilot

```powershell
.\.venv\Scripts\python.exe scripts\acquire_indpensim.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m maverickx.cli pilot
```

The pilot command writes:

- `artifacts/isolation_forest_v1.joblib` — trained model plus preprocessing settings.
- `artifacts/isolation_forest_v1_metadata.json` — readable model parameters, threshold, and feature definitions.
- `artifacts/split_manifest.json` — exact healthy train/validation/test batch membership.
- `outputs/pilot/` — timelines, JSON summaries, and charts for one healthy and selected faulty batches.

## Reproduce phase 2

The original IndPenSim checkout must exist at `IndPenSim/`. The local portable
Octave path used for this run was:

```powershell
.\.venv\Scripts\python.exe scripts\generate_phase2_indpensim.py --octave "tools\octave-11.3.0-w64\octave-11.3.0-w64\mingw64\bin\octave-cli.exe" --workers 6
.\.venv\Scripts\python.exe scripts\run_phase2_experiment.py
.\.venv\Scripts\python.exe scripts\run_phase2_experiment.py --render-only
.\.venv\Scripts\python.exe scripts\validate_phase2_dataset.py
.\.venv\Scripts\python.exe scripts\quick_threshold_sweep.py --posthoc-test --adopt
.\.venv\Scripts\python.exe -m pytest -q
```

The generator intentionally refuses to overwrite existing simulator evidence.
The upstream simulator is stochastic, so a clean new campaign may not reproduce
identical numeric values; `outputs/phase2/dataset_validation.json` records the
exact SHA-256 hashes used in this evaluation.

Phase-2 outputs include:

- `artifacts/phase2/selected_detector.joblib` — locked detector and preprocessing;
- `artifacts/phase2/selected_detector_v2_1.joblib` — threshold-only operational update for SHAP and reporting;
- `outputs/phase2/selection_lock.json` — selected configuration and development results;
- `outputs/phase2/independent_test_comparison.json` — selected model versus v1;
- `outputs/phase2/development_leaderboard.csv` — all algorithm/window/sensor/threshold comparisons;
- `outputs/phase2/test_interval_diagnostics.json` — detected and missed interval evidence;
- `outputs/phase2/quick_threshold_sweep.json` — development-only threshold comparison;
- `outputs/phase2/quick_threshold_posthoc_test.json` — explicitly post-hoc v2.1 confirmation;
- `outputs/phase2/*timeline.csv` and `.png` — example healthy and coolant timelines.

## Analyze one historical CSV

After training the pilot model:

```powershell
.\.venv\Scripts\python.exe -m maverickx.cli analyze path\to\Batch_101_fault_0.csv --output-dir outputs\batch_101
```

The CSV adapter accepts IndPenSim's original exported headings. Required measurements are elapsed time, dissolved oxygen, pH, temperature, and pressure. Missing required columns or long missing-data gaps produce a clear validation error rather than invented measurements.

## Generate a complete post-batch investigation report

Use the frozen v2.1 model and the healthy Phase 2 background:

```powershell
.\.venv\Scripts\python.exe -m maverickx.cli investigate "data\raw\phase2_indpensim\csv\Batch_40_fault_8.csv" --output-dir "outputs\phase3\demo"
```

For each supported CSV, this command produces:

- `*_timeline.csv` with every rolling-window score and flag;
- `*_investigation.json` for website integration;
- `*_investigation.md` for a scientist-readable review;
- `*_timeline.png` with score and sensor traces; and
- `*_shap.png` with event-level positive SHAP evidence by sensor.

Reproduce the fixed Phase 3 evaluation with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_phase3.py --force
```

The final post-hoc evaluation found the predefined acceptable investigation in
the Top 3 for 28 of 29 detected simulated fault intervals (96.6%). Counting the
five detector misses as investigation misses gives an end-to-end Top-3 rate of
28/34 (82.4%). See `docs/phase3_results.md` for definitions and limitations.

## Important scientific limits

- The benchmark is a simulation of penicillin fermentation.
- Alerts identify unusual process patterns; they do not prove contamination.
- SHAP explains the model score; it does not prove a physical cause.
- Investigation scores are relative evidence priorities, not root-cause probabilities.

See [PLAN.md](PLAN.md) for milestones, [PROGRESS.md](PROGRESS.md) for executed
results, [docs/phase2_results.md](docs/phase2_results.md) for detector evaluation,
[docs/phase3_results.md](docs/phase3_results.md) for explanation/ranking results,
and [docs/data_contract.md](docs/data_contract.md) for the internal
input/output contract. See [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) for
the detailed teammate handoff and recommended next work.


