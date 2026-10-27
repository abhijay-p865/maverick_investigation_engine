# Project Progress Log

## 2026-10-24 — Chunk 1 started

Completed:

- Read the full `PRD.md` and inspected the repository.
- Confirmed the repository initially contained the PRD and the IndPenSim MATLAB/Octave source, with no generated CSV files.
- Confirmed normal `python`, GNU Octave, and MATLAB commands were unavailable.
- Located Codex's bundled Python 3.12 runtime and created `.venv`.
- Installed NumPy, pandas, scikit-learn, SciPy, matplotlib, joblib, PyArrow, PyYAML, and pytest.
- Added the modular package structure, project rules, plan, README, dependency metadata, and Git exclusions.

Decisions:

- Use the small verified public IndPenSim 100-batch measurement files for the pilot because no simulator runtime is installed and the simulator repository contains no generated CSVs.
- Keep all ground-truth fields outside feature engineering.
- Use complete-batch, control-regime-stratified healthy splits.

Tests/results:

- Pending after the first package modules are added.

Blockers:

- None. Octave is not needed for this pilot because genuine published simulator output is available.

Next:

- Implement and test the IndPenSim loader and validation layer.

## 2026-10-24 — Chunks 2 and 3 completed

Completed:

- Downloaded and hash-verified the public 100-batch IndPenSim parquet compilation.
- Implemented a source adapter, stable internal names/units, exact fault-interval preservation, conservative short-gap interpolation, and clear validation failures.
- Documented the source-quality issue in the compilation's misleading `Batch ID` and `Fault flag` columns. Reconstructed batch IDs from the 100 time resets and checked them against the 100 statistics records.
- Implemented 72-minute causal rolling windows with mean, standard deviation, slope, and first-to-last change for dissolved oxygen, pH, temperature, and pressure.
- Added control-strategy-specific phase normalization using only healthy training data.
- Created a complete-batch 54/18/18 healthy train/validation/test split plus 10 fault development-holdout batches (seed 42).
- Trained `isolation_forest_v1` on healthy training batches and set its threshold to the 99th percentile of healthy validation scores.
- Saved the model, readable feature/model metadata, split manifest, timeline CSVs, JSON reports, and PNG charts.
- Verified the standalone CSV command by exporting source batch 91 to CSV, analyzing it, and confirming all 1,285 timeline rows exactly matched direct parquet analysis.

Commands executed successfully:

```powershell
.\.venv\Scripts\python.exe scripts\acquire_indpensim.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m maverickx.cli pilot
.\.venv\Scripts\python.exe -m maverickx.cli analyze data\processed\Batch_91_fault_3.csv --output-dir outputs\csv_smoke_test
```

Verified results:

- Dataset: 113,935 rows; 100 batches; 90 healthy and 10 fault batches; 12-minute nominal sampling.
- Automated tests: 14 passed in 8.53 seconds on the final v1 code.
- Healthy test false-alert window fraction: 0.946% across 18 batches.
- Healthy false-alert events: 0.331 per batch-day.
- At least one in-interval alert: 8 of 10 fault batches.
- Known fault intervals with at least one alert: 13 of 21 (61.9%).
- Window recall inside known fault intervals: 7.01%; precision: 16.45%; F2: 7.92%.
- Fault-batch 91 CSV smoke test: 1,285 windows, 37 flags, 15 grouped events, maximum score 0.70735.

Issues and interpretation:

- The current detector misses both coolant-flow fault intervals and only flags parts of several other intervals. It is a working pipeline, not yet a strong detector.
- Fault windows were inspected while debugging preprocessing, so these 10 batches are a development holdout, not an untouched final benchmark. Additional simulator runs are needed for unbiased final testing.
- The first artifact uses only the four required sensors so minimum-input CSVs remain usable. Direct manipulated-variable channels could improve process-fault detection and should be compared as a separate extended model.
- Events can still be fragmented; the grouping bridges one clear 12-minute window but does not perform aggressive smoothing.
- These measurements contain simulated process faults, not microbial contamination.

Next:

- Before SHAP, expand the simulator dataset to create a fresh untouched test set and compare window sizes, a statistical baseline, One-Class SVM, and an optional extended-sensor Isolation Forest. After selecting a detector, add SHAP explanations and investigation prioritization.

## 2026-10-25 — Model improvement and independent validation completed

Completed:

- Installed and verified portable GNU Octave 11.3.0, then smoke-tested the full IndPenSim-to-CSV path.
- Locked a 48-batch campaign and development/test split before generation.
- Generated 16 fresh healthy batches and 32 fresh fault batches (four for each code 1–8) from IndPenSim commit `45c3d8c53ba423671acfb63de5855ecd4c8aa9d4`.
- Validated 55,155 rows, 0.2-hour sampling, correct fault codes, zero missing required measurements, and saved SHA-256 hashes for all CSVs.
- Compared Isolation Forest, One-Class SVM, and a robust phase-adjusted statistical baseline across 3, 6, 12, and 18-sample windows, core versus extended sensors, and six healthy-validation thresholds.
- Selected a three-sample core-sensor One-Class SVM under the predeclared maximum of 0.5 healthy false alert events per batch-day.
- Saved the selection lock before opening the untouched test partition, then evaluated the selected model and v1 once on 6 healthy and 16 fault test batches.
- Added interval-level diagnostics and headless timeline rendering. No SHAP or investigation-ranking component was started.

Independent test results:

- Selected model fault-event recall: 27/34 = 79.4%; v1: 25/34 = 73.5%.
- Healthy false alert events per batch-day: 0.330; v1: 0.487.
- Healthy false-alert window fraction: 0.828%; v1: 2.344%.
- Median detected-event delay: 0.0 h for both; mean delay: 14.54 h selected versus 12.16 h v1.
- On the 23 intervals detected by both, the selected model was 2.14 h faster on average.
- Window precision/recall/F2: 88.66% / 13.86% / 16.67%; v1: 45.59% / 3.56% / 4.37%.
- Coolant-flow recall: 2/2 selected versus 0/2 v1; selected delays were 9.8 h and 9.0 h.

Issues and interpretation:

- Seven of 34 test fault intervals remain missed: two late substrate-feed periods, two second base-flow periods, two related combined-fault periods, and one temperature-sensor batch.
- Mean delay regressed because the selected model found additional but late events, especially coolant; common-event delay improved.
- The extended sensor model reduced false alarms but had slightly lower development event recall, so it was not selected under the declared recall-first policy.
- The max robust-z statistical baseline saturated at the normalizer clip and produced no useful alerts at the tested healthy thresholds. It remains a comparison baseline, not a candidate for deployment.
- The model detects simulated process and sensor faults only. It is not validated for microbial contamination or real manufacturing data.

Verification commands:

```powershell
.\.venv\Scripts\python.exe scripts\generate_phase2_indpensim.py --octave "tools\octave-11.3.0-w64\octave-11.3.0-w64\mingw64\bin\octave-cli.exe" --workers 6
.\.venv\Scripts\python.exe scripts\run_phase2_experiment.py
.\.venv\Scripts\python.exe scripts\run_phase2_experiment.py --render-only
.\.venv\Scripts\python.exe scripts\validate_phase2_dataset.py
.\.venv\Scripts\python.exe -m pytest -q
```

Next:

- The detector is ready for a research-stage SHAP/explanation and investigation-prioritization prototype, while retaining clear warnings that it is not production-ready and does not detect contamination.

## 2026-10-25 — Time-boxed threshold improvement completed

Completed:

- Swept 11 nearby SVM thresholds using development batches only; the estimator, features, phase normalization, and rolling window were unchanged.
- Selected the 98.45th healthy-score percentile because it achieved the best development fault-event recall under the existing 0.5 false-alert-event cap.
- Preserved the original locked artifact and saved the operational update as `artifacts/phase2/selected_detector_v2_1.joblib`.
- Ran an explicitly post-hoc confirmation on the already-viewed test set. These numbers do not replace the original independent benchmark.

Measured results:

- Development event recall: 85.5% to 90.9% (+3 of 55 intervals); false events: 0.336 to 0.461 per healthy batch-day.
- Post-hoc test event recall: 79.4% to 85.3% (+2 of 34 intervals).
- Post-hoc mean detection delay: 14.54 h to 7.14 h.
- Both recovered events were second base-flow intervals, detected after 0.2 h and 0.4 h.
- No previously detected test event was lost.
- Post-hoc healthy false events increased to 0.470 per batch-day, still below the declared 0.5 cap.
- Window precision decreased modestly from 88.66% to 86.13%, while window recall and F2 improved.

Decision:

- Use v2.1 for SHAP explanations and investigation reporting. Do not perform more detector optimization before the end-to-end prototype is complete.

## 2026-10-25 — Phase 3 explanations and investigation reports completed

Completed:

- Added Kernel SHAP explanations for the exact negative `score_samples` quantity returned by the saved v2.1 One-Class SVM.
- Used 20 representative healthy phase-normalized windows as the SHAP background and 100 coalition samples per explained window.
- Aggregated feature contributions into event-level sensor evidence while preserving observed values, units, trends, and timestamps.
- Added transparent rules covering the 12 PRD investigation areas. Rules combine positive SHAP support with optional actuator/process deviations from healthy phase references.
- Stabilized low-variance process comparisons with documented engineering-scale floors; priority scores are explicitly not root-cause probabilities.
- Added JSON, Markdown, window CSV, anomaly timeline PNG, and SHAP event PNG outputs for any supported batch CSV.
- Added a predefined IndPenSim-to-investigation mapping before the final benchmark summary and evaluated combined faults separately.

Measured results on the existing Phase 2 test set:

- Detector result remained unchanged: 29/34 known fault intervals detected (85.3%).
- Investigation Top-1 among detected intervals: 18/29 = 62.1%.
- Investigation Top-3 among detected intervals: 28/29 = 96.6%.
- End-to-end Top-3, counting detector misses as misses: 28/34 = 82.4%.
- Single-fault Top-3 among detected intervals: 20/21 = 95.2%.
- Combined-fault Top-3 among detected intervals: 8/8 = 100%; reported separately.
- SHAP additivity passed for every explained window in 17 evaluation reports; worst absolute reconstruction residual was 4.44e-16.
- Maximum measured report-generation runtime in that run was 1.84 seconds per batch on this workspace with one representative window per event.
- Healthy example `phase2_011` still produced 30 flagged windows grouped into 8 events, demonstrating the remaining false-alert burden.

Issues and interpretation:

- These ranking results are post-hoc because the Phase 2 test set had already been viewed during the threshold decision, and a numerical context-scaling defect was corrected after the first Phase 3 benchmark run.
- The only detected interval whose acceptable area was absent from the Top 3 was the early pressure interval in `phase2_034`; downstream DO, thermal, and dosing evidence outranked pressure review.
- Five intervals were not ranked because the frozen detector did not flag them: two late substrate-feed intervals, two combined base-flow intervals, and one temperature-sensor batch.
- Coolant review was Top-3 but not Top-1 for the two coolant cases because temperature-probe review received stronger model evidence; this is useful uncertainty rather than proof that the probe was responsible.
- Healthy data can still produce investigation leads. The report states that these are review priorities, not diagnoses.
- No microbial contamination records, operator logs, or real manufacturing data were evaluated.

Verification commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate_phase3.py --force
.\.venv\Scripts\python.exe -m maverickx.cli investigate "data\raw\phase2_indpensim\csv\Batch_40_fault_8.csv" --output-dir "outputs\phase3\demo"
```

Next:

- Freeze this end-to-end prototype for the demonstration. If time permits, improve report presentation and add real process-log inputs; do not claim contamination-source identification.

## 2026-10-25 — Interactive website and teammate handoff completed

Completed:

- Added a FastAPI website that accepts a CSV upload and runs the existing v2.1 detection, Kernel SHAP, investigation ranking, and report-generation pipeline.
- Added a responsive MaverickX-inspired visual system based on the public site's monochrome industrial design, bold typography, square controls, and minimal accent color.
- Added result summary cards, ranked investigation evidence, event-level SHAP details, sensor trends, timeline/SHAP chart switching, and JSON/Markdown/CSV downloads.
- Added 25 MB upload limits, extension validation, clear validation errors, safe result-file routing, and path-traversal protection.
- Packaged only the runtime assets needed by a fresh clone: the frozen v2.1 model, ten healthy reference batches, and one demonstration CSV.
- Added `IMPROVEMENT_ROADMAP.md`, written for a teammate with similar scientific and programming experience.
- Added a GitHub Actions workflow for automated tests.

Verified:

- The website health endpoint reported the v2.1 model and healthy background ready.
- A real `Batch_40_fault_8.csv` upload completed through the HTTP API and returned pH probe as the first investigation priority.
- The returned JSON report was downloadable through the protected results route.
- Automated tests: 28 passed. Remaining warnings are third-party SHAP/Matplotlib and FastAPI TestClient deprecation notices.

Issues:

- The website is a local demonstration server. It does not yet include authentication, durable job storage, a background queue, or production deployment.
- Uploaded files and reports remain on local disk until manually removed.
- The GitHub repository still needs to be connected, committed, pushed, and confirmed public.

Next:

- Push the tested source and small runtime assets to the public GitHub repository, then hand off using `IMPROVEMENT_ROADMAP.md`.

## 2026-10-25 — Public browser deployment prepared

Completed:

- Added a Vercel-compatible FastAPI entrypoint and deployment configuration for the existing v2.1 model pipeline.
- Added a one-click example analysis so visitors can evaluate the complete workflow without downloading a CSV or using the command line.
- Embedded generated reports and charts in the analysis response so browser downloads work reliably on stateless hosting.
- Limited public uploads to 4 MB, which covers the current IndPenSim batches while staying below the hosting platform's request limit.
- Pinned the model's runtime dependency versions and removed the web deployment's unnecessary PyArrow dependency.

Verified:

- Full automated suite: 31 tests passed.
- A real example analysis returned a 1.6 MB response, below the host's 4.5 MB response limit.
- Hosted-mode simulation used temporary storage and removed each job directory after producing the browser response.
- Deployment configuration and the public FastAPI entrypoint imported successfully.

Issues:

- Production deployment is waiting for the project owner's one-time Vercel team-access approval.

Next:

- Connect the public GitHub repository to Vercel, deploy the production build, run a real example analysis at the public URL, and add that URL to the README.

## 2026-10-25 — Public no-install deployment completed

Completed:

- Connected the public GitHub repository to Vercel with automatic production deployment from `main`.
- Published the permanent public tool at `https://maverick-investigation-engine.vercel.app/`.
- Enabled Vercel large Python functions so the complete scikit-learn, SHAP, pandas, and charting runtime remains intact.
- Converted only the saved model's pandas column-label metadata from PyArrow-backed strings to ordinary Python strings, removing an unnecessary hosted dependency without retraining or retuning v2.1.
- Added `scripts/make_model_portable.py` so that serialization change is reproducible and checks exact score equality before replacing the artifact.

Verified:

- Portable-artifact check: every anomaly score for the demonstration batch was exactly unchanged, with zero numerical tolerance.
- Full local automated suite: 31 tests passed.
- Public home page: HTTP 200 and expected MaverickX interface rendered in a browser.
- Public health endpoint: ready, v2.1 model present, ten healthy background batches present, hosted mode active.
- Public one-click example: HTTP 200 in 8.4 seconds; 8 anomaly events; pH probe ranked first; all three report downloads and both embedded charts returned.
- Public CSV upload: HTTP 200 in 3.2 seconds; 8 anomaly events; SHAP additivity checks passed; three downloadable report formats returned.
- Browser result rendering: summary, rankings, anomaly timeline/SHAP tabs, event evidence, sensor trends, downloads, and scientific limitations all appeared successfully.

Issues:

- The public prototype intentionally has no user accounts or durable job history. Uploaded data is processed for the response using temporary function storage.
- The first request after an idle period may be slower because the scientific Python function can start cold.
- The tool remains validated on simulated process and sensor faults, not real microbial contamination data.

Next:

- Give the public URL to the product-website teammate and link to it as the interactive investigation demonstration.
- Keep the current detector and scientific limitations unchanged unless a future validation phase supplies new evidence.

## 2026-10-25 — Fresh downloadable IndPenSim test pack

Completed:

- Extended `scripts/generate_phase2_indpensim.py` with optional custom fault-code, campaign-name, and output-directory arguments while preserving the locked Phase 2 defaults.
- Generated five new batches directly with GNU Octave: one healthy batch plus aeration, coolant-flow, temperature-sensor, and pH-sensor fault batches.
- Saved the CSVs to `C:\Users\abhij\Downloads\MaverickX_fresh_IndPenSim_2026-10-25` for easy upload to the public website.

Verified:

- The simulator completed all five independent batches successfully.
- The project loader accepted all five CSVs: 1,130–1,195 rows each, 0.2-hour median sampling, no duplicate timestamps, and no missing required measurements after loading.
- SHA-256 hashes were calculated for all five files so their exact contents can be distinguished from earlier campaigns.

Issues:

- These are fresh simulated process and sensor-fault batches, not real manufacturing or microbial-contamination data.

Next:

- Upload any CSV from the Downloads test-pack folder to the public investigation tool and compare the healthy and faulty reports.

## 2026-10-26 — Anomalyze interface and evidence-timing revision

Completed:

- Removed the top MaverickX/research-prototype bar and the separate `01 / ANALYZE` / `Evidence first. Conclusions second.` introduction.
- Renamed the visible web experience to Anomalyze and updated the upload explanation accordingly.
- Made investigation cards use the same evidence vocabulary as the on-page guide: confirmed observation, possible explanation, suggested check, and priority score.
- Added `When observed` to every batch-level priority and linked it to the anomaly event where that rule received its strongest evidence score.
- Relabeled event details as confirmed model observations and possible investigations to verify.
- Added more precise display behavior for very small nonzero anomaly scores.

Verified:

- JavaScript syntax validation passed.
- Focused web and investigation tests passed, including a new test that prevents a batch-level priority from being misleadingly linked to every anomaly event.
- Local browser verification confirmed the removed sections, Anomalyze wording, labeled evidence, strongest-event time, priority score, event details, and responsive layout all render correctly.
- Reanalyzed the fresh aeration-fault batch without using its fault label as input: six events and 45 flagged windows were reproduced, while aeration investigation remained fifth. This is recorded as a real ranking miss rather than being corrected from ground truth.

Issues:

- The fresh aeration-fault example shows that a correct detected period does not guarantee the correct equipment area ranks first or in the Top 3. The model/rules can prioritize correlated pH, thermal, and dissolved-oxygen evidence instead.
- `When observed` identifies the event with the strongest evidence for a recommendation; it does not claim that the physical issue began at that exact time.

Next:

- Deploy the interface revision and verify the complete report on the public production URL.

## 2026-10-25 — Report contract and reproducibility hardening

Completed:

- Added versioned JSON Schema `schemas/investigation-report-v1.schema.json` and a Phase 3 regression assertion that validates the generated report against it.
- Added report schema version, UTC analysis timestamp, SHA-256 of the exact CSV bytes parsed, and model artifact SHA-256 when the CLI or web server loads the detector from a known file. Reports created from in-memory batches accurately record null hashes.
- Kept existing model version and feature configuration in the report. A code commit SHA is not claimed because this checkout has no Git metadata.
- Added a CI runtime-asset check and a regression test for the model, demo CSV, and healthy background CSVs.
- Added the downloadable example CSV and a separate upload schema guide with the required IndPenSim headings, units, and validation behavior.
- Added finite-value rejection for required measurements and timestamps.

Verified:

- `PYTHONPATH=src pytest -q tests/test_data.py tests/test_package.py`: 10 passed, 1 skipped (optional downloaded IndPenSim dataset unavailable).
- `PYTHONPATH=src pytest -q --ignore=tests/test_web.py --ignore=tests/test_phase3.py`: 22 passed, 1 skipped.
- Python syntax compilation passed.
- JSON Schema meta-schema validation and GitHub Actions YAML parsing passed using the available Anaconda Python environment.
- Initial full-suite collection in the base environment was blocked by missing `shap` and `fastapi`. After installing only the declared `requirements-dev.txt` dependencies into `.venv`, the full suite passed (see the later local verification entry).

Partially completed:

- Upload guidance and a downloadable template are available, but there is no pre-analysis quality preview or editable user column mapping.
- Direct Python callers that pass an in-memory detector do not provide an artifact hash; the report records null in that case. Git commit SHA is not available in this checkout.

Remaining:

- Add configurable plant-tag/unit mappings and a pre-analysis preview before supporting non-IndPenSim exports.

## 2026-10-25 — Local verification and public deployment check

Locally verified:

- Installed the existing development requirements in the ignored project `.venv`; no undeclared dependencies were added.
- `.venv/bin/python -m pytest -q`: 35 passed, 1 skipped, with 4 third-party deprecation warnings. The skip is the optional integration test requiring the separately downloaded IndPenSim parquet dataset.
- The Phase 3 generated-report test validates the saved JSON against the versioned schema. The HTTP upload test validates the returned report against the schema and checks exact input and model file SHA-256 values.
- The existing v2.1 model artifact was not modified. No detector retraining, threshold changes, or ranking-weight changes were made.
- The v2.1 artifact SHA-256 remains `0ceead8f569fdd9a4f5045b057f50084bbc2dc08d111a35df8a8678c7cd782a2`.

Public site checked:

- The site loaded and its one-click demo and CSV upload both generated reports (8 events; v2.1 model).
- The live upload report did not contain `analysis_metadata`, and the live upload page did not contain the new CSV example download or schema-guide link. The public deployment therefore does not include these local changes.

Deployment status: not deployed.

- The Desktop source folder has no `.git` directory, `.vercel/project.json`, or installed Vercel CLI. It has been left unchanged.
- The user-provided GitHub URL was verified as the application repository: its homepage metadata matches the live URL and its default/only remote branch is `main` at `de14cd4`.
- GitHub reports a successful `Vercel` status for that `main` commit, targeting the existing `abhijayprabhu0-7017s-projects/maverick-investigation-engine` project. This confirms the GitHub repository is associated with the live Vercel project.
- The reviewed local changes are in a separate clone on feature branch `codex/complete-report-metadata`; no production settings have been changed. Publication is pending the authenticated push/PR workflow.
- A `git push --dry-run` to the verified remote was blocked because the authenticated account lacked write permission. The branch has not been pushed, no PR has been opened, and no deployment has been triggered.

Remaining:

- Pre-analysis quality preview and user-editable column mapping.
- Configurable plant tag/unit mappings, broader scientific validation, process logs, false-alert reduction, and production security/storage/job processing.
- Deploy the verified local changes and re-check the live upload report and metadata.

## 2026-10-25 — Desktop integration and handoff verification

Completed:

- Compared the Desktop project with reviewed commit `36a8708`. All reviewed source, test, schema, upload, and CI changes were already present; only the repository/deployment status wording in this progress log differed. Integrated the verified repository association and actual 403 push-permission result without replacing the rest of the Desktop log.
- Preserved the v2.1 model artifact. Its SHA-256 is `0ceead8f569fdd9a4f5045b057f50084bbc2dc08d111a35df8a8678c7cd782a2`.

Verified:

- Bare `.venv/bin/python -m pytest -q` could not collect tests because this existing environment does not resolve the editable package when invoked without a source path (`ModuleNotFoundError: maverickx`).
- `PYTHONPATH=src .venv/bin/python -m pytest -q`: 35 passed, 1 skipped, 4 third-party deprecation warnings. The skip requires the separately downloaded IndPenSim parquet dataset.
- The generated Phase 3 report and HTTP CSV upload report passed JSON Schema validation; the upload test checked input/model hashes and report response. Example download and schema-guide routes passed.

Remaining:

- The verified GitHub repository refused push access for `sreyakurup0-alt` with HTTP 403. The local reviewed branch and this Desktop project remain ready for handoff; no PR, merge, or deployment occurred.
- Create and validate the requested self-contained ZIP handoff before sending it to the repository owner.

## 2026-10-26 — Reviewed handoff integrated into the working repository

Completed:

- Applied the reviewed `Maverick_Handoff` patch directly to its intended base commit, `de14cd4d40b61d0e29146d86788f563bc6601fe8`.
- Added the versioned report schema, report provenance hashes and UTC timestamp, finite-value input validation, runtime-asset CI checks, CSV schema guidance, downloadable example CSV, and their regression tests.
- Installed the newly declared `jsonschema` development dependency in the ignored local `.venv` so the upgraded suite could be exercised.
- Preserved the existing v2.1 One-Class SVM model without retraining or changing its threshold, features, or investigation-ranking weights.

Verified:

- `PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q`: 36 passed with 4 third-party deprecation warnings in 257.30 seconds.
- The optional IndPenSim parquet integration test ran in this workspace rather than being skipped.
- The model SHA-256 remains `0ceead8f569fdd9a4f5045b057f50084bbc2dc08d111a35df8a8678c7cd782a2`.
- Git's patch check and post-application diff check passed.

Issues:

- The four warnings come from SHAP/Matplotlib and FastAPI/Starlette dependencies; they do not indicate failed MaverickX behavior.
- This upgrade improves engineering quality and reproducibility, not measured model performance.

Next:

- Review and commit the integrated changes, then deploy and verify the live report metadata, CSV guide, example download, and upload flow when publication is requested.

## 2026-10-26 — Plain-language website results and score definitions

Completed:

- Added a presentation-only translation layer that preserves stable machine identifiers in JSON/CSV output while giving measurements and rolling features readable website/report names.
- Replaced raw feature names in event tables with labels such as `Dissolved oxygen — recent variability` and added a short explanation of what each measurement pattern represents.
- Replaced internal sensor and process-channel identifiers in investigation observations, event interpretations, sensor trends, Markdown reports, and generated chart legends.
- Renamed technical website labels to plain-language terms, including `Unusual periods`, `Analysis windows flagged`, `Highest unusual-behavior score`, and `Model evidence`.
- Defined the priority score as a relative 0-to-1 evidence score: higher values mean stronger support for checking an area before lower-ranked areas, not a probability or diagnosis.
- Added concise explanations for the unusual-behavior score and analysis windows and clarified that model effects do not establish a physical cause.

Verified:

- Python compilation and JavaScript syntax checks passed.
- Focused website and Phase 3 tests passed: 13 passed with 4 third-party deprecation warnings.
- Full suite passed: 36 passed with the same 4 third-party deprecation warnings.
- Local browser verification ran the packaged example end to end and confirmed readable feature names, definitions, score wording, investigation observations, sensor trends, and chart legend labels without raw variable names.

Issues:

- The application still exposes technical identifiers in machine-readable JSON/CSV downloads by design so integrations remain stable; they are no longer used as user-facing website labels.
- This presentation change does not alter model scores, thresholds, ranking weights, or scientific performance.

Next:

- Possible future improvement: design a separate batch-record and SOP analysis tool that can connect operational evidence to investigation rankings. Any future implementation must keep those documents out of anomaly-model features and must not present a match as proof of contamination or root cause.

## 2026-10-26 — Batch-record and SOP analysis deferred

Completed:

- Stopped development of the batch-record and SOP feature at the project owner's request.
- Removed the unfinished document parsers, upload controls, API fields, ranking adjustments, tests, and project dependency declarations.
- Retained the idea only as the possible future improvement above.

Verified:

- Repository search confirmed that the unfinished document-ingestion implementation is no longer present.
- Python compilation and JavaScript syntax checks passed.
- Focused Phase 3 and website tests passed: 13 passed with 4 third-party deprecation warnings.

Issues:

- The optional document libraries installed in the ignored local virtual environment are harmless but are no longer declared or required by the project.

Next:

- Keep the current product focused on CSV batch analysis and the completed plain-language website results.

## 2026-10-26 — Day 1 multi-timescale detector experiment

Completed:

- Added a reproducible development-only experiment comparing the current three-sample v2.1 One-Class SVM with short-plus-long-window candidates using 12-sample (2.4 h) and 18-sample (3.6 h) branches.
- Put the two branch scores on a common empirical healthy-development percentile scale and calibrated the combined threshold using healthy development batches only.
- Preserved the complete-batch split, kept fault labels out of model features, did not load the Phase 2 test partition, and did not overwrite the v2.1 production artifact.
- Saved the development leaderboard and decision record under `outputs/day1_multiscale/`.

Verified:

- The selected development candidate was the three-plus-12-sample SVM at the tested 0.975 healthy-score quantile.
- Development fault-event recall was unchanged at 50/55 (90.9%).
- Healthy false-alert episodes decreased from 0.461 to 0.377 per batch-day.
- Window recall increased from 17.82% to 22.66%, and window F2 increased from 21.01% to 26.18%.
- Window precision decreased from 73.90% to 68.94%, while the healthy false-alert window fraction increased from 1.56% to 2.50%. The candidate therefore produces fewer but longer healthy alert episodes.
- No fault category gained or lost a detected interval on development data.
- The experiment runner compiled and completed successfully. Focused regression tests passed: 8 passed.

Issues:

- This is mixed development evidence, not an independent performance result. The candidate improved alert episode burden and window coverage but did not recover an additional fault interval and reduced precision.
- The original Phase 2 test set has already been viewed, so it cannot provide a new independent claim. A genuinely fresh simulator validation set is required before promotion.

Next:

- Advance the 3+12-sample candidate only to a Day 2 fresh-validation decision. Keep v2.1 in production unless the candidate demonstrates a worthwhile tradeoff on newly generated complete batches.

## 2026-10-27 — Website copy, scientific limits, hero image, and domain

Completed:

- Rewrote the landing-page headline and supporting copy to state the workflow directly without model-version language or inflated claims.
- Replaced the technical `What happens next` descriptions with four general steps: check the file, find unusual periods, show the evidence, and rank follow-up checks.
- Reduced the scientific-limits section from six partly repetitive statements to four concise limitations and replaced the diagnostic-sounding headline with guidance-focused wording.
- Added the user-supplied Edward Jenner/Pexels laboratory image as the hero background with a heavy black overlay for legibility.
- Renamed the Vercel project to `maverickx-anomalyze` and attached the verified production domain `maverickx-anomalyze.vercel.app`.

Verified:

- Focused Phase 3 and website tests passed: 13 passed with 4 third-party deprecation warnings.
- A separate real-pipeline regression check passed after removing the remaining visible v2.1 summary wording.
- Local browser verification loaded the hero asset, completed the example analysis through the API, rendered the simplified report language, and displayed the revised four-item limits section.
- The complete automated suite passed: 38 passed with the same 4 third-party deprecation warnings.
- The reviewed source was pushed to `main`, the Vercel production deployment reached `READY`, and `https://maverickx-anomalyze.vercel.app/` served the new landing page.
- Production browser verification ran the example analysis and displayed the simplified report, plain-language priority-score definition, and four scientific limits without visible model-version wording.

Issues:

- No blocking issues. The test suite still reports four third-party deprecation warnings from SHAP/Matplotlib and FastAPI/Starlette.
- The previous Vercel address remains attached as an alias, while `maverickx-anomalyze.vercel.app` is the primary requested address.

Next:

- Review the live site with stakeholders and collect feedback using real, appropriately governed batch CSVs. Continue to describe the tool as an investigation aid evaluated on simulated process and sensor faults.

