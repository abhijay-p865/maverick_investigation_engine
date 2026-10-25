# Phase 3: SHAP explanations and investigation prioritization

## What was built

The complete prototype now runs:

`IndPenSim CSV → saved v2.1 One-Class SVM → anomaly events → Kernel SHAP → ranked investigations → JSON/Markdown report`

SHAP explains the detector's actual public score: the negative of One-Class
SVM `score_samples`. Larger scores are more anomalous, so a positive SHAP value
means that feature pushed the score toward a stronger anomaly. The explainer
uses 20 representative healthy, phase-normalized rolling windows and 100
coalition samples for each selected event window.

Rankings use transparent evidence rules rather than a second classifier. Sensor
support comes from positive SHAP contributions. Optional aeration, agitation,
feed, dosing, coolant, heating, and pressure patterns are compared with healthy
values at the same batch phase. The output separates confirmed observations
from possible explanations and recommends operator checks without asserting a
root cause.

## Verification

The saved detector and threshold were not changed. Across 16 fault-batch
reports and one healthy report, every SHAP explanation reconstructed the model
score within the configured `1e-6` tolerance. The largest observed absolute
residual was `4.44e-16`.

Automated test result: **24 passed**. Tests cover report structure, SHAP
additivity, score quantity, probe-versus-utility ranking behavior, and honest
end-to-end metric accounting.

## Measured investigation results

The predefined acceptable-category mapping is
`configs/phase3_investigation_mapping.json`. Labels and known intervals were
read only after each report was generated.

| Scope | Detected intervals | Top-1 given detected | Top-3 given detected | End-to-end Top-3 |
|---|---:|---:|---:|---:|
| Single faults | 21/24 (87.5%) | 11/21 (52.4%) | 20/21 (95.2%) | 20/24 (83.3%) |
| Combined faults | 8/10 (80.0%) | 7/8 (87.5%) | 8/8 (100%) | 8/10 (80.0%) |
| All mapped intervals | 29/34 (85.3%) | 18/29 (62.1%) | 28/29 (96.6%) | 28/34 (82.4%) |

“Given detected” measures ranking quality only where the detector supplied an
event to explain. “End-to-end” counts an undetected interval as a miss. Both are
needed: the first evaluates prioritization; the second reflects the full user
workflow.

## Representative cases

Successful case — `phase2_040`, simulated pH-sensor fault:

- The detector produced anomaly events within the known interval.
- pH-probe review was the top batch-level recommendation and an accepted Top-1
  result for the mapped interval.
- The report shows the pH feature values, trends, timestamps, SHAP effects, and
  suggested calibration/maintenance checks.

Ambiguous case — `phase2_034`, early simulated pressure fault:

- The interval was detected, but pressure regulation was outside the Top 3.
- DO, cooling/heating, and acid/base evidence ranked higher, illustrating that
  downstream correlated behavior can obscure the manipulated process area.
- A later pressure interval in the same batch did prioritize pressure review.

Missed case — `phase2_039`, simulated temperature-sensor fault:

- The frozen detector produced zero alerts, so SHAP and ranking correctly
  produced no event-level diagnosis.
- This is a detector limitation, not an explanation-layer failure.

Healthy case — `phase2_011`:

- The batch produced 30 flagged windows grouped into 8 events.
- This demonstrates the known false-alert burden and why every report warns
  that an alert is a review lead, not contamination evidence.

## Scientific limits

- The data contain simulated industrial process and sensor faults, not measured
  microbial contamination.
- SHAP explains model behavior, not physical causality.
- Investigation priorities are relative evidence scores, not calibrated
  probabilities.
- No operator event logs were available, so intervention timing could not be
  checked.
- The final Phase 3 figures are post-hoc: the Phase 2 test set had already been
  viewed, and a numerical scaling defect was corrected after an initial Phase 3
  run. A future study needs a newly locked campaign for unbiased confirmation.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate_phase3.py --force
.\.venv\Scripts\python.exe -m maverickx.cli investigate "data\raw\phase2_indpensim\csv\Batch_40_fault_8.csv" --output-dir "outputs\phase3\demo"
```

Measured metrics are stored in `outputs/phase3/investigation_benchmark.json`.
The individual machine-readable, Markdown, CSV, and PNG reports are under
`outputs/phase3/evaluation_batches/` and `outputs/phase3/examples/`.
