# MaverickX Improvement Roadmap

This document is the handoff guide for the next teammate. It explains what the
prototype does today, what is still weak, and which improvements are most worth
building next. It assumes you understand the scientific goal but are still
learning software engineering and machine learning.

## 1. What exists now

The working pipeline is:

`batch CSV → data checks → rolling features → v2.1 One-Class SVM → anomaly events → Kernel SHAP → ranked investigations → JSON/Markdown/web report`

The model looks at dissolved oxygen, pH, temperature, and pressure. It compares
short rolling windows with healthy simulated batches at the same process time.
The model was trained only on healthy batches. Fault labels are used only after
analysis to measure performance.

Current measured performance on the existing simulated test set:

- The detector identified 29 of 34 simulated fault intervals: **85.3% event recall**.
- When a fault interval was detected, the acceptable investigation area was in
  the Top 3 for 28 of 29 intervals: **96.6% conditional Top-3 accuracy**.
- Counting undetected faults as failures, end-to-end Top-3 performance was 28 of
  34 intervals: **82.4%**.
- A healthy example still produced eight grouped anomaly events. False alerts
  therefore remain important.

These numbers are post-hoc simulator results. They are not proof of real-world
contamination detection or root-cause identification.

## 2. Highest-priority improvements

Work roughly in this order. Each step should leave the system runnable.

| Priority | Improvement | Why it matters | Approximate effort |
|---|---|---|---|
| 1 | Create a new locked simulator test campaign | The current test set has been viewed, so a new untouched set is needed for an unbiased result | 0.5–1 day |
| 2 | Reduce healthy false alerts | Scientists will lose trust if healthy batches create long investigation lists | 1–2 days |
| 3 | Improve the five missed fault intervals | Missed events cannot receive SHAP explanations or investigation rankings | 1–3 days |
| 4 | Improve pressure and coolant prioritization | Some detected events emphasize downstream symptoms instead of the manipulated system | 1–2 days |
| 5 | Add real process/event logs | Sampling, maintenance, alarms, and operator actions are essential investigation context | 2–5 days |
| 6 | Harden deployment, security, and storage | The current web server is a local demonstration, not a production service | 2–5 days |

## 3. Data and validation improvements

### 3.1 Generate a truly untouched test set

Before changing model settings, create a new batch campaign and write its batch
IDs into a split file. Do not inspect graphs, scores, or fault results from its
test portion until all choices are frozen.

Use three partitions:

1. **Training:** healthy batches used to fit the detector.
2. **Validation:** healthy and faulty batches used to choose features,
   thresholds, and ranking weights.
3. **Test:** opened exactly once for the final measurement.

Keep a file hash for every CSV. IndPenSim is stochastic, meaning a regenerated
batch may not contain exactly the same numbers.

### 3.2 Add more variation to healthy data

The current model can treat normal recipe-to-recipe variation as abnormal. Add
healthy batches covering:

- different starting conditions;
- expected batch lengths;
- control strategies;
- normal set-point adjustments;
- realistic sensor noise and drift; and
- ordinary sampling or feed events.

Do not simply add noisy copies of existing batches. The goal is to represent
real normal operating variability.

### 3.3 Make the CSV schema configurable

Real plants will use different tag names and units. Create a YAML or JSON
mapping file that translates plant tags into MaverickX names. Validate units
before analysis. For example, Celsius accidentally read as Kelvin could create
a very strong but meaningless anomaly.

Add checks for:

- duplicate or decreasing timestamps;
- unexpected sampling intervals;
- unit mismatches;
- long missing-data gaps;
- sensors stuck at one value;
- values outside physically possible ranges; and
- changes in batch recipe or control strategy.

## 4. Detector improvements

### 4.1 Investigate the five missed intervals

The frozen detector misses:

- two late substrate-feed intervals;
- two combined-fault intervals dominated by base-flow behavior; and
- one temperature-sensor fault batch.

For each miss, compare the raw sensor response against several healthy batches
at the same process phase. Ask whether the fault actually changed the four
model inputs. If it did not, lowering the alert threshold may only create more
false alarms. The better solution may be to add a relevant direct process
channel.

### 4.2 Evaluate an interpretable two-layer detector

Keep the existing four-sensor OCSVM as the general process detector. Add a
small set of direct-channel monitors for feed, coolant, aeration, and dosing.
These could be phase-adjusted robust limits rather than another complicated ML
model.

One possible design:

- Layer A: OCSVM identifies multivariable process behavior that differs from
  healthy history.
- Layer B: direct-channel rules identify unusual actuator commands or measured
  flows.
- Event grouping combines nearby evidence from both layers.

Evaluate the combined system by event recall, delay, and healthy false events.
Do not select it solely because its window-level accuracy looks better.

### 4.3 Use phase-specific or severity-specific thresholds

The distribution of normal scores changes during a long fermentation. A single
threshold may be too strict in one phase and too loose in another. Compare:

- one global threshold;
- thresholds for broad process phases; and
- thresholds based on the healthy score distribution at each phase.

Avoid extremely narrow time bins because they can memorize simulator timing.
Every threshold must be selected using validation batches only.

### 4.4 Improve event grouping

The current grouping bridges only short gaps. This can split one physical issue
into several report events. Test rules such as:

- requiring two of three adjacent windows to be anomalous;
- merging events separated by less than one or two hours when the same sensors
  dominate; and
- recording peak, start, recovery, and total abnormal duration separately.

Measure the effect on healthy event frequency as well as fault recall.

### 4.5 Compare carefully chosen alternatives

Reasonable models to compare include:

- Isolation Forest;
- Local Outlier Factor in novelty-detection mode;
- robust multivariate distance;
- PCA reconstruction error; and
- a small autoencoder only after enough healthy data exists.

Complex models are not automatically better. For a post-batch tool, stable
results and understandable evidence are more useful than a small benchmark
improvement that scientists cannot audit.

## 5. SHAP and explanation improvements

### 5.1 Test explanation stability

Kernel SHAP samples combinations of features. Run the same explanation with
several random seeds and slightly different healthy backgrounds. Report whether
the top sensors stay consistent. If the top explanation changes frequently,
mark that event as explanation-unstable.

### 5.2 Handle correlated features

Mean, slope, standard deviation, and change from the same sensor are strongly
related. SHAP may divide evidence among them in ways that are mathematically
valid but visually confusing. Keep feature-level values in JSON, but consider
showing sensor-level totals first in the scientist report.

### 5.3 Improve healthy background selection

The current background uses evenly sampled healthy windows. Compare this with:

- phase-stratified sampling;
- clustering healthy windows and selecting cluster representatives; and
- choosing background windows only from the same broad process phase as the
  event.

Track runtime and explanation stability. Always keep the SHAP additivity test:
`expected score + SHAP contributions ≈ actual model score`.

### 5.4 Add confidence language

Create an explanation-quality field based on:

- SHAP reconstruction error;
- stability across backgrounds/seeds;
- amount of positive evidence concentrated in the top sensor; and
- availability of direct process context.

Call this “explanation strength,” not root-cause confidence.

## 6. Investigation-ranking improvements

### 6.1 Tune rules only on validation data

The current ranking coefficients are transparent but preliminary. Put them in a
configuration file, freeze that file, then evaluate on untouched batches.

Optimize for:

- correct area in the Top 3;
- a useful Top-1 recommendation;
- short, non-repetitive lists; and
- few misleading recommendations on healthy batches.

### 6.2 Improve sensor-versus-process distinction

The difficult question is often whether a sensor is wrong or the process truly
changed. Add agreement checks such as:

- temperature versus coolant/heating demand;
- pH versus acid/base dosing;
- dissolved oxygen versus aeration, agitation, OUR, and off-gas oxygen; and
- pressure versus valve command or an independent pressure measurement.

If a measurement changes while all related evidence remains normal, prioritize
the probe. If both the measurement and actuator behavior change together,
prioritize the process system.

### 6.3 Add operator and process logs

Allow an optional second CSV containing events such as:

- sampling;
- manual additions;
- pump or valve alarms;
- maintenance;
- calibration;
- transfers; and
- operator notes.

Display events near each anomaly. A log entry is temporal context, not proof
that the action caused the anomaly.

### 6.4 Use a documented investigation knowledge base

Move investigation categories, evidence mappings, suggested checks, and
uncertainty statements into reviewed configuration. Record who approved each
rule and when it changed. This allows process experts to improve the logic
without editing Python.

## 7. Website and user-experience improvements

### 7.1 Support a clear upload template

Add a downloadable example CSV and a separate schema guide. Before analysis,
show detected columns, units, time range, sampling interval, and missing-data
warnings. Let the user correct a column mapping instead of immediately failing.

### 7.2 Improve report navigation

Useful additions include:

- click an event to zoom the timeline;
- hover over a score to see exact sensor readings;
- filter events by sensor or severity;
- compare the batch with a healthy reference band; and
- export a compact PDF for review meetings.

Do not hide limitations behind a tooltip. Keep the contamination and causality
warnings visible in the report.

### 7.3 Add job history carefully

The local prototype writes results to `outputs/web`. A shared deployment needs
a database for job metadata and protected object storage for CSVs and reports.
Users should be able to delete a batch and its outputs. Define a retention
period instead of keeping uploads forever.

## 8. Production architecture improvements

The current FastAPI server performs analysis inside the web request. That is
acceptable for a demonstration but fragile for multiple users.

A stronger architecture would use:

1. **Web API:** validates the upload and creates a job ID.
2. **Job queue:** stores pending analysis work.
3. **Worker:** loads the model and runs detection/SHAP outside the web request.
4. **Object storage:** holds uploaded CSVs and generated reports.
5. **Database:** stores job status, model version, file references, and audit
   information.
6. **Frontend:** polls job status and displays the completed report.

This prevents long analyses from blocking the server and makes failures
recoverable.

### 8.1 Version every analysis

Store these values with each report:

- model artifact hash;
- feature configuration;
- threshold and normalizer version;
- investigation-rule version;
- code commit SHA;
- input file hash; and
- analysis time.

This makes a report reproducible even after the system changes.

### 8.2 Add authentication and privacy controls

Before using real manufacturing data, add:

- sign-in and role-based access;
- encrypted transport and storage;
- upload size and file-type limits;
- malware scanning where required;
- automatic deletion/retention rules;
- audit logs; and
- a review of whether batch data may leave the plant network.

Never publish real batch data or trained artifacts containing proprietary data
in a public repository.

### 8.3 Package deployment explicitly

Add a Dockerfile after choosing the hosting environment. The model needs Python,
scikit-learn, SHAP, and several native numerical libraries, so a static-site
host alone is not enough. Measure memory use and cold-start time before choosing
a serverless platform.

## 9. Testing and engineering improvements

### 9.1 Continuous integration

The repository includes a GitHub Actions test workflow. Expand it to:

- run formatting and type checks;
- test Python versions intentionally supported;
- verify that the runtime model/background assets exist;
- run one real end-to-end upload; and
- publish test artifacts when a failure occurs.

### 9.2 Add regression fixtures

Keep a few small, licensed test batches with expected:

- event count;
- score checksum or tolerance;
- SHAP additivity;
- Top-3 recommendation; and
- JSON schema.

Do not assert every floating-point digit. Numerical libraries can differ
slightly across operating systems.

### 9.3 Add load and failure testing

Test:

- two simultaneous uploads;
- malformed and extremely wide CSVs;
- maximum-size files;
- server restart during analysis;
- missing model/background assets; and
- disk-full or report-write failures.

### 9.4 Define a stable JSON schema

Create a versioned JSON Schema file for the website/report contract. When fields
change, either keep backward compatibility or increase the schema version.
This lets the frontend and model pipeline evolve independently.

## 10. Scientific validation path

The current project does **not** detect microbial contamination. A defensible
contamination study would require real batches with:

- independently confirmed contamination status;
- organism and assay information where available;
- timestamps for first microbiological evidence;
- batch operations and interventions;
- equipment and maintenance records; and
- appropriate clean comparison batches.

Decide the intended claim before modeling. “Find unusual process periods after
a confirmed contamination” is much easier to validate than “detect
contamination automatically.” Use prospective validation and scientist review,
not only retrospective benchmark scores.

## 11. Suggested first week for the next teammate

Day 1:

- Run the full test suite and website.
- Read `PRD.md`, `docs/phase2_results.md`, and `docs/phase3_results.md`.
- Upload the included demo batch and inspect the JSON report.

Day 2:

- Generate and lock a new simulator development/test campaign.
- Add a data-quality preview to the website.

Day 3:

- Diagnose the missed substrate, combined, and temperature-sensor intervals.
- Prototype direct-channel rules without changing the test set.

Day 4:

- Improve sensor-versus-process ranking on development data.
- Put ranking weights and knowledge rules into configuration.

Day 5:

- Freeze choices, evaluate the untouched test set once, and update the report.
- Review claims and limitations with a scientist.

## 12. Commands to start

```powershell
# Install
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# Test
.\.venv\Scripts\python.exe -m pytest -q

# Run the website
.\.venv\Scripts\python.exe -m uvicorn maverickx.web:app --host 127.0.0.1 --port 8000

# Open http://127.0.0.1:8000 and upload:
# runtime_assets\demo\Batch_40_fault_8.csv
```

## 13. Important rules to preserve

- Never feed fault labels or known fault intervals into the model or ranking
  engine.
- Split datasets by complete batch, never individual rows.
- Freeze choices before opening a test set.
- Preserve the original v2.1 artifact when experimenting.
- Never describe an anomaly score as a contamination probability.
- Never describe SHAP as proof of physical causality.
- Report failed targets and negative results instead of hiding them.
- Keep the command-line and Python interfaces working even if the website
  changes.

The best next version is not necessarily the most complex one. It is the
version whose results a scientist can reproduce, understand, challenge, and
use to decide what to inspect next.
