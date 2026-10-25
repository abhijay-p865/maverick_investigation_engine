# Phase-2 independent evaluation results

## Outcome

The selected detector is a One-Class SVM with three-sample rolling windows,
the original four sensors (dissolved oxygen, pH, temperature, and pressure),
and the 99th percentile of fresh healthy development scores as its threshold.
It was selected under a limit of 0.5 false alert events per healthy batch-day.

The configuration was saved before the untouched test CSVs were opened. The
test set contained six healthy batches and sixteen fault batches, with two
batches for each simulator fault code 1–8.

## Selected model versus original v1

| Metric | Phase-2 selected | Original v1 | Change |
|---|---:|---:|---:|
| Known fault intervals detected | 27 / 34 | 25 / 34 | +2 intervals |
| Fault-event recall | 79.4% | 73.5% | +5.9 percentage points |
| Median detection delay, detected intervals | 0.0 h | 0.0 h | no change |
| Mean detection delay, detected intervals | 14.54 h | 12.16 h | 2.38 h worse |
| Healthy false alert events / batch-day | 0.330 | 0.487 | 32.1% lower |
| Healthy false-alert window fraction | 0.828% | 2.344% | 64.7% lower |
| Window precision in fault batches | 88.66% | 45.59% | +43.07 points |
| Window recall in known intervals | 13.86% | 3.56% | +10.30 points |
| Window F2 | 16.67% | 4.37% | +12.31 points |

The overall mean delay worsened because the selected model detected four events
that v1 missed, including two coolant events detected late. For the 23 intervals
both models detected, the selected model was 2.14 hours faster on average; the
median paired delay change was zero.

## Independent recall by simulated fault category

| Fault category | Phase-2 selected | Original v1 |
|---|---:|---:|
| Aeration | 4 / 4 | 3 / 4 |
| Pressure | 4 / 4 | 4 / 4 |
| Substrate feed | 4 / 6 | 4 / 6 |
| Base flow | 2 / 4 | 4 / 4 |
| Coolant flow | 2 / 2 | 0 / 2 |
| Combined process faults | 8 / 10 | 8 / 10 |
| Temperature sensor | 1 / 2 | 1 / 2 |
| pH sensor | 2 / 2 | 1 / 2 |

## Why coolant faults were missed before

The simulator forces coolant flow to 2 L/h during 70–90 h and to 10 L/h during
240–270 h. The generated batches generally end near 230 h, so the first interval
is the consistently evaluable event.

V1 did register the resulting temperature changes, but its peak scores in the
two fresh coolant test batches were only 0.00289 and 0.00024 below its alert
threshold. The faults were therefore near-threshold misses, not invisible
events. The shorter window and nonlinear One-Class SVM boundary pushed both over
threshold. They were detected 9.8 h and 9.0 h after fault onset, so coolant
detection is improved but still slow.

Adding direct manipulated-variable sensors was tested. The best extended-sensor
candidate reached 83.6% event recall on development data with 0.241 false alert
events per batch-day. The selected core model reached 85.5% recall with 0.336
events per batch-day, so the predeclared recall-first policy chose the core
model. This shows that more sensors did not automatically improve the overall
investigation result.

## Remaining misses

Seven test intervals were missed:

- both 200–214 h substrate-feed intervals;
- both 140–160 h base-flow intervals;
- the corresponding 140–160 h periods in both combined-fault batches;
- one long-running temperature-sensor-fault batch.

The missed substrate intervals showed moderate dissolved-oxygen slope or
variation changes, while missed base/combined intervals showed pH variation.
Their joint SVM scores remained below the healthy-derived threshold. Lowering
the threshold after seeing the test set would make the benchmark invalid, so no
post-test retuning was performed.

## Readiness recommendation

The model is ready for a research-stage explanation layer and investigation
prioritization prototype. It has a locked artifact, independent evidence, and
meaningfully fewer misleading alerts than v1. It is not ready as a production
quality decision system: 20.6% of known simulated intervals were missed,
coolant alerts were delayed, and no microbial-contamination or real-plant data
has been evaluated. SHAP and prioritization should therefore explain and rank
evidence, not claim root cause or contamination.

## Time-boxed threshold improvement

After the original independent results were reviewed, a development-only sweep
tested slightly lower thresholds for the same fitted SVM. No estimator,
features, preprocessing, or window definition changed. The 98.45th percentile
was the lowest selected threshold that maximized development event recall while
remaining under the existing 0.5 false-event-per-batch-day cap.

Development performance changed from 47/55 events (85.5%) at the original 99th
percentile to 50/55 (90.9%). Healthy false-event burden increased from 0.336 to
0.461 per batch-day.

Because the test results had already been viewed, the following comparison is
post-hoc confirmation, not a second untouched benchmark:

| Metric | v2.1 threshold | Original locked threshold | Change |
|---|---:|---:|---:|
| Fault-event recall | 29/34 (85.3%) | 27/34 (79.4%) | +5.9 points |
| Mean detection delay | 7.14 h | 14.54 h | 7.40 h lower |
| Healthy false events / batch-day | 0.470 | 0.330 | +0.139 |
| Window precision | 86.13% | 88.66% | -2.53 points |
| Window recall | 17.13% | 13.86% | +3.27 points |
| Window F2 | 20.40% | 16.67% | +3.72 points |

The lower threshold recovered both previously missed 140–160 h base-flow
events, at delays of 0.2 and 0.4 hours, without losing any previously detected
event. Across the 27 events detected by both versions, mean delay improved by
6.89 hours. The original artifact is preserved; v2.1 is saved separately and is
the recommended target for SHAP and investigation reporting.
