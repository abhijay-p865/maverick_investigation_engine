# Phase-2 model improvement method

This phase treats anomaly detection as a post-batch investigation aid. The most important question is whether a known simulated fault period produced at least one useful alert. A model is not rewarded merely for labeling many overlapping windows.

## Fresh data and independence

The locked campaign contains 48 newly simulated batches:

- 16 healthy batches;
- four independently simulated batches for each fault code 1 through 8;
- development data: healthy batches 1–10 and fault batches 17–32;
- untouched test data: healthy batches 11–16 and fault batches 33–48.

The split was written to `configs/phase2_split_manifest.json` before generation. The experiment code opens only development files during tuning, saves the selected configuration, and then opens the untouched test files once. Test results are not used for another tuning round.

## Fair comparison

Every candidate is fitted on the same 90 healthy batches from the published IndPenSim compilation. Candidates differ only in:

- detector: Isolation Forest, One-Class SVM, or a robust phase-adjusted z-score baseline;
- rolling window: 3, 6, 12, or 18 samples (36 minutes to 3.6 hours at 12-minute sampling);
- sensors: the original four-sensor set, or an extended set that also includes aeration, substrate feed, base flow, and coolant flow.

All candidates use mean, variation, slope, and total change features. Phase normalization compares a reading with healthy batches at the same elapsed fermentation time. Batch IDs, fault codes, fault labels, and fault timing are not model inputs.

Thresholds are healthy-score quantiles from the ten fresh healthy development batches. Fault labels are used only after scores exist, to compare candidate performance.

## Selection policy

Candidates must first stay at or below 0.5 false alert events per healthy batch-day when possible. Eligible candidates are ranked in this order:

1. higher known fault-event recall;
2. fewer false alert events on healthy batches;
3. shorter median delay from fault onset to first alert;
4. higher window-level F2 score.

The false-alert event count groups nearby flagged rolling windows into one investigation episode. Window precision and recall remain reported, but they are not the main optimization target.

## Scientific limits

Known intervals come from simulator code, not from measured contamination. An alert means that a period looks different from healthy simulator history. It does not establish a physical cause, microbial contamination, product impact, or real-time detection capability.
