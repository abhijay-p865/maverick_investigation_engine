"""Small, honest development evaluation for the first detector."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import pandas as pd

from .data import BatchData
from .events import group_anomaly_events
from .model import IsolationForestDetector


def evaluate_batch(
    detector: IsolationForestDetector, batch: BatchData
) -> dict[str, Any]:
    timeline, normalized = detector.score_batch(batch)
    events = group_anomaly_events(timeline, normalized)
    predicted = timeline["is_flagged"].astype(bool)
    truth = pd.Series(False, index=timeline.index)
    interval_results = []
    for start, end in batch.metadata.fault_intervals_h:
        in_interval = (timeline["window_end_h"] >= start) & (
            timeline["window_end_h"] <= end
        )
        truth |= in_interval
        detected_times = timeline.loc[in_interval & predicted, "window_end_h"]
        interval_results.append(
            {
                "start_h": start,
                "end_h": end,
                "detected": not detected_times.empty,
                "detection_delay_h": (
                    None if detected_times.empty else float(detected_times.iloc[0] - start)
                ),
            }
        )
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    return {
        "batch_id": batch.metadata.batch_id,
        "fault_label": batch.metadata.fault_label,
        "batch_duration_h": float(batch.measurements["elapsed_time_h"].max()),
        "window_count": int(len(timeline)),
        "flagged_window_count": int(predicted.sum()),
        "event_count": len(events),
        "maximum_anomaly_score": float(timeline["anomaly_score"].max()),
        "true_positive_windows": true_positive,
        "false_positive_windows": false_positive,
        "false_negative_windows": false_negative,
        "fault_intervals": interval_results,
    }


def summarize_development_evaluation(
    detector: IsolationForestDetector,
    batches: Iterable[BatchData],
) -> dict[str, Any]:
    by_id = {batch.metadata.batch_id: batch for batch in batches}
    healthy_results = [evaluate_batch(detector, by_id[bid]) for bid in detector.split.healthy_test]
    fault_results = [evaluate_batch(detector, by_id[bid]) for bid in detector.split.fault_test]

    healthy_flags = sum(result["flagged_window_count"] for result in healthy_results)
    healthy_windows = sum(result["window_count"] for result in healthy_results)
    healthy_events = sum(result["event_count"] for result in healthy_results)
    healthy_days = sum(result["batch_duration_h"] for result in healthy_results) / 24.0

    tp = sum(result["true_positive_windows"] for result in fault_results)
    fp = sum(result["false_positive_windows"] for result in fault_results)
    fn = sum(result["false_negative_windows"] for result in fault_results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0

    interval_count = sum(len(result["fault_intervals"]) for result in fault_results)
    detected_intervals = sum(
        interval["detected"]
        for result in fault_results
        for interval in result["fault_intervals"]
    )
    detected_batches = sum(
        any(interval["detected"] for interval in result["fault_intervals"])
        for result in fault_results
    )

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in fault_results:
        categories[str(result["fault_label"])].append(result)
    by_category = {}
    for label, results in sorted(categories.items()):
        intervals = [interval for result in results for interval in result["fault_intervals"]]
        by_category[label] = {
            "batch_count": len(results),
            "detected_batch_count": sum(
                any(interval["detected"] for interval in result["fault_intervals"])
                for result in results
            ),
            "detected_interval_count": sum(interval["detected"] for interval in intervals),
            "fault_interval_count": len(intervals),
        }

    return {
        "evaluation_type": "development holdout (not a final untouched benchmark)",
        "model_version": detector.model_version,
        "aggregate": {
            "healthy_test_batch_count": len(healthy_results),
            "healthy_false_alert_window_fraction": healthy_flags / healthy_windows,
            "healthy_false_alert_events_per_batch_day": healthy_events / healthy_days,
            "fault_batch_count": len(fault_results),
            "fault_batches_with_in_interval_alert": detected_batches,
            "fault_interval_recall": detected_intervals / interval_count,
            "window_precision_during_known_fault_intervals": precision,
            "window_recall_during_known_fault_intervals": recall,
            "window_f2_during_known_fault_intervals": f2,
        },
        "by_fault_category": by_category,
        "healthy_batches": healthy_results,
        "fault_batches": fault_results,
        "limitations": [
            "The fault holdout was inspected while debugging the first pipeline, so it is not an untouched final test set.",
            "Only four simulated process-fault categories are present.",
            "A fault may cause delayed effects after its injected interval; interval-only metrics do not credit those alerts.",
            "No contamination examples are present.",
        ],
    }
