"""Post-batch evaluation metrics centered on known fault events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from .data import BatchData
from .events import group_anomaly_events


class Detector(Protocol):
    model_version: str

    def score_batch(self, batch: BatchData) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@dataclass
class ScoredBatch:
    batch: BatchData
    timeline: pd.DataFrame
    normalized_features: pd.DataFrame


def evaluate_scored_batch(
    batch: BatchData,
    timeline: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    timeline = timeline.copy()
    if threshold is not None:
        timeline["alert_threshold"] = float(threshold)
        timeline["is_flagged"] = timeline["anomaly_score"] > float(threshold)
    predicted = timeline["is_flagged"].astype(bool)
    truth = pd.Series(False, index=timeline.index)
    intervals: list[dict[str, Any]] = []
    for start, end in batch.metadata.fault_intervals_h:
        in_interval = timeline["window_end_h"].between(start, end, inclusive="both")
        truth |= in_interval
        detected = timeline.loc[in_interval & predicted, "window_end_h"]
        intervals.append(
            {
                "start_h": float(start),
                "end_h": float(end),
                "detected": not detected.empty,
                "detection_delay_h": None if detected.empty else float(detected.iloc[0] - start),
            }
        )
    events = group_anomaly_events(timeline, normalized)
    return {
        "batch_id": batch.metadata.batch_id,
        "fault_label": batch.metadata.fault_label,
        "duration_h": float(batch.measurements["elapsed_time_h"].max()),
        "window_count": int(len(timeline)),
        "flagged_window_count": int(predicted.sum()),
        "alert_event_count": int(len(events)),
        "true_positive_windows": int((predicted & truth).sum()),
        "false_positive_windows": int((predicted & ~truth).sum()),
        "false_negative_windows": int((~predicted & truth).sum()),
        "fault_intervals": intervals,
    }


def evaluate_one_batch(detector: Detector, batch: BatchData) -> dict[str, Any]:
    timeline, normalized = detector.score_batch(batch)
    return evaluate_scored_batch(batch, timeline, normalized)


def score_batches(detector: Detector, batches: Iterable[BatchData]) -> list[ScoredBatch]:
    return [
        ScoredBatch(batch, *detector.score_batch(batch))
        for batch in batches
    ]


def summarize_metrics(
    detector: Detector,
    healthy_batches: Iterable[BatchData],
    fault_batches: Iterable[BatchData],
) -> dict[str, Any]:
    healthy = [evaluate_one_batch(detector, batch) for batch in healthy_batches]
    faulty = [evaluate_one_batch(detector, batch) for batch in fault_batches]
    return _summarize_rows(detector.model_version, healthy, faulty)


def summarize_scored_metrics(
    model_version: str,
    healthy_scored: Iterable[ScoredBatch],
    fault_scored: Iterable[ScoredBatch],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate many thresholds without repeating expensive model inference."""

    healthy = [
        evaluate_scored_batch(
            item.batch, item.timeline, item.normalized_features, threshold=threshold
        )
        for item in healthy_scored
    ]
    faulty = [
        evaluate_scored_batch(
            item.batch, item.timeline, item.normalized_features, threshold=threshold
        )
        for item in fault_scored
    ]
    return _summarize_rows(model_version, healthy, faulty)


def _summarize_rows(
    model_version: str,
    healthy: list[dict[str, Any]],
    faulty: list[dict[str, Any]],
) -> dict[str, Any]:
    healthy_windows = sum(row["window_count"] for row in healthy)
    healthy_flags = sum(row["flagged_window_count"] for row in healthy)
    healthy_days = sum(row["duration_h"] for row in healthy) / 24.0
    healthy_events = sum(row["alert_event_count"] for row in healthy)

    intervals = [item for row in faulty for item in row["fault_intervals"]]
    detected = [item for item in intervals if item["detected"]]
    delays = [float(item["detection_delay_h"]) for item in detected]
    tp = sum(row["true_positive_windows"] for row in faulty)
    fp = sum(row["false_positive_windows"] for row in faulty)
    fn = sum(row["false_negative_windows"] for row in faulty)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in faulty:
        categories[str(row["fault_label"])].extend(row["fault_intervals"])
    by_category = {
        label: {
            "detected_intervals": int(sum(item["detected"] for item in items)),
            "fault_intervals": len(items),
            "fault_event_recall": float(sum(item["detected"] for item in items) / len(items)),
        }
        for label, items in sorted(categories.items())
        if items
    }
    return {
        "model_version": model_version,
        "aggregate": {
            "fault_event_recall": len(detected) / len(intervals) if intervals else 0.0,
            "detected_fault_intervals": len(detected),
            "fault_intervals": len(intervals),
            "median_detection_delay_h": float(np.median(delays)) if delays else None,
            "mean_detection_delay_h": float(np.mean(delays)) if delays else None,
            "healthy_false_alert_events_per_batch_day": (
                healthy_events / healthy_days if healthy_days else 0.0
            ),
            "healthy_false_alert_window_fraction": (
                healthy_flags / healthy_windows if healthy_windows else 0.0
            ),
            "window_precision_during_known_fault_intervals": precision,
            "window_recall_during_known_fault_intervals": recall,
            "window_f2_during_known_fault_intervals": f2,
        },
        "by_fault_category": by_category,
        "healthy_batches": healthy,
        "fault_batches": faulty,
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank event recall first, then alert burden, delay, then window F2."""

    values = metrics["aggregate"]
    delay = values["median_detection_delay_h"]
    return (
        float(values["fault_event_recall"]),
        -float(values["healthy_false_alert_events_per_batch_day"]),
        -float(delay if delay is not None else 1e9),
        float(values["window_f2_during_known_fault_intervals"]),
    )
