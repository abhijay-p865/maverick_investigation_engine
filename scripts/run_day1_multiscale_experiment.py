"""Day 1: compare v2.1 with a short-plus-long-window detector on development data.

This script deliberately does not load the Phase 2 test partition or overwrite a
production model artifact. Fault labels are used only after scoring to evaluate
the fixed candidates and their healthy-derived thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from maverickx.data import BatchData, load_indpensim_csv, load_indpensim_dataset
from maverickx.features import FeatureConfig
from maverickx.metrics import (
    ScoredBatch,
    score_batches,
    selection_key,
    summarize_scored_metrics,
)
from maverickx.phase2 import CandidateDetector, fit_candidate
from maverickx.schema import INITIAL_MODEL_SENSORS


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "data" / "raw" / "phase2_indpensim" / "csv"
MANIFEST_PATH = ROOT / "configs" / "phase2_split_manifest.json"
BASELINE_PATH = ROOT / "artifacts" / "phase2" / "selected_detector_v2_1.joblib"
OUTPUT_DIR = ROOT / "outputs" / "day1_multiscale"
FALSE_EVENT_CAP = 0.5
LONG_WINDOWS = (12, 18)
THRESHOLD_QUANTILES = (0.975, 0.98, 0.9845, 0.9875, 0.99, 0.9925, 0.995, 0.9975)


def load_generated(numbers: Iterable[int]) -> list[BatchData]:
    paths = {
        int(path.name.split("_", 2)[1]): path
        for path in GENERATED_DIR.glob("Batch_*.csv")
    }
    requested = list(numbers)
    missing = sorted(set(requested) - set(paths))
    if missing:
        raise FileNotFoundError(f"Missing generated batches: {missing}")
    return [
        load_indpensim_csv(paths[number], batch_id_prefix="phase2")
        for number in requested
    ]


def _raw_scores(detector: CandidateDetector, batches: Iterable[BatchData]) -> np.ndarray:
    return np.concatenate(
        [detector.score_batch(batch)[0]["anomaly_score"].to_numpy() for batch in batches]
    )


def _healthy_percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores to their empirical percentile in healthy development data."""

    reference = np.sort(np.asarray(reference, dtype=float))
    if reference.size == 0:
        raise ValueError("Healthy score reference cannot be empty.")
    ranks = np.searchsorted(reference, np.asarray(scores, dtype=float), side="right")
    return ranks.astype(float) / float(reference.size)


@dataclass
class MultiScaleDetector:
    """Combine short and long SVM evidence on a common healthy percentile scale."""

    short_detector: CandidateDetector
    long_detector: CandidateDetector
    short_reference: np.ndarray
    long_reference: np.ndarray
    alert_threshold: float = float("inf")
    model_version: str = "maverickx_multiscale_development_candidate"

    def score_batch(self, batch: BatchData) -> tuple[pd.DataFrame, pd.DataFrame]:
        short_timeline, short_normalized = self.short_detector.score_batch(batch)
        long_timeline, _ = self.long_detector.score_batch(batch)

        short_score = _healthy_percentile(
            short_timeline["anomaly_score"].to_numpy(), self.short_reference
        )
        long_score = _healthy_percentile(
            long_timeline["anomaly_score"].to_numpy(), self.long_reference
        )
        long_by_time = pd.Series(
            long_score,
            index=long_timeline["window_end_h"].round(8),
        )
        aligned_long = (
            short_timeline["window_end_h"].round(8).map(long_by_time).fillna(0.0).to_numpy()
        )

        timeline = short_timeline.copy()
        timeline["anomaly_score"] = np.maximum(short_score, aligned_long)
        timeline["alert_threshold"] = float(self.alert_threshold)
        timeline["is_flagged"] = timeline["anomaly_score"] > float(self.alert_threshold)
        timeline["model_version"] = self.model_version
        return timeline, short_normalized


def _aggregate_row(
    *,
    name: str,
    long_window_samples: int | None,
    threshold_quantile: float,
    threshold: float,
    metrics: dict,
) -> dict:
    return {
        "candidate": name,
        "long_window_samples": long_window_samples,
        "long_window_hours": (
            None if long_window_samples is None else long_window_samples * 0.2
        ),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        **metrics["aggregate"],
    }


def _category_delta(candidate: dict, baseline: dict) -> dict[str, dict[str, float | int]]:
    labels = sorted(set(candidate) | set(baseline))
    output: dict[str, dict[str, float | int]] = {}
    for label in labels:
        candidate_row = candidate.get(label, {})
        baseline_row = baseline.get(label, {})
        candidate_detected = int(candidate_row.get("detected_intervals", 0))
        baseline_detected = int(baseline_row.get("detected_intervals", 0))
        output[label] = {
            "baseline_detected": baseline_detected,
            "candidate_detected": candidate_detected,
            "detected_interval_delta": candidate_detected - baseline_detected,
            "fault_intervals": int(
                candidate_row.get("fault_intervals", baseline_row.get("fault_intervals", 0))
            ),
        }
    return output


def _score_candidate(
    detector: MultiScaleDetector,
    healthy_batches: list[BatchData],
    fault_batches: list[BatchData],
) -> tuple[list[ScoredBatch], list[ScoredBatch], list[dict], dict[float, dict]]:
    healthy_scored = score_batches(detector, healthy_batches)
    fault_scored = score_batches(detector, fault_batches)
    healthy_scores = pd.concat(
        [item.timeline["anomaly_score"] for item in healthy_scored], ignore_index=True
    )
    rows: list[dict] = []
    metrics_by_quantile: dict[float, dict] = {}
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(healthy_scores.quantile(quantile))
        metrics = summarize_scored_metrics(
            detector.model_version,
            healthy_scored,
            fault_scored,
            threshold,
        )
        metrics_by_quantile[quantile] = metrics
        rows.append(
            _aggregate_row(
                name=detector.model_version,
                long_window_samples=detector.long_detector.feature_config.window_samples,
                threshold_quantile=quantile,
                threshold=threshold,
                metrics=metrics,
            )
        )
    return healthy_scored, fault_scored, rows, metrics_by_quantile


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    published = load_indpensim_dataset(
        ROOT / "data" / "raw" / "indpensim" / "batches.parquet",
        ROOT / "data" / "raw" / "indpensim" / "statistics.parquet",
    )
    training_healthy = [
        batch for batch in published if batch.metadata.is_fault_batch is False
    ]
    published_faults = [
        batch for batch in published if batch.metadata.is_fault_batch is True
    ]
    development_healthy = load_generated(manifest["development"]["healthy_batches"])
    development_faults = load_generated(manifest["development"]["fault_batches"])
    evaluation_faults = [*published_faults, *development_faults]

    baseline = CandidateDetector.load(BASELINE_PATH)
    baseline_healthy_scored = score_batches(baseline, development_healthy)
    baseline_fault_scored = score_batches(baseline, evaluation_faults)
    baseline_metrics = summarize_scored_metrics(
        baseline.model_version,
        baseline_healthy_scored,
        baseline_fault_scored,
        baseline.alert_threshold,
    )
    baseline_row = _aggregate_row(
        name="v2.1 baseline",
        long_window_samples=None,
        threshold_quantile=baseline.threshold_quantile,
        threshold=baseline.alert_threshold,
        metrics=baseline_metrics,
    )

    short_reference = _raw_scores(baseline, development_healthy)
    candidate_rows: list[dict] = []
    candidate_metrics: dict[tuple[int, float], dict] = {}
    for long_window in LONG_WINDOWS:
        print(f"Fitting {long_window}-sample long-window SVM", flush=True)
        long_detector = fit_candidate(
            training_healthy,
            algorithm="one_class_svm",
            feature_config=FeatureConfig(
                window_samples=long_window,
                sensors=INITIAL_MODEL_SENSORS,
            ),
            random_seed=42,
        )
        long_reference = _raw_scores(long_detector, development_healthy)
        combined = MultiScaleDetector(
            short_detector=baseline,
            long_detector=long_detector,
            short_reference=short_reference,
            long_reference=long_reference,
            model_version=f"multiscale_3_plus_{long_window}_sample_svm",
        )
        _, _, rows, metrics_by_quantile = _score_candidate(
            combined, development_healthy, evaluation_faults
        )
        candidate_rows.extend(rows)
        candidate_metrics.update(
            {(long_window, quantile): metrics for quantile, metrics in metrics_by_quantile.items()}
        )

    eligible = [
        row
        for row in candidate_rows
        if row["healthy_false_alert_events_per_batch_day"] <= FALSE_EVENT_CAP
    ]
    selection_pool = eligible or candidate_rows
    selected_row = max(selection_pool, key=lambda row: selection_key({"aggregate": row}))
    selected_key = (
        int(selected_row["long_window_samples"]),
        float(selected_row["threshold_quantile"]),
    )
    selected_metrics = candidate_metrics[selected_key]

    baseline_key = selection_key(baseline_metrics)
    candidate_key = selection_key(selected_metrics)
    recommendation = (
        "advance_candidate_to_day2_validation"
        if selected_row["healthy_false_alert_events_per_batch_day"] <= FALSE_EVENT_CAP
        and candidate_key > baseline_key
        else "keep_v2_1_and_reject_multiscale_candidate"
    )

    leaderboard = pd.DataFrame([baseline_row, *candidate_rows]).sort_values(
        [
            "fault_event_recall",
            "healthy_false_alert_events_per_batch_day",
            "median_detection_delay_h",
        ],
        ascending=[False, True, True],
    )
    leaderboard.to_csv(OUTPUT_DIR / "development_leaderboard.csv", index=False)

    report = {
        "experiment": "day1_short_plus_long_window_detector",
        "date": "2026-10-26",
        "scope": "development data only; no production artifact changed",
        "test_partition_loaded": False,
        "fault_labels_used_as_model_features": False,
        "selection_policy": {
            "healthy_false_alert_event_cap_per_batch_day": FALSE_EVENT_CAP,
            "ranking": [
                "maximize fault-event recall",
                "minimize healthy false-alert events",
                "minimize median delay",
                "maximize window F2",
            ],
        },
        "baseline": baseline_row,
        "selected_candidate": selected_row,
        "delta_candidate_minus_baseline": {
            name: selected_row[name] - baseline_row[name]
            for name in (
                "fault_event_recall",
                "detected_fault_intervals",
                "healthy_false_alert_events_per_batch_day",
                "healthy_false_alert_window_fraction",
                "window_precision_during_known_fault_intervals",
                "window_recall_during_known_fault_intervals",
                "window_f2_during_known_fault_intervals",
            )
        },
        "fault_category_comparison": _category_delta(
            selected_metrics["by_fault_category"], baseline_metrics["by_fault_category"]
        ),
        "recommendation": recommendation,
        "scientific_limit": (
            "This is simulation-based development evidence, not an independent test result "
            "and not evidence of microbial-contamination detection."
        ),
    }
    (OUTPUT_DIR / "day1_decision.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
