"""SHAP explanations for the saved phase-2 One-Class SVM detector."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap

from .data import BatchData
from .features import extract_rolling_features
from .phase2 import CandidateDetector
from .presentation import (
    feature_display_name,
    feature_plain_language_meaning,
    measurement_display_name,
)


def _feature_unit(name: str, units: dict[str, str]) -> str:
    sensor, statistic = name.split("__", 1)
    base = units.get(sensor, "unknown")
    return f"{base}/h" if statistic == "slope_per_h" else base


def build_healthy_background(
    detector: CandidateDetector,
    healthy_batches: Iterable[BatchData],
    *,
    maximum_rows: int = 20,
) -> pd.DataFrame:
    """Build a deterministic, phase-normalized healthy SHAP background."""

    batches = list(healthy_batches)
    if not batches:
        raise ValueError("At least one healthy background batch is required.")
    if any(batch.metadata.is_fault_batch is not False for batch in batches):
        raise ValueError("SHAP background batches must be explicitly healthy.")
    normalized = []
    for batch in batches:
        features = extract_rolling_features(batch, detector.feature_config)
        normalized.append(detector.phase_normalizer.transform(features))
    combined = pd.concat(normalized, ignore_index=True)
    count = min(maximum_rows, len(combined))
    positions = np.linspace(0, len(combined) - 1, count, dtype=int)
    return combined.iloc[positions].reset_index(drop=True)


def _sensor_readings(
    batch: BatchData,
    *,
    start_h: float,
    end_h: float,
    sensors: Iterable[str],
) -> list[dict[str, Any]]:
    frame = batch.measurements
    window = frame.loc[
        frame["elapsed_time_h"].between(start_h - 1e-8, end_h + 1e-8)
    ]
    observations: list[dict[str, Any]] = []
    for sensor in sensors:
        if sensor not in window or window.empty:
            continue
        values = window[sensor].to_numpy(dtype=float)
        delta = float(values[-1] - values[0])
        direction = "rising" if delta > 1e-9 else "falling" if delta < -1e-9 else "stable"
        observations.append(
            {
                "sensor": sensor,
                "display_name": measurement_display_name(sensor),
                "unit": batch.metadata.units.get(sensor, "unknown"),
                "start_value": float(values[0]),
                "end_value": float(values[-1]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "change": delta,
                "trend": direction,
                "start_h": float(start_h),
                "end_h": float(end_h),
            }
        )
    return observations


def explain_anomaly_events(
    detector: CandidateDetector,
    batch: BatchData,
    timeline: pd.DataFrame,
    normalized_features: pd.DataFrame,
    events: list[dict[str, Any]],
    background: pd.DataFrame,
    *,
    windows_per_event: int = 2,
    maximum_evaluations: int | None = None,
    additivity_tolerance: float = 1e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Explain representative windows from every grouped anomaly event.

    SHAP receives the same phase-normalized features and predicts the exact
    public score returned by ``CandidateDetector``: negative OCSVM
    ``score_samples``. Therefore positive SHAP values push a window toward a
    larger, more anomalous score.
    """

    if detector.algorithm != "one_class_svm" or detector.estimator is None or detector.scaler is None:
        raise ValueError("Phase-3 explanations require the saved One-Class SVM detector.")
    if list(background.columns) != detector.feature_names:
        background = background.loc[:, detector.feature_names]
    raw_features = extract_rolling_features(batch, detector.feature_config)

    selected: list[tuple[int, int]] = []
    for event_index, event in enumerate(events):
        in_event = timeline.index[
            timeline["window_end_h"].between(event["start_h"], event["end_h"])
            & timeline["is_flagged"]
        ]
        top = timeline.loc[in_event].nlargest(windows_per_event, "anomaly_score").index
        selected.extend((event_index, int(position)) for position in top)
    if not selected:
        return [], {
            "method": "SHAP KernelExplainer (model-agnostic kernel SHAP)",
            "explained_window_count": 0,
            "score_direction": "larger values are more anomalous",
            "all_additivity_checks_passed": True,
            "maximum_absolute_additivity_residual": 0.0,
        }

    positions = [position for _, position in selected]
    inputs = normalized_features.iloc[positions].loc[:, detector.feature_names]

    def predict(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        named = pd.DataFrame(matrix, columns=detector.feature_names)
        scaled = detector.scaler.transform(named)
        return -detector.estimator.score_samples(scaled)

    explainer = shap.KernelExplainer(
        predict,
        background.to_numpy(dtype=float),
        feature_names=detector.feature_names,
        seed=42,
    )
    evaluations = maximum_evaluations or 100
    values_matrix = np.asarray(
        explainer.shap_values(
            inputs.to_numpy(dtype=float), nsamples=evaluations, silent=True
        ),
        dtype=float,
    )
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    event_windows: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(events))}
    residuals: list[float] = []
    for row_number, (event_index, position) in enumerate(selected):
        values = np.asarray(values_matrix[row_number], dtype=float)
        score = float(timeline.iloc[position]["anomaly_score"])
        reconstructed = float(base_value + values.sum())
        residual = score - reconstructed
        residuals.append(abs(residual))
        feature_details = []
        for feature_position, name in enumerate(detector.feature_names):
            sensor, statistic = name.split("__", 1)
            contribution = float(values[feature_position])
            feature_details.append(
                {
                    "feature": name,
                    "display_name": feature_display_name(name),
                    "plain_language_meaning": feature_plain_language_meaning(name),
                    "sensor": sensor,
                    "statistic": statistic,
                    "shap_contribution": contribution,
                    "effect_on_anomaly_score": "increases" if contribution > 0 else "decreases",
                    "normalized_value": float(inputs.iloc[row_number, feature_position]),
                    "observed_feature_value": float(raw_features.iloc[position][name]),
                    "unit": _feature_unit(name, batch.metadata.units),
                }
            )
        feature_details.sort(key=lambda item: abs(item["shap_contribution"]), reverse=True)
        start_h = float(timeline.iloc[position]["window_start_h"])
        end_h = float(timeline.iloc[position]["window_end_h"])
        event_windows[event_index].append(
            {
                "window_start_h": start_h,
                "window_end_h": end_h,
                "anomaly_score": score,
                "alert_threshold": float(detector.alert_threshold),
                "healthy_background_expected_score": base_value,
                "reconstructed_score": reconstructed,
                "additivity_residual": float(residual),
                "additivity_check_passed": bool(abs(residual) <= additivity_tolerance),
                "top_features": feature_details[:8],
                "sensor_readings": _sensor_readings(
                    batch,
                    start_h=start_h,
                    end_h=end_h,
                    sensors=detector.feature_config.sensors,
                ),
                "_all_feature_contributions": feature_details,
            }
        )

    explained_events: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        windows = event_windows[event_index]
        by_sensor: dict[str, dict[str, float]] = {}
        by_feature: dict[str, dict[str, Any]] = {}
        for window in windows:
            for detail in window.pop("_all_feature_contributions"):
                sensor = detail["sensor"]
                summary = by_sensor.setdefault(sensor, {"positive": 0.0, "negative": 0.0, "absolute": 0.0})
                value = detail["shap_contribution"]
                summary["positive"] += max(value, 0.0)
                summary["negative"] += min(value, 0.0)
                summary["absolute"] += abs(value)
                aggregate = by_feature.setdefault(detail["feature"], {**detail, "values": []})
                aggregate["values"].append(value)
        divisor = max(len(windows), 1)
        sensor_summary = [
            {
                "sensor": sensor,
                "display_name": measurement_display_name(sensor),
                "mean_positive_contribution": values["positive"] / divisor,
                "mean_negative_contribution": values["negative"] / divisor,
                "mean_absolute_contribution": values["absolute"] / divisor,
            }
            for sensor, values in by_sensor.items()
        ]
        sensor_summary.sort(key=lambda item: item["mean_positive_contribution"], reverse=True)
        aggregate_features = []
        for item in by_feature.values():
            values = item.pop("values")
            item["mean_shap_contribution"] = float(np.mean(values))
            item["mean_absolute_shap_contribution"] = float(np.mean(np.abs(values)))
            aggregate_features.append(item)
        aggregate_features.sort(
            key=lambda item: item["mean_absolute_shap_contribution"], reverse=True
        )
        positive_sensors = [item for item in sensor_summary if item["mean_positive_contribution"] > 0]
        names = [item["display_name"] for item in positive_sensors[:2]]
        interpretation = (
            "The model score was pushed upward most by " + ", ".join(names) + "."
            if names
            else "No sensor had a net positive contribution in the selected windows."
        )
        explained_events.append(
            {
                **event,
                "explanation": {
                    "score_quantity": "negative One-Class SVM score_samples",
                    "score_direction": "positive SHAP values increase the anomaly score",
                    "plain_language_interpretation": interpretation,
                    "sensor_contributions": sensor_summary,
                    "top_features": aggregate_features[:8],
                    "explained_windows": windows,
                },
            }
        )

    consistency = {
        "method": "SHAP KernelExplainer (model-agnostic kernel SHAP)",
        "background_type": "representative healthy phase-normalized rolling windows",
        "background_window_count": int(len(background)),
        "explained_window_count": int(len(selected)),
        "model_score_quantity": "negative One-Class SVM score_samples",
        "score_direction": "larger values are more anomalous; positive SHAP raises the score",
        "coalition_samples_per_window": int(evaluations),
        "additivity_tolerance": float(additivity_tolerance),
        "maximum_absolute_additivity_residual": float(max(residuals, default=0.0)),
        "all_additivity_checks_passed": bool(all(value <= additivity_tolerance for value in residuals)),
    }
    return explained_events, consistency
