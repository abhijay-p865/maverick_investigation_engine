"""Write machine-readable timelines and a first diagnostic chart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import BatchData
from .events import group_anomaly_events
from .model import IsolationForestDetector


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def plot_anomaly_timeline(
    batch: BatchData,
    timeline: pd.DataFrame,
    path: str | Path,
    *,
    show_known_fault_intervals: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sensors = [
        ("dissolved_oxygen_mg_l", "Dissolved oxygen (mg/L)"),
        ("ph", "pH"),
        ("temperature_k", "Temperature (K)"),
        ("pressure_bar", "Pressure (bar)"),
    ]
    figure, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    axes[0].plot(timeline["window_end_h"], timeline["anomaly_score"], color="#275dad")
    axes[0].axhline(
        timeline["alert_threshold"].iloc[0], color="#b3261e", linestyle="--", label="Threshold"
    )
    flagged = timeline[timeline["is_flagged"]]
    axes[0].scatter(
        flagged["window_end_h"], flagged["anomaly_score"], color="#b3261e", s=12, label="Flagged"
    )
    axes[0].set_ylabel("Unusual-behavior score")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Unusual behavior across the batch — {batch.metadata.batch_id}")

    for axis, (sensor, label) in zip(axes[1:], sensors):
        axis.plot(
            batch.measurements["elapsed_time_h"],
            batch.measurements[sensor],
            color="#18794e",
            linewidth=1,
        )
        axis.set_ylabel(label)

    for axis in axes:
        if show_known_fault_intervals:
            for start, end in batch.metadata.fault_intervals_h:
                axis.axvspan(start, end, color="#f5a524", alpha=0.12)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Elapsed batch time (h)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def analyze_and_write(
    detector: IsolationForestDetector,
    batch: BatchData,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline, normalized = detector.score_batch(batch)
    events = group_anomaly_events(timeline, normalized)

    prefix = output_dir / batch.metadata.batch_id
    timeline_path = prefix.with_name(prefix.name + "_timeline.csv")
    json_path = prefix.with_name(prefix.name + "_analysis.json")
    chart_path = prefix.with_name(prefix.name + "_timeline.png")
    timeline.to_csv(timeline_path, index=False)
    plot_anomaly_timeline(batch, timeline, chart_path)

    report: dict[str, Any] = {
        "batch_metadata": batch.metadata.to_dict(),
        "data_quality": batch.quality.to_dict(),
        "model": {
            "model_version": detector.model_version,
            "model_type": "Isolation Forest",
            "score_interpretation": "Larger values are more unusual; not a probability.",
            "alert_threshold": detector.alert_threshold,
            "threshold_source": (
                f"{detector.threshold_quantile:.1%} quantile of healthy validation scores"
            ),
            "feature_config": detector.feature_config.to_dict(),
        },
        "summary": {
            "eligible_window_count": int(len(timeline)),
            "flagged_window_count": int(timeline["is_flagged"].sum()),
            "flagged_window_fraction": float(timeline["is_flagged"].mean()),
            "anomaly_event_count": len(events),
            "maximum_anomaly_score": float(timeline["anomaly_score"].max()),
        },
        "anomaly_events": events,
        "timeline": timeline.to_dict(orient="records"),
        "evaluation_ground_truth": {
            "fault_label": batch.metadata.fault_label,
            "known_fault_intervals_h": batch.metadata.fault_intervals_h,
        },
        "limitations": [
            "This result is based on simulated industrial process data.",
            "An anomaly is not evidence of microbial contamination.",
            "Affected-sensor labels are based on standardized feature deviation, not SHAP or root-cause proof.",
            "The first model has not yet been compared with alternative detectors.",
        ],
        "files": {
            "timeline_csv": str(timeline_path),
            "chart_png": str(chart_path),
        },
    }
    json_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    report["files"]["analysis_json"] = str(json_path)
    return report

