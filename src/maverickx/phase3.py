"""Complete phase-3 batch investigation workflow and report writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import BatchData
from .events import group_anomaly_events
from .explanations import build_healthy_background, explain_anomaly_events
from .investigation import (
    HealthyProcessReference,
    rank_batch_investigations,
    rank_event_investigations,
)
from .phase2 import CandidateDetector
from .presentation import measurement_display_name
from .reporting import plot_anomaly_timeline


LIMITATIONS = [
    "Evaluated on simulated process and sensor faults, not real manufacturing deviations.",
    "Flags show unusual sensor behavior relative to healthy simulator history; they do not estimate contamination risk or prove a physical cause.",
    "Priority scores order follow-up checks. They are not root-cause probabilities.",
    "Without process or operator logs, interventions cannot be confirmed by time. A flag may be benign, and a missed alert does not prove normal operation.",
]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _plot_event_importance(events: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for event in events:
        for item in event["explanation"]["sensor_contributions"]:
            rows.append(
                {
                    "event": f"Event {event['event_id']}",
                    "sensor": measurement_display_name(item["sensor"]),
                    "positive": item["mean_positive_contribution"],
                }
            )
    figure, axis = plt.subplots(figsize=(10, max(3.5, 0.45 * max(len(rows), 1))))
    if rows:
        frame = pd.DataFrame(rows)
        pivot = frame.pivot(index="event", columns="sensor", values="positive").fillna(0)
        pivot.plot(kind="barh", stacked=True, ax=axis)
        axis.set_xlabel("Average contribution that raised the unusual-behavior score")
        axis.set_ylabel("")
        axis.legend(title="Measurement", fontsize=8)
    else:
        axis.text(0.5, 0.5, "No anomaly events to explain", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Why each period looked unusual to the model")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _markdown(report: dict[str, Any]) -> str:
    summary = report["executive_summary"]
    lines = [
        f"# MaverickX post-batch investigation — {report['batch_metadata']['batch_id']}",
        "",
        "## Executive summary",
        "",
        summary["plain_language_summary"],
        "",
        f"- Model: `{report['model']['model_version']}` ({report['model']['model_type']})",
        f"- Eligible rolling windows: {summary['eligible_window_count']}",
        f"- Flagged windows: {summary['flagged_window_count']} ({summary['flagged_window_fraction']:.1%})",
        f"- Grouped anomaly events: {summary['anomaly_event_count']}",
        f"- Maximum anomaly score: {summary['maximum_anomaly_score']:.4f} (threshold {report['model']['alert_threshold']:.4f})",
        "",
        "A larger score means the pattern was more unusual compared with healthy simulator history. It is not a contamination probability.",
        "",
        "## Anomaly timeline",
        "",
        f"![Anomaly timeline]({Path(report['files']['timeline_chart_png']).name})",
        "",
    ]
    if not report["anomaly_events"]:
        lines += ["No grouped anomaly event crossed the saved v2.1 threshold.", ""]
    for event in report["anomaly_events"]:
        lines += [
            f"### Event {event['event_id']}: {event['start_h']:.1f}–{event['end_h']:.1f} h",
            "",
            f"Highest unusual-behavior score: **{event['maximum_anomaly_score']:.4f}** across {event['flagged_window_count']} flagged window(s).",
            "",
            event["explanation"]["plain_language_interpretation"],
            "",
            "Top model contributions:",
            "",
            "| Measurement pattern | What it means | Observed value | Effect on unusual-behavior score |",
            "|---|---|---:|---:|",
        ]
        for feature in event["explanation"]["top_features"][:5]:
            lines.append(
                f"| {feature['display_name']} | {feature['plain_language_meaning']} | "
                f"{feature['observed_feature_value']:.4g} {feature['unit']} | {feature['mean_shap_contribution']:+.4f} |"
            )
        lines += ["", "Representative sensor readings and trends:", ""]
        peak = event["explanation"]["explained_windows"][0]
        for reading in peak["sensor_readings"]:
            lines.append(
                f"- {reading['display_name']} was {reading['trend']} from {reading['start_value']:.4g} to "
                f"{reading['end_value']:.4g} {reading['unit']} between {reading['start_h']:.1f} and {reading['end_h']:.1f} h."
            )
        lines += ["", "Highest-priority checks for this event:", ""]
        for item in event["ranked_investigations"][:3]:
            lines.append(
                f"{item['rank']}. **{item['investigation_title']}** — evidence score {item['priority_score']:.3f}. "
                f"{item['possible_explanation']} {item['suggested_operator_verification']}"
            )
        lines += [""]
    lines += [
        "## Batch-level investigation priorities",
        "",
    ]
    if report["ranked_investigations"]:
        for item in report["ranked_investigations"][:6]:
            lines += [
                f"### {item['rank']}. {item['investigation_title']} — {item['priority_score']:.3f}",
                "",
                f"Supporting anomaly event(s): {', '.join(map(str, item['supporting_event_ids']))}.",
                "",
            ]
            for observation in item["confirmed_observations"][:4]:
                lines.append(f"- Confirmed observation: {observation}")
            lines += [
                f"- Possible explanation: {item['possible_explanation']}",
                f"- Recommended check: {item['suggested_operator_verification']}",
                f"- Uncertainty: {item['uncertainty']}",
                "",
            ]
    else:
        lines += ["No anomaly-driven investigation priority was generated.", ""]
    lines += [
        "## Explanation verification",
        "",
        f"Model-agnostic Kernel SHAP explained {report['shap_consistency']['explained_window_count']} representative windows. "
        f"Score reconstruction checks passed: **{report['shap_consistency']['all_additivity_checks_passed']}**; "
        f"maximum absolute residual {report['shap_consistency']['maximum_absolute_additivity_residual']:.2e}.",
        "",
        "## Limitations and uncertainty",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += [
        "",
        "## Output files",
        "",
        f"- Machine-readable analysis: `{Path(report['files']['analysis_json']).name}`",
        f"- Window-level scores: `{Path(report['files']['timeline_csv']).name}`",
        f"- Timeline chart: `{Path(report['files']['timeline_chart_png']).name}`",
        f"- Model-evidence chart: `{Path(report['files']['shap_chart_png']).name}`",
        "",
    ]
    return "\n".join(lines)


def analyze_batch_phase3(
    detector: CandidateDetector,
    batch: BatchData,
    healthy_background_batches: Iterable[BatchData],
    output_dir: str | Path,
    *,
    background_rows: int = 20,
    windows_per_event: int = 2,
    model_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Run v2.1 detection, SHAP, prioritization, and both report formats."""

    if detector.model_version != "maverickx_phase2_threshold_v2_1":
        raise ValueError("Phase 3 must use the saved v2.1 detector without retuning.")
    healthy_batches = list(healthy_background_batches)
    started = perf_counter()
    timeline, normalized = detector.score_batch(batch)
    events = group_anomaly_events(timeline, normalized)
    background = build_healthy_background(detector, healthy_batches, maximum_rows=background_rows)
    explained, shap_consistency = explain_anomaly_events(
        detector,
        batch,
        timeline,
        normalized,
        events,
        background,
        windows_per_event=windows_per_event,
    )
    process_reference = HealthyProcessReference.fit(healthy_batches)
    for event in explained:
        rankings, context = rank_event_investigations(event, batch, process_reference)
        event["ranked_investigations"] = rankings
        event["process_context"] = context
    batch_rankings = rank_batch_investigations(explained)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / batch.metadata.batch_id
    timeline_path = prefix.with_name(prefix.name + "_timeline.csv")
    json_path = prefix.with_name(prefix.name + "_investigation.json")
    markdown_path = prefix.with_name(prefix.name + "_investigation.md")
    timeline_chart = prefix.with_name(prefix.name + "_timeline.png")
    shap_chart = prefix.with_name(prefix.name + "_shap.png")
    timeline.to_csv(timeline_path, index=False)
    plot_anomaly_timeline(
        batch, timeline, timeline_chart, show_known_fault_intervals=False
    )
    _plot_event_importance(explained, shap_chart)

    flagged = int(timeline["is_flagged"].sum())
    event_count = len(explained)
    period_word = "period" if event_count == 1 else "periods"
    plain_summary = (
        f"The analysis grouped {flagged} flagged windows into {event_count} unusual {period_word}. "
        + (
            f"Start by reviewing {batch_rankings[0]['investigation_title']}, which has the strongest evidence in this batch."
            if batch_rankings
            else "No follow-up priority was produced."
        )
    )
    report: dict[str, Any] = {
        "analysis_metadata": {
            "report_schema_version": "1.0.0",
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_sha256": batch.metadata.input_sha256,
            "model_artifact_sha256": model_artifact_sha256,
        },
        "report_type": "MaverickX post-batch investigation prototype",
        "batch_metadata": batch.metadata.to_dict(),
        "data_quality": batch.quality.to_dict(),
        "model": {
            "model_version": detector.model_version,
            "model_type": "One-Class SVM",
            "score_quantity": "negative One-Class SVM score_samples",
            "score_interpretation": "Larger values are more unusual; not a probability.",
            "alert_threshold": float(detector.alert_threshold),
            "threshold_quantile": float(detector.threshold_quantile),
            "feature_config": detector.feature_config.to_dict(),
            "feature_names": detector.feature_names,
        },
        "label_use_statement": "Fault labels and known intervals were not used for scoring, model explanation, or investigation ranking.",
        "executive_summary": {
            "plain_language_summary": plain_summary,
            "eligible_window_count": int(len(timeline)),
            "flagged_window_count": flagged,
            "flagged_window_fraction": float(timeline["is_flagged"].mean()),
            "anomaly_event_count": event_count,
            "maximum_anomaly_score": float(timeline["anomaly_score"].max()),
        },
        "anomaly_events": explained,
        "ranked_investigations": batch_rankings,
        "timeline": timeline.to_dict(orient="records"),
        "shap_consistency": shap_consistency,
        "runtime_seconds": float(perf_counter() - started),
        "limitations": LIMITATIONS,
        "files": {
            "analysis_json": str(json_path),
            "markdown_report": str(markdown_path),
            "timeline_csv": str(timeline_path),
            "timeline_chart_png": str(timeline_chart),
            "shap_chart_png": str(shap_chart),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
