"""Tune candidate detectors on development data, then open the locked test set once."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from maverickx.data import BatchData, load_indpensim_csv, load_indpensim_dataset
from maverickx.features import FeatureConfig
from maverickx.metrics import (
    score_batches,
    selection_key,
    summarize_metrics,
    summarize_scored_metrics,
)
from maverickx.model import make_stratified_batch_split, train_isolation_forest
from maverickx.phase2 import CandidateDetector, fit_candidate
from maverickx.schema import EXTENDED_MODEL_SENSORS, INITIAL_MODEL_SENSORS


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "data" / "raw" / "phase2_indpensim" / "csv"
OUTPUT_DIR = ROOT / "outputs" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts" / "phase2"
MANIFEST_PATH = ROOT / "configs" / "phase2_split_manifest.json"
FALSE_EVENT_CAP_PER_BATCH_DAY = 0.5
THRESHOLD_QUANTILES = (0.99, 0.9925, 0.995, 0.9975, 0.999, 0.9995)


def load_generated(numbers: list[int]) -> list[BatchData]:
    """Load only the explicitly requested partition, not the whole directory."""

    paths: dict[int, Path] = {}
    for path in GENERATED_DIR.glob("Batch_*.csv"):
        batch_number = int(path.name.split("_", 2)[1])
        paths[batch_number] = path
    missing = sorted(set(numbers) - set(paths))
    if missing:
        raise FileNotFoundError(f"Missing generated batches: {missing}")
    return [
        load_indpensim_csv(paths[number], batch_id_prefix="phase2") for number in numbers
    ]


def flat_row(detector, quantile: float, metrics: dict, sensor_set: str) -> dict:
    aggregate = metrics["aggregate"]
    return {
        "model_version": detector.model_version,
        "algorithm": detector.algorithm,
        "sensor_set": sensor_set,
        "window_samples": detector.feature_config.window_samples,
        "window_hours": detector.feature_config.approximate_window_hours,
        "threshold_quantile": quantile,
        "threshold": detector.alert_threshold,
        **aggregate,
    }


def choose_configuration(rows: list[dict]) -> dict:
    eligible = [
        row
        for row in rows
        if row["healthy_false_alert_events_per_batch_day"] <= FALSE_EVENT_CAP_PER_BATCH_DAY
    ]
    pool = eligible or rows
    return max(pool, key=lambda row: selection_key({"aggregate": row}))


def plot_timeline(detector, batch: BatchData, destination: Path) -> None:
    timeline, _ = detector.score_batch(batch)
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(timeline["window_end_h"], timeline["anomaly_score"], linewidth=1.0)
    axis.axhline(detector.alert_threshold, color="tab:red", linestyle="--", label="alert threshold")
    flagged = timeline[timeline["is_flagged"]]
    axis.scatter(
        flagged["window_end_h"],
        flagged["anomaly_score"],
        color="tab:red",
        s=10,
        label="flagged window",
    )
    for start, end in batch.metadata.fault_intervals_h:
        axis.axvspan(start, end, color="tab:orange", alpha=0.16)
    axis.set(
        title=f"{batch.metadata.batch_id}: {batch.metadata.fault_label or 'healthy'}",
        xlabel="Elapsed batch time (hours)",
        ylabel="Anomaly score",
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    timeline.to_csv(destination.with_suffix(".csv"), index=False)


def interval_diagnostics(detector, batches: list[BatchData]) -> list[dict]:
    """Describe score margin and strongest measured deviation for each event."""

    results: list[dict] = []
    for batch in batches:
        timeline, normalized = detector.score_batch(batch)
        for start, end in batch.metadata.fault_intervals_h:
            mask = timeline["window_end_h"].between(start, end, inclusive="both")
            region = timeline.loc[mask]
            if region.empty:
                continue
            peak_index = int(region["anomaly_score"].idxmax())
            peak_features = normalized.loc[peak_index].abs().sort_values(ascending=False)
            first_alert = region.loc[region["is_flagged"], "window_end_h"]
            results.append(
                {
                    "batch_id": batch.metadata.batch_id,
                    "fault_label": batch.metadata.fault_label,
                    "start_h": start,
                    "end_h": end,
                    "detected": not first_alert.empty,
                    "detection_delay_h": (
                        None if first_alert.empty else float(first_alert.iloc[0] - start)
                    ),
                    "peak_score": float(region["anomaly_score"].max()),
                    "threshold": float(detector.alert_threshold),
                    "peak_score_margin_above_threshold": float(
                        region["anomaly_score"].max() - detector.alert_threshold
                    ),
                    "largest_normalized_feature_at_peak": str(peak_features.index[0]),
                    "largest_absolute_normalized_value": float(peak_features.iloc[0]),
                }
            )
    return results


def render_locked_outputs() -> None:
    """Render charts from the already-selected artifact without refitting."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected = CandidateDetector.load(ARTIFACT_DIR / "selected_detector.joblib")
    test_healthy = load_generated(manifest["untouched_test"]["healthy_batches"])
    test_faults = load_generated(manifest["untouched_test"]["fault_batches"])
    plot_timeline(selected, test_healthy[0], OUTPUT_DIR / "healthy_test_timeline.png")
    coolant = next(batch for batch in test_faults if batch.metadata.fault_label == "coolant_flow_fault")
    plot_timeline(selected, coolant, OUTPUT_DIR / "coolant_fault_test_timeline.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render charts from the locked model without training or model selection.",
    )
    args = parser.parse_args()
    if args.render_only:
        render_locked_outputs()
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    published = load_indpensim_dataset(
        ROOT / "data" / "raw" / "indpensim" / "batches.parquet",
        ROOT / "data" / "raw" / "indpensim" / "statistics.parquet",
    )
    published_healthy = [batch for batch in published if batch.metadata.is_fault_batch is False]
    published_faults = [batch for batch in published if batch.metadata.is_fault_batch is True]

    # Up to this point the locked test CSVs have not been opened.
    development_healthy = load_generated(manifest["development"]["healthy_batches"])
    development_faults = load_generated(manifest["development"]["fault_batches"])
    validation_faults = [*published_faults, *development_faults]

    rows: list[dict] = []
    fitted: dict[tuple[str, str, int], object] = {}
    metrics_by_key: dict[tuple[str, str, int, float], dict] = {}
    sensor_sets = {
        "core": INITIAL_MODEL_SENSORS,
        "extended": EXTENDED_MODEL_SENSORS,
    }
    for sensor_set, sensors in sensor_sets.items():
        for window_samples in (3, 6, 12, 18):
            config = FeatureConfig(window_samples=window_samples, sensors=sensors)
            for algorithm in ("isolation_forest", "one_class_svm", "robust_z_score"):
                print(f"Fitting {algorithm}, {sensor_set}, {window_samples} samples", flush=True)
                detector = fit_candidate(
                    published_healthy,
                    algorithm=algorithm,
                    feature_config=config,
                    random_seed=42,
                )
                key = (algorithm, sensor_set, window_samples)
                fitted[key] = detector
                healthy_scored = score_batches(detector, development_healthy)
                fault_scored = score_batches(detector, validation_faults)
                healthy_scores = pd.concat(
                    [item.timeline["anomaly_score"] for item in healthy_scored],
                    ignore_index=True,
                )
                thresholds = {
                    quantile: float(healthy_scores.quantile(quantile))
                    for quantile in THRESHOLD_QUANTILES
                }
                for quantile, threshold in thresholds.items():
                    detector.with_threshold(threshold, quantile)
                    metrics = summarize_scored_metrics(
                        detector.model_version,
                        healthy_scored,
                        fault_scored,
                        threshold,
                    )
                    metrics_by_key[(*key, quantile)] = metrics
                    rows.append(flat_row(detector, quantile, metrics, sensor_set))

    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "fault_event_recall",
            "healthy_false_alert_events_per_batch_day",
            "median_detection_delay_h",
        ],
        ascending=[False, True, True],
    )
    leaderboard.to_csv(OUTPUT_DIR / "development_leaderboard.csv", index=False)
    selected_row = choose_configuration(rows)
    selected_key = (
        selected_row["algorithm"],
        selected_row["sensor_set"],
        int(selected_row["window_samples"]),
    )
    selected = fitted[selected_key].with_threshold(
        selected_row["threshold"], selected_row["threshold_quantile"]
    )
    selected.model_version = "maverickx_phase2_selected"
    selected.save(ARTIFACT_DIR / "selected_detector.joblib")
    selected_development_metrics = metrics_by_key[
        (*selected_key, selected_row["threshold_quantile"])
    ]
    selection_lock = {
        "false_alert_cap_events_per_batch_day": FALSE_EVENT_CAP_PER_BATCH_DAY,
        "selected_configuration": selected_row,
        "development_metrics": selected_development_metrics,
        "test_data_opened_before_lock": False,
    }
    (OUTPUT_DIR / "selection_lock.json").write_text(
        json.dumps(selection_lock, indent=2), encoding="utf-8"
    )
    print("Selection locked. Opening untouched test partition now.", flush=True)

    # Independent evaluation begins here, after the artifact and configuration
    # were saved. Results below must not be used to retune this model.
    test_healthy = load_generated(manifest["untouched_test"]["healthy_batches"])
    test_faults = load_generated(manifest["untouched_test"]["fault_batches"])
    selected_test_metrics = summarize_metrics(selected, test_healthy, test_faults)

    original_split = make_stratified_batch_split(published, random_seed=42)
    original = train_isolation_forest(published, original_split)
    original_test_metrics = summarize_metrics(original, test_healthy, test_faults)
    comparison = {
        "evaluation_type": "single independent evaluation on locked fresh simulator batches",
        "selected_phase2": selected_test_metrics,
        "original_v1": original_test_metrics,
        "delta_selected_minus_original": {
            name: selected_test_metrics["aggregate"][name]
            - original_test_metrics["aggregate"][name]
            for name in selected_test_metrics["aggregate"]
            if isinstance(selected_test_metrics["aggregate"][name], (int, float))
            and selected_test_metrics["aggregate"][name] is not None
            and original_test_metrics["aggregate"][name] is not None
        },
        "no_test_set_retuning": True,
    }
    (OUTPUT_DIR / "independent_test_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    diagnostics = {
        "purpose": "Post-test explanation of detected and missed simulated fault intervals; not used for retuning",
        "selected_phase2": interval_diagnostics(selected, test_faults),
        "original_v1": interval_diagnostics(original, test_faults),
    }
    (OUTPUT_DIR / "test_interval_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    plot_timeline(selected, test_healthy[0], OUTPUT_DIR / "healthy_test_timeline.png")
    coolant = next(batch for batch in test_faults if batch.metadata.fault_label == "coolant_flow_fault")
    plot_timeline(selected, coolant, OUTPUT_DIR / "coolant_fault_test_timeline.png")
    print(json.dumps(comparison["delta_selected_minus_original"], indent=2), flush=True)


if __name__ == "__main__":
    main()
