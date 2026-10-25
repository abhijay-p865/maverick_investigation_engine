import json
from pathlib import Path

import numpy as np
import pandas as pd

from maverickx.data import BatchData, BatchMetadata, DataQualityReport
from maverickx.features import FeatureConfig
from maverickx.investigation import rank_event_investigations
from maverickx.investigation_evaluation import evaluate_investigation_reports
from maverickx.phase2 import fit_candidate
from maverickx.phase3 import analyze_batch_phase3


def make_batch(batch_id: str, *, faulty: bool, offset: float = 0.0) -> BatchData:
    rows = 60
    time = np.arange(rows) * 0.2
    frame = pd.DataFrame(
        {
            "batch_id": batch_id,
            "elapsed_time_h": time,
            "dissolved_oxygen_mg_l": 15 + np.sin(time) + offset,
            "ph": 6.5 + 0.01 * np.cos(time),
            "temperature_k": 298 + 0.02 * np.sin(time) + offset,
            "pressure_bar": 0.6 + 0.01 * np.cos(time),
            "aeration_lph": 8 + 0 * time,
            "agitation_rpm": 100 + 0 * time,
            "substrate_feed_lph": 1 + 0 * time,
            "acid_flow_lph": 0.1 + 0 * time,
            "base_flow_lph": 0.2 + 0 * time,
            "coolant_flow_lph": 2 + (8 if faulty else 0) + 0 * time,
            "heating_flow_lph": 0.1 + 0 * time,
        }
    )
    metadata = BatchMetadata(
        batch_id=batch_id,
        source_batch_number=None,
        source="fixture",
        source_version="1",
        control_strategy="recipe_driven",
        is_fault_batch=faulty,
        fault_label="coolant_flow_fault" if faulty else None,
        fault_onset_h=0.0 if faulty else None,
        fault_intervals_h=[(0.0, 12.0)] if faulty else [],
        units={name: "unit" for name in frame.columns},
    )
    quality = DataQualityReport(rows, {}, {}, 0, 0.2, 0)
    return BatchData(frame, metadata, quality)


def test_complete_report_uses_exact_score_and_writes_both_formats(tmp_path: Path) -> None:
    healthy = [make_batch(f"healthy_{index}", faulty=False, offset=index * 0.005) for index in range(3)]
    detector = fit_candidate(
        healthy,
        algorithm="one_class_svm",
        feature_config=FeatureConfig(window_samples=3),
        maximum_svm_training_windows=100,
    ).with_threshold(float("-inf"), 0.9845)
    detector.model_version = "maverickx_phase2_threshold_v2_1"
    report = analyze_batch_phase3(
        detector,
        make_batch("fault", faulty=True, offset=3.0),
        healthy,
        tmp_path,
        background_rows=8,
        windows_per_event=1,
    )
    assert report["shap_consistency"]["all_additivity_checks_passed"]
    assert report["shap_consistency"]["maximum_absolute_additivity_residual"] < 1e-6
    assert report["model"]["score_quantity"] == "negative One-Class SVM score_samples"
    assert report["ranked_investigations"][0]["category"] == "cooling_heating_system"
    assert Path(report["files"]["analysis_json"]).exists()
    assert Path(report["files"]["markdown_report"]).exists()
    saved = json.loads(Path(report["files"]["analysis_json"]).read_text())
    assert saved["label_use_statement"].startswith("Fault labels")


class StableProcessReference:
    def event_context(self, batch, *, start_h, end_h):
        return [
            {"channel": name, "unit": "unit", "maximum_absolute_robust_deviation": 0.0,
             "signed_peak_robust_deviation": 0.0, "observed_value_at_peak": 1.0,
             "healthy_phase_median_at_peak": 1.0, "time_h": start_h}
            for name in ("coolant_flow_lph", "heating_flow_lph")
        ]


def test_stable_thermal_controls_prioritize_temperature_probe() -> None:
    event = {
        "start_h": 2.0,
        "end_h": 2.4,
        "explanation": {
            "sensor_contributions": [
                {"sensor": "temperature_k", "mean_positive_contribution": 1.0}
            ]
        },
    }
    rankings, _ = rank_event_investigations(
        event, make_batch("test", faulty=False), StableProcessReference()
    )
    assert rankings[0]["category"] == "temperature_probe"


def test_investigation_evaluation_counts_detection_misses_end_to_end() -> None:
    batch = make_batch("fault", faulty=True)
    reports = {
        "fault": {
            "anomaly_events": [
                {
                    "event_id": 1,
                    "start_h": 0.0,
                    "end_h": 2.0,
                    "ranked_investigations": [
                        {"category": "cooling_heating_system"},
                        {"category": "temperature_probe"},
                    ],
                }
            ]
        }
    }
    mapping = {
        "single_fault_categories": {"coolant_flow_fault": ["cooling_heating_system"]},
        "combined_process_faults_by_interval": {},
    }
    result = evaluate_investigation_reports(reports, [batch], mapping)
    assert result["single_faults"]["top1_accuracy_given_detected"] == 1.0
    assert result["single_faults"]["end_to_end_top3_rate"] == 1.0
