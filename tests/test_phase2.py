import numpy as np
import pandas as pd
import pytest

from maverickx.data import BatchData, BatchMetadata, DataQualityReport
from maverickx.features import FeatureConfig
from maverickx.metrics import score_batches, summarize_metrics, summarize_scored_metrics
from maverickx.phase2 import fit_candidate


def make_batch(batch_id: str, *, faulty: bool, offset: float = 0.0) -> BatchData:
    rows = 80
    time = np.arange(rows) * 0.2
    frame = pd.DataFrame(
        {
            "batch_id": batch_id,
            "elapsed_time_h": time,
            "dissolved_oxygen_mg_l": 15 + np.sin(time) + offset,
            "ph": 6.5 + 0.01 * np.cos(time),
            "temperature_k": 298 + 0.02 * np.sin(time),
            "pressure_bar": 0.6 + 0.01 * np.cos(time),
        }
    )
    metadata = BatchMetadata(
        batch_id=batch_id,
        source_batch_number=None,
        source="fixture",
        source_version="1",
        control_strategy="recipe_driven",
        is_fault_batch=faulty,
        fault_label="fixture_fault" if faulty else None,
        fault_onset_h=8.0 if faulty else None,
        fault_intervals_h=[(8.0, 12.0)] if faulty else [],
        units={},
    )
    quality = DataQualityReport(rows, {}, {}, 0, 0.2, 0)
    return BatchData(frame, metadata, quality)


@pytest.mark.parametrize("algorithm", ["isolation_forest", "one_class_svm", "robust_z_score"])
def test_all_candidate_algorithms_score_with_common_output(algorithm) -> None:
    healthy = [make_batch(f"healthy_{index}", faulty=False, offset=index * 0.01) for index in range(3)]
    detector = fit_candidate(
        healthy,
        algorithm=algorithm,
        feature_config=FeatureConfig(window_samples=3),
        maximum_svm_training_windows=100,
    ).with_threshold(float("-inf"), 0.99)
    timeline, normalized = detector.score_batch(make_batch("fault", faulty=True, offset=3.0))
    assert len(timeline) == 78
    assert timeline["anomaly_score"].notna().all()
    assert timeline["is_flagged"].all()
    assert list(normalized) == detector.feature_names


def test_training_rejects_fault_labels() -> None:
    with pytest.raises(ValueError, match="healthy"):
        fit_candidate(
            [make_batch("fault", faulty=True)],
            algorithm="robust_z_score",
            feature_config=FeatureConfig(window_samples=3),
        )


def test_metrics_prioritize_events_and_do_not_feed_labels_to_features() -> None:
    healthy = [make_batch("healthy", faulty=False)]
    detector = fit_candidate(
        healthy,
        algorithm="robust_z_score",
        feature_config=FeatureConfig(window_samples=3),
    ).with_threshold(0.0, 0.5)
    summary = summarize_metrics(
        detector,
        healthy,
        [make_batch("fault", faulty=True, offset=3.0)],
    )
    assert summary["aggregate"]["fault_event_recall"] == 1.0
    assert "fault_active" not in detector.feature_names
    assert all("label" not in name for name in detector.feature_names)


def test_cached_threshold_evaluation_matches_direct_scoring() -> None:
    healthy = [make_batch("healthy", faulty=False)]
    faulty = [make_batch("fault", faulty=True, offset=3.0)]
    detector = fit_candidate(
        healthy,
        algorithm="robust_z_score",
        feature_config=FeatureConfig(window_samples=3),
    ).with_threshold(2.0, 0.99)
    direct = summarize_metrics(detector, healthy, faulty)
    cached = summarize_scored_metrics(
        detector.model_version,
        score_batches(detector, healthy),
        score_batches(detector, faulty),
        detector.alert_threshold,
    )
    assert cached["aggregate"] == direct["aggregate"]
