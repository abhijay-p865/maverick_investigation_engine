import numpy as np
import pandas as pd
import pytest

from maverickx.data import BatchData, BatchMetadata, DataQualityReport
from maverickx.features import FeatureConfig, extract_rolling_features, feature_names


def make_batch(values: np.ndarray) -> BatchData:
    rows = len(values)
    frame = pd.DataFrame(
        {
            "batch_id": ["batch"] * rows,
            "elapsed_time_h": np.arange(1, rows + 1) * 0.2,
            "dissolved_oxygen_mg_l": values,
            "ph": values + 5,
            "temperature_k": values + 295,
            "pressure_bar": values / 10,
        }
    )
    metadata = BatchMetadata(
        batch_id="batch",
        source_batch_number=None,
        source="test",
        source_version="test",
        control_strategy="test",
        is_fault_batch=False,
        fault_label=None,
        fault_onset_h=None,
        fault_intervals_h=[],
        units={},
    )
    quality = DataQualityReport(rows, {}, {}, 0, 0.2, 0, [])
    return BatchData(frame, metadata, quality)


def test_rolling_feature_values_are_interpretable() -> None:
    batch = make_batch(np.array([1.0, 2.0, 3.0, 4.0]))
    config = FeatureConfig(window_samples=3, sensors=("dissolved_oxygen_mg_l",))
    result = extract_rolling_features(batch, config)
    first = result.iloc[0]
    assert first["dissolved_oxygen_mg_l__mean"] == pytest.approx(2.0)
    assert first["dissolved_oxygen_mg_l__std"] == pytest.approx(np.sqrt(2 / 3))
    assert first["dissolved_oxygen_mg_l__slope_per_h"] == pytest.approx(5.0)
    assert first["dissolved_oxygen_mg_l__change"] == pytest.approx(2.0)
    assert first["window_start_h"] == pytest.approx(0.2)
    assert first["window_end_h"] == pytest.approx(0.6)


def test_future_values_do_not_change_earlier_features() -> None:
    original = make_batch(np.arange(1.0, 9.0))
    changed = make_batch(np.r_[np.arange(1.0, 8.0), 9999.0])
    config = FeatureConfig(window_samples=3, sensors=("dissolved_oxygen_mg_l",))
    left = extract_rolling_features(original, config).iloc[:-1]
    right = extract_rolling_features(changed, config).iloc[:-1]
    pd.testing.assert_frame_equal(left, right)


def test_feature_names_contain_no_labels_or_identifiers() -> None:
    names = feature_names(FeatureConfig())
    forbidden = ("fault", "label", "batch_id", "control")
    assert all(not any(word in name for word in forbidden) for name in names)

