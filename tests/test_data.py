from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maverickx.data import (
    DataValidationError,
    load_indpensim_csv,
    load_indpensim_csv_directory,
    load_indpensim_dataset,
    validate_internal_batch,
)


def valid_internal_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch_id": ["test_batch"] * rows,
            "elapsed_time_h": np.arange(1, rows + 1) * 0.2,
            "dissolved_oxygen_mg_l": np.linspace(15, 12, rows),
            "ph": np.linspace(6.5, 6.4, rows),
            "temperature_k": np.linspace(298, 298.2, rows),
            "pressure_bar": np.linspace(0.6, 0.7, rows),
        }
    )


def test_validation_requires_all_minimum_sensors() -> None:
    frame = valid_internal_frame().drop(columns="pressure_bar")
    with pytest.raises(DataValidationError, match="pressure_bar"):
        validate_internal_batch(frame)


def test_short_missing_gap_is_reported_and_interpolated() -> None:
    frame = valid_internal_frame()
    frame.loc[3:4, "ph"] = np.nan
    clean, report = validate_internal_batch(frame)
    assert clean["ph"].isna().sum() == 0
    assert report.imputed_counts["ph"] == 2
    assert report.warnings


def test_long_missing_gap_is_rejected() -> None:
    frame = valid_internal_frame()
    frame.loc[2:5, "temperature_k"] = np.nan
    with pytest.raises(DataValidationError, match="temperature_k"):
        validate_internal_batch(frame)


def test_duplicate_timestamp_is_rejected() -> None:
    frame = valid_internal_frame()
    frame.loc[4, "elapsed_time_h"] = frame.loc[3, "elapsed_time_h"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_internal_batch(frame)


def test_non_increasing_timestamp_is_rejected() -> None:
    frame = valid_internal_frame()
    frame.loc[4, "elapsed_time_h"] = 0.1
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_internal_batch(frame)


def test_indpensim_csv_adapter_preserves_fault_metadata(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "Time (h)": np.arange(1, 9) * 0.2,
            "Dissolved oxygen concentration(DO2:mg/L)": np.linspace(15, 14, 8),
            "pH(pH:pH)": np.linspace(6.5, 6.45, 8),
            "Temperature(T:K)": np.linspace(298, 298.1, 8),
            "Air head pressure(pressure:bar)": np.linspace(0.6, 0.7, 8),
            "Batch_ref": [12] * 8,
            "Fault_code": [1] * 8,
        }
    )
    path = tmp_path / "Batch_12_fault_1.csv"
    source.to_csv(path, index=False)
    batch = load_indpensim_csv(path)
    assert batch.metadata.batch_id == "indpensim_012"
    assert batch.metadata.fault_label == "aeration_fault"
    assert batch.metadata.fault_onset_h is None  # This short fixture ends before 20 h.
    assert set(batch.measurements).issuperset(
        {"elapsed_time_h", "dissolved_oxygen_mg_l", "ph", "temperature_k", "pressure_bar"}
    )


def test_generated_csv_directory_has_distinct_prefixed_ids(tmp_path) -> None:
    for batch_number, fault_code in [(2, 5), (1, 0)]:
        source = pd.DataFrame(
            {
                "Time (h)": np.arange(1, 9) * 0.2,
                "Dissolved oxygen concentration(DO2:mg/L)": np.linspace(15, 14, 8),
                "pH(pH:pH)": np.linspace(6.5, 6.45, 8),
                "Temperature(T:K)": np.linspace(298, 298.1, 8),
                "Air head pressure(pressure:bar)": np.linspace(0.6, 0.7, 8),
                "Batch_ref": [batch_number] * 8,
                "Fault_code": [fault_code] * 8,
            }
        )
        source.to_csv(tmp_path / f"Batch_{batch_number:02d}_fault_{fault_code}.csv", index=False)

    batches = load_indpensim_csv_directory(tmp_path, batch_id_prefix="phase2")
    assert [batch.metadata.batch_id for batch in batches] == ["phase2_001", "phase2_002"]
    assert batches[1].metadata.fault_label == "coolant_flow_fault"


def test_real_indpensim_dataset_loads_with_fault_metadata() -> None:
    root = Path(__file__).parents[1]
    batches_path = root / "data/raw/indpensim/batches.parquet"
    stats_path = root / "data/raw/indpensim/statistics.parquet"
    if not batches_path.exists() or not stats_path.exists():
        pytest.skip("Run scripts/acquire_indpensim.py to enable the integration test.")

    batches = load_indpensim_dataset(batches_path, stats_path)
    healthy = [batch for batch in batches if not batch.metadata.is_fault_batch]
    faulty = [batch for batch in batches if batch.metadata.is_fault_batch]
    assert len(batches) == 100
    assert len(healthy) == 90
    assert len(faulty) == 10
    assert {batch.metadata.fault_label for batch in faulty} == {
        "aeration_fault",
        "base_flow_fault",
        "coolant_flow_fault",
        "substrate_feed_fault",
    }
    assert batches[90].metadata.fault_intervals_h == [
        (20.0, 30.0),
        (76.0, 92.0),
        (200.0, 214.0),
    ]
    assert batches[0].quality.median_sampling_interval_h == pytest.approx(0.2)
    assert batches[0].measurements["dissolved_oxygen_mg_l"].notna().all()
