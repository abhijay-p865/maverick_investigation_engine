"""Load and validate historical fermentation batches."""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import INDPENSIM_COLUMN_MAP, REQUIRED_COLUMNS, REQUIRED_MEASUREMENTS, UNITS


class DataValidationError(ValueError):
    """Raised when a batch cannot be used without inventing measurements."""


@dataclass
class DataQualityReport:
    row_count: int
    missing_counts: dict[str, int]
    imputed_counts: dict[str, int]
    duplicate_timestamp_count: int
    median_sampling_interval_h: float | None
    irregular_interval_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchMetadata:
    batch_id: str
    source_batch_number: int | None
    source: str
    source_version: str
    control_strategy: str | None
    is_fault_batch: bool | None
    fault_label: str | None
    fault_onset_h: float | None
    fault_intervals_h: list[tuple[float, float]]
    units: dict[str, str]
    input_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchData:
    measurements: pd.DataFrame
    metadata: BatchMetadata
    quality: DataQualityReport


FAULT_LABEL_BY_BATCH = {
    91: "substrate_feed_fault",
    92: "base_flow_fault",
    93: "coolant_flow_fault",
    94: "aeration_fault",
    95: "substrate_feed_fault",
    96: "coolant_flow_fault",
    97: "substrate_feed_fault",
    98: "base_flow_fault",
    99: "aeration_fault",
    100: "aeration_fault",
}

FAULT_LABEL_BY_CODE = {
    0: None,
    1: "aeration_fault",
    2: "pressure_fault",
    3: "substrate_feed_fault",
    4: "base_flow_fault",
    5: "coolant_flow_fault",
    6: "combined_process_faults",
    7: "temperature_sensor_fault",
    8: "ph_sensor_fault",
}

FAULT_INTERVALS_BY_CODE = {
    0: [],
    1: [(20.0, 24.0), (100.0, 110.0)],
    2: [(100.0, 104.0), (200.0, 240.0)],
    3: [(20.0, 30.0), (76.0, 92.0), (200.0, 214.0)],
    4: [(80.0, 84.0), (140.0, 160.0)],
    5: [(70.0, 90.0), (240.0, 270.0)],
    # Code 6 applies all five process-fault schedules. These are the exact
    # unions of those schedules; the clear 92--100 h gap must not count as a
    # known-fault interval during evaluation.
    6: [(20.0, 30.0), (70.0, 92.0), (100.0, 110.0), (140.0, 160.0), (200.0, 270.0)],
    7: [(40.0, float("inf"))],
    8: [(40.0, float("inf"))],
}

FAULT_REFERENCE_COLUMN = "Fault reference(Fault_ref:Fault ref)"
CONTROL_REFERENCE_COLUMN = (
    "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"
)
RAMAN_REFERENCE_COLUMN = "1- No Raman spec"


def _fault_intervals(times: pd.Series, active: pd.Series) -> list[tuple[float, float]]:
    active_times = times.loc[active.astype(bool)].to_numpy(dtype=float)
    if active_times.size == 0:
        return []
    typical_step = float(np.median(np.diff(times.to_numpy(dtype=float))))
    breaks = np.flatnonzero(np.diff(active_times) > typical_step * 1.5)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, active_times.size - 1]
    return [(float(active_times[a]), float(active_times[z])) for a, z in zip(starts, ends)]


def _control_strategy(batch_number: int, source: pd.DataFrame) -> str:
    if 1 <= batch_number <= 30:
        return "recipe_driven"
    if 31 <= batch_number <= 60:
        return "operator_controlled"
    if 61 <= batch_number <= 90:
        return "advanced_process_control"
    return "fault_campaign_recipe_driven"


def _validate_and_fill(
    frame: pd.DataFrame,
    *,
    max_short_gap_points: int = 2,
) -> tuple[pd.DataFrame, DataQualityReport]:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    clean = frame.copy()
    numeric_columns = ["elapsed_time_h", *REQUIRED_MEASUREMENTS]
    for column in numeric_columns:
        converted = pd.to_numeric(clean[column], errors="coerce")
        newly_invalid = int(converted.isna().sum() - clean[column].isna().sum())
        if newly_invalid:
            raise DataValidationError(
                f"Column {column!r} contains {newly_invalid} non-numeric values."
            )
        clean[column] = converted
        finite_values = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_values).all():
            raise DataValidationError(
                f"Column {column!r} contains a non-finite value. Replace Infinity with a finite value; leave missing readings blank."
            )

    if clean["elapsed_time_h"].isna().any():
        raise DataValidationError("Elapsed time contains missing values.")

    duplicate_count = int(clean["elapsed_time_h"].duplicated().sum())
    if duplicate_count:
        raise DataValidationError(
            f"Elapsed time contains {duplicate_count} duplicate reading(s)."
        )

    time_values = clean["elapsed_time_h"].to_numpy(dtype=float)
    differences = np.diff(time_values)
    if np.any(differences <= 0):
        raise DataValidationError("Elapsed time must be strictly increasing.")

    median_interval = float(np.median(differences)) if differences.size else None
    irregular_count = 0
    if median_interval is not None:
        irregular_count = int(
            np.count_nonzero(~np.isclose(differences, median_interval, rtol=0.02, atol=1e-6))
        )

    missing_before = {column: int(clean[column].isna().sum()) for column in clean.columns}
    imputed_counts: dict[str, int] = {}
    for column in REQUIRED_MEASUREMENTS:
        before = int(clean[column].isna().sum())
        if before:
            clean[column] = clean[column].interpolate(
                method="linear", limit=max_short_gap_points, limit_area="inside"
            )
        after = int(clean[column].isna().sum())
        imputed_counts[column] = before - after
        if after:
            raise DataValidationError(
                f"Required measurement {column!r} has {after} missing value(s) "
                "after conservative short-gap interpolation."
            )

    warnings: list[str] = []
    if irregular_count:
        warnings.append(f"Found {irregular_count} inconsistent sampling interval(s).")
    if sum(imputed_counts.values()):
        warnings.append(
            f"Interpolated {sum(imputed_counts.values())} required measurement value(s) "
            f"across gaps no longer than {max_short_gap_points} samples."
        )

    report = DataQualityReport(
        row_count=len(clean),
        missing_counts=missing_before,
        imputed_counts=imputed_counts,
        duplicate_timestamp_count=duplicate_count,
        median_sampling_interval_h=median_interval,
        irregular_interval_count=irregular_count,
        warnings=warnings,
    )
    return clean, report


def _batch_from_source_frame(
    source: pd.DataFrame,
    *,
    batch_id: str,
    source_batch_number: int | None,
    source_name: str,
    source_version: str,
    control_strategy: str | None,
    is_fault_batch: bool | None,
    fault_label: str | None,
    fault_active: pd.Series | None,
    fault_intervals: list[tuple[float, float]] | None = None,
    input_sha256: str | None = None,
) -> BatchData:
    available = {key: value for key, value in INDPENSIM_COLUMN_MAP.items() if key in source}
    frame = source[list(available)].rename(columns=available).copy()
    frame.insert(0, "batch_id", batch_id)

    if fault_active is not None:
        frame["fault_active"] = fault_active.to_numpy(dtype=bool)
        intervals = _fault_intervals(frame["elapsed_time_h"], frame["fault_active"])
    else:
        intervals = fault_intervals or []

    frame, quality = _validate_and_fill(frame)
    onset = intervals[0][0] if intervals else None
    metadata = BatchMetadata(
        batch_id=batch_id,
        source_batch_number=source_batch_number,
        source=source_name,
        source_version=source_version,
        control_strategy=control_strategy,
        is_fault_batch=is_fault_batch,
        fault_label=fault_label,
        fault_onset_h=onset,
        fault_intervals_h=intervals,
        units={key: UNITS[key] for key in frame.columns if key in UNITS},
        input_sha256=input_sha256,
    )
    return BatchData(frame.reset_index(drop=True), metadata, quality)


def load_indpensim_dataset(
    batches_path: str | Path,
    statistics_path: str | Path,
) -> list[BatchData]:
    """Load the published 100-batch IndPenSim parquet compilation.

    The compilation's last two columns named ``Batch ID`` and ``Fault flag``
    contain unrelated large numeric values. Stable batch IDs are therefore
    reconstructed from the 100 documented elapsed-time resets and verified
    against the 100 ordered rows in ``statistics.parquet``.
    """

    raw = pd.read_parquet(batches_path)
    statistics = pd.read_parquet(statistics_path)
    if "Time (h)" not in raw:
        raise DataValidationError("IndPenSim dataset is missing 'Time (h)'.")
    if len(statistics) != 100 or statistics["Batch ref"].tolist() != list(range(1, 101)):
        raise DataValidationError("Statistics file does not contain ordered batches 1 through 100.")

    times = pd.to_numeric(raw["Time (h)"], errors="raise").to_numpy(dtype=float)
    starts = np.r_[0, np.flatnonzero(times[1:] <= times[:-1]) + 1]
    if len(starts) != len(statistics):
        raise DataValidationError(
            f"Found {len(starts)} time-reset batches but {len(statistics)} statistics rows."
        )
    ends = np.r_[starts[1:], len(raw)]

    batches: list[BatchData] = []
    for batch_number, (start, end) in enumerate(zip(starts, ends), start=1):
        source = raw.iloc[start:end].copy()
        fault_active = source[FAULT_REFERENCE_COLUMN].astype(bool)
        is_fault = bool(statistics.iloc[batch_number - 1]["Fault ref(0-NoFault 1-Fault)"])
        label = FAULT_LABEL_BY_BATCH.get(batch_number)
        if is_fault != (label is not None):
            raise DataValidationError(
                f"Fault metadata disagreement for source batch {batch_number}."
            )
        batch = _batch_from_source_frame(
            source,
            batch_id=f"indpensim_{batch_number:03d}",
            source_batch_number=batch_number,
            source_name="IndPenSim",
            source_version="100-batch V3 public compilation",
            control_strategy=_control_strategy(batch_number, source),
            is_fault_batch=is_fault,
            fault_label=label,
            fault_active=fault_active,
        )
        batches.append(batch)
    return batches


def load_indpensim_csv(path: str | Path, *, batch_id_prefix: str = "indpensim") -> BatchData:
    """Load one CSV written by the IndPenSim simulator repository."""

    path = Path(path)
    input_bytes = path.read_bytes()
    source = pd.read_csv(io.BytesIO(input_bytes))
    source_batch_number: int | None = None
    if "Batch_ref" in source and source["Batch_ref"].nunique(dropna=True) == 1:
        source_batch_number = int(source["Batch_ref"].dropna().iloc[0])
    batch_id = (
        f"{batch_id_prefix}_{source_batch_number:03d}"
        if source_batch_number is not None
        else f"{batch_id_prefix}_{path.stem}"
    )

    fault_code: int | None = None
    if "Fault_code" in source:
        codes = source["Fault_code"].dropna().astype(int).unique()
        if len(codes) != 1 or int(codes[0]) not in FAULT_LABEL_BY_CODE:
            raise DataValidationError("Fault_code must contain one documented value from 0 through 8.")
        fault_code = int(codes[0])

    control_strategy = "recipe_driven"
    control_columns = [
        column for column in source if column.startswith("0 - Recipe driven 1 - Operator controlled")
    ]
    raman_columns = [column for column in source if "No Raman spec" in column]
    if control_columns and int(source[control_columns[0]].dropna().iloc[0]) == 1:
        control_strategy = "operator_controlled"
    elif raman_columns and int(source[raman_columns[0]].dropna().iloc[0]) == 2:
        control_strategy = "advanced_process_control"

    intervals = [] if fault_code is None else FAULT_INTERVALS_BY_CODE[fault_code]
    max_time = float(pd.to_numeric(source["Time (h)"], errors="raise").max())
    clipped_intervals = [
        (start, min(end, max_time)) for start, end in intervals if start <= max_time
    ]
    return _batch_from_source_frame(
        source,
        batch_id=batch_id,
        source_batch_number=source_batch_number,
        source_name="IndPenSim CSV",
        source_version="Lemniscabio simulator export",
        control_strategy=control_strategy,
        is_fault_batch=None if fault_code is None else fault_code != 0,
        fault_label=None if fault_code is None else FAULT_LABEL_BY_CODE[fault_code],
        fault_active=None,
        fault_intervals=clipped_intervals,
        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
    )


def load_indpensim_csv_directory(
    directory: str | Path, *, batch_id_prefix: str = "indpensim_generated"
) -> list[BatchData]:
    """Load and validate every simulator CSV in numeric batch order."""

    directory = Path(directory)
    paths = list(directory.glob("Batch_*.csv"))
    if not paths:
        raise DataValidationError(f"No Batch_*.csv files were found in {directory}.")
    batches = [
        load_indpensim_csv(path, batch_id_prefix=batch_id_prefix) for path in paths
    ]
    batches.sort(
        key=lambda batch: (
            batch.metadata.source_batch_number is None,
            batch.metadata.source_batch_number or 0,
        )
    )
    source_numbers = [batch.metadata.source_batch_number for batch in batches]
    if any(number is None for number in source_numbers):
        raise DataValidationError("Every generated CSV must contain one Batch_ref value.")
    if len(set(source_numbers)) != len(source_numbers):
        raise DataValidationError("Generated CSV directory contains duplicate Batch_ref values.")
    return batches


def validate_internal_batch(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Public validation entry point for already-normalized batch tables."""

    return _validate_and_fill(frame)

