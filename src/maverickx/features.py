"""Causal rolling-window features for completed fermentation batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import BatchData, DataValidationError
from .schema import INITIAL_MODEL_SENSORS


@dataclass(frozen=True)
class FeatureConfig:
    window_samples: int = 6
    expected_sampling_interval_h: float = 0.2
    sensors: tuple[str, ...] = INITIAL_MODEL_SENSORS
    statistics: tuple[str, ...] = ("mean", "std", "slope_per_h", "change")

    @property
    def approximate_window_hours(self) -> float:
        return self.window_samples * self.expected_sampling_interval_h

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["approximate_window_hours"] = self.approximate_window_hours
        return result


def feature_names(config: FeatureConfig) -> list[str]:
    return [
        f"{sensor}__{statistic}"
        for sensor in config.sensors
        for statistic in config.statistics
    ]


def _comparison_group(control_strategy: str | None) -> str:
    # The published fault campaign uses the recipe-driven control setup.
    # This mapping selects the comparable healthy baseline; it is not a model label.
    if control_strategy == "fault_campaign_recipe_driven":
        return "recipe_driven"
    return control_strategy or "unknown"


def _rolling_slope(values: np.ndarray, window: int, interval_h: float) -> np.ndarray:
    positions = np.arange(window, dtype=float) * interval_h
    centered = positions - positions.mean()
    weights = centered / np.dot(centered, centered)
    valid = np.convolve(values, weights[::-1], mode="valid")
    result = np.full(values.shape, np.nan, dtype=float)
    result[window - 1 :] = valid
    return result


def extract_rolling_features(
    batch: BatchData,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Calculate features using only readings at or before each window end."""

    config = config or FeatureConfig()
    frame = batch.measurements
    missing = [sensor for sensor in config.sensors if sensor not in frame]
    if missing:
        raise DataValidationError(
            "Batch is missing configured model sensor(s): " + ", ".join(missing)
        )
    if len(frame) < config.window_samples:
        raise DataValidationError(
            f"Batch has {len(frame)} rows; at least {config.window_samples} are needed."
        )

    times = frame["elapsed_time_h"].to_numpy(dtype=float)
    interval = float(np.median(np.diff(times)))
    if not np.isclose(interval, config.expected_sampling_interval_h, rtol=0.02):
        raise DataValidationError(
            f"Median sampling interval is {interval:g} h, expected approximately "
            f"{config.expected_sampling_interval_h:g} h."
        )

    output = pd.DataFrame(
        {
            "batch_id": frame["batch_id"].astype(str),
            "comparison_group": _comparison_group(batch.metadata.control_strategy),
            "window_start_h": frame["elapsed_time_h"].shift(config.window_samples - 1),
            "window_end_h": frame["elapsed_time_h"],
        }
    )
    for sensor in config.sensors:
        values = frame[sensor].to_numpy(dtype=float)
        rolling = pd.Series(values).rolling(config.window_samples, min_periods=config.window_samples)
        output[f"{sensor}__mean"] = rolling.mean().to_numpy()
        output[f"{sensor}__std"] = rolling.std(ddof=0).to_numpy()
        output[f"{sensor}__slope_per_h"] = _rolling_slope(
            values, config.window_samples, interval
        )
        output[f"{sensor}__change"] = pd.Series(values).diff(
            config.window_samples - 1
        ).to_numpy()

    output = output.iloc[config.window_samples - 1 :].reset_index(drop=True)
    if output[feature_names(config)].isna().any().any():
        raise DataValidationError("Rolling features contain missing values.")
    return output


class PhaseNormalizer:
    """Express features relative to healthy batches at the same process phase."""

    def __init__(self, bin_width_h: float = 1.0, clip: float = 25.0) -> None:
        self.bin_width_h = bin_width_h
        self.clip = clip
        self.feature_names_: list[str] = []
        self.medians_: dict[str, pd.DataFrame] = {}
        self.scales_: dict[str, pd.DataFrame] = {}

    def _bins(self, window_end_h: pd.Series) -> np.ndarray:
        return np.rint(window_end_h.to_numpy(dtype=float) / self.bin_width_h).astype(int)

    def fit(self, features: pd.DataFrame, names: list[str]) -> "PhaseNormalizer":
        self.feature_names_ = list(names)
        global_scale = features[self.feature_names_].quantile(0.75) - features[
            self.feature_names_
        ].quantile(0.25)
        global_scale = global_scale.mask(global_scale <= 1e-9, 1.0)
        self.medians_ = {}
        self.scales_ = {}
        for comparison_group, group_rows in features.groupby("comparison_group"):
            work = group_rows[self.feature_names_].copy()
            work.insert(0, "phase_bin", self._bins(group_rows["window_end_h"]))
            grouped = work.groupby("phase_bin")[self.feature_names_]
            self.medians_[str(comparison_group)] = grouped.median().sort_index()
            scales = grouped.quantile(0.75).sort_index() - grouped.quantile(0.25).sort_index()
            for name in self.feature_names_:
                scales[name] = scales[name].mask(scales[name] <= 1e-9, global_scale[name])
            self.scales_[str(comparison_group)] = scales.fillna(global_scale).clip(lower=1e-9)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_ or not self.scales_:
            raise RuntimeError("PhaseNormalizer must be fitted before transform().")
        normalized = pd.DataFrame(index=features.index, columns=self.feature_names_, dtype=float)
        for comparison_group, group_rows in features.groupby("comparison_group"):
            key = str(comparison_group)
            if key not in self.medians_:
                raise ValueError(f"No healthy phase reference for comparison group {key!r}.")
            bins = self._bins(group_rows["window_end_h"])
            medians = self.medians_[key].reindex(bins, method="nearest").set_axis(group_rows.index)
            scales = self.scales_[key].reindex(bins, method="nearest").set_axis(group_rows.index)
            normalized.loc[group_rows.index] = (
                group_rows[self.feature_names_] - medians
            ) / scales
        return normalized.clip(lower=-self.clip, upper=self.clip)

    def fit_transform(self, features: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        return self.fit(features, names).transform(features)

