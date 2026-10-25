"""Comparable, interpretable detectors for the phase-2 model study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .data import BatchData
from .features import FeatureConfig, PhaseNormalizer, extract_rolling_features, feature_names


Algorithm = Literal["isolation_forest", "one_class_svm", "robust_z_score"]


@dataclass
class CandidateDetector:
    """A fitted detector with one common scoring interface."""

    model_version: str
    algorithm: Algorithm
    feature_config: FeatureConfig
    phase_normalizer: PhaseNormalizer
    scaler: StandardScaler | None
    estimator: IsolationForest | OneClassSVM | None
    alert_threshold: float
    threshold_quantile: float
    feature_names: list[str]

    def score_features(self, features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        normalized = self.phase_normalizer.transform(features)
        if self.algorithm == "robust_z_score":
            # One strongly unusual feature is enough to make a window
            # suspicious. The feature-level normalized values remain directly
            # readable as robust deviations from healthy batches at that time.
            scores = normalized.abs().max(axis=1).to_numpy(dtype=float)
        else:
            if self.scaler is None or self.estimator is None:
                raise RuntimeError("Learned detector is missing fitted components.")
            transformed = self.scaler.transform(normalized)
            scores = -self.estimator.score_samples(transformed)
        return np.asarray(scores, dtype=float), normalized

    def score_batch(self, batch: BatchData) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = extract_rolling_features(batch, self.feature_config)
        scores, normalized = self.score_features(features)
        timeline = features[["batch_id", "window_start_h", "window_end_h"]].copy()
        timeline["anomaly_score"] = scores
        timeline["alert_threshold"] = self.alert_threshold
        timeline["is_flagged"] = timeline["anomaly_score"] > self.alert_threshold
        timeline["model_version"] = self.model_version
        return timeline, normalized

    def with_threshold(self, threshold: float, quantile: float) -> "CandidateDetector":
        self.alert_threshold = float(threshold)
        self.threshold_quantile = float(quantile)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CandidateDetector":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not a CandidateDetector.")
        return loaded


def combine_features(batches: Iterable[BatchData], config: FeatureConfig) -> pd.DataFrame:
    batches = list(batches)
    if not batches:
        raise ValueError("At least one batch is required.")
    return pd.concat(
        [extract_rolling_features(batch, config) for batch in batches],
        ignore_index=True,
    )


def fit_candidate(
    healthy_training_batches: Iterable[BatchData],
    *,
    algorithm: Algorithm,
    feature_config: FeatureConfig,
    random_seed: int = 42,
    maximum_svm_training_windows: int = 8_000,
) -> CandidateDetector:
    """Fit on healthy batches only; thresholding is intentionally separate."""

    training_batches = list(healthy_training_batches)
    if not training_batches or any(
        batch.metadata.is_fault_batch is not False for batch in training_batches
    ):
        raise ValueError("Candidate training requires only explicitly healthy batches.")
    names = feature_names(feature_config)
    features = combine_features(training_batches, feature_config)
    normalizer = PhaseNormalizer(bin_width_h=0.2)
    normalized = normalizer.fit_transform(features, names)

    scaler: StandardScaler | None = None
    estimator: IsolationForest | OneClassSVM | None = None
    if algorithm != "robust_z_score":
        scaler = StandardScaler()
        transformed = scaler.fit_transform(normalized)
        if algorithm == "isolation_forest":
            estimator = IsolationForest(
                n_estimators=300,
                contamination="auto",
                max_samples="auto",
                random_state=random_seed,
                n_jobs=-1,
            )
        elif algorithm == "one_class_svm":
            if len(transformed) > maximum_svm_training_windows:
                rng = np.random.default_rng(random_seed)
                positions = np.sort(
                    rng.choice(len(transformed), maximum_svm_training_windows, replace=False)
                )
                transformed = transformed[positions]
            estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01, cache_size=1024)
        else:  # pragma: no cover - guarded by the public type and CLI choices
            raise ValueError(f"Unknown algorithm: {algorithm}")
        estimator.fit(transformed)

    window_hours = feature_config.approximate_window_hours
    sensor_tag = "extended" if len(feature_config.sensors) > 4 else "core"
    return CandidateDetector(
        model_version=f"{algorithm}_{sensor_tag}_{window_hours:g}h",
        algorithm=algorithm,
        feature_config=feature_config,
        phase_normalizer=normalizer,
        scaler=scaler,
        estimator=estimator,
        alert_threshold=float("inf"),
        threshold_quantile=float("nan"),
        feature_names=names,
    )


def healthy_score_thresholds(
    detector: CandidateDetector,
    healthy_validation_batches: Iterable[BatchData],
    quantiles: Iterable[float],
) -> dict[float, float]:
    """Derive thresholds using healthy validation scores, never fault labels."""

    scores = np.concatenate(
        [detector.score_batch(batch)[0]["anomaly_score"].to_numpy() for batch in healthy_validation_batches]
    )
    return {float(q): float(np.quantile(scores, q)) for q in quantiles}
