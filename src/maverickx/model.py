"""Healthy-only Isolation Forest training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .data import BatchData
from .features import (
    FeatureConfig,
    PhaseNormalizer,
    extract_rolling_features,
    feature_names,
)


@dataclass(frozen=True)
class BatchSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    healthy_test: tuple[str, ...]
    fault_test: tuple[str, ...]
    random_seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_stratified_batch_split(
    batches: Iterable[BatchData], random_seed: int = 42
) -> BatchSplit:
    """Split healthy batches within each control regime, never splitting rows."""

    healthy = [batch for batch in batches if batch.metadata.is_fault_batch is False]
    faulty = [batch for batch in batches if batch.metadata.is_fault_batch is True]
    groups: dict[str, list[str]] = {}
    for batch in healthy:
        key = batch.metadata.control_strategy or "unknown"
        groups.setdefault(key, []).append(batch.metadata.batch_id)

    rng = np.random.default_rng(random_seed)
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for key in sorted(groups):
        ids = np.array(sorted(groups[key]), dtype=object)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * 0.60))
        n_validation = int(round(n * 0.20))
        train.extend(ids[:n_train].tolist())
        validation.extend(ids[n_train : n_train + n_validation].tolist())
        test.extend(ids[n_train + n_validation :].tolist())

    return BatchSplit(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        healthy_test=tuple(sorted(test)),
        fault_test=tuple(sorted(batch.metadata.batch_id for batch in faulty)),
        random_seed=random_seed,
    )


def _combine_features(
    batches_by_id: dict[str, BatchData], ids: Iterable[str], config: FeatureConfig
) -> pd.DataFrame:
    return pd.concat(
        [extract_rolling_features(batches_by_id[batch_id], config) for batch_id in ids],
        ignore_index=True,
    )


@dataclass
class IsolationForestDetector:
    model_version: str
    feature_config: FeatureConfig
    phase_normalizer: PhaseNormalizer
    scaler: StandardScaler
    estimator: IsolationForest
    alert_threshold: float
    threshold_quantile: float
    feature_names: list[str]
    split: BatchSplit

    def score_features(self, features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        normalized = self.phase_normalizer.transform(features)
        scaled = self.scaler.transform(normalized)
        # scikit-learn assigns larger score_samples values to normal points.
        # Negating it makes the public MaverickX score increase with unusualness.
        anomaly_scores = -self.estimator.score_samples(scaled)
        return anomaly_scores, normalized

    def score_batch(self, batch: BatchData) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = extract_rolling_features(batch, self.feature_config)
        scores, normalized = self.score_features(features)
        timeline = features[["batch_id", "window_start_h", "window_end_h"]].copy()
        timeline["anomaly_score"] = scores
        timeline["alert_threshold"] = self.alert_threshold
        timeline["is_flagged"] = timeline["anomaly_score"] > self.alert_threshold
        timeline["model_version"] = self.model_version
        return timeline, normalized

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not an IsolationForestDetector.")
        return loaded


def train_isolation_forest(
    batches: Iterable[BatchData],
    split: BatchSplit,
    *,
    feature_config: FeatureConfig | None = None,
    threshold_quantile: float = 0.99,
    n_estimators: int = 300,
) -> IsolationForestDetector:
    batches = list(batches)
    batches_by_id = {batch.metadata.batch_id: batch for batch in batches}
    for batch_id in (*split.train, *split.validation):
        if batches_by_id[batch_id].metadata.is_fault_batch is not False:
            raise ValueError(f"Training/validation batch {batch_id} is not labeled healthy.")
    if set(split.train) & set(split.validation):
        raise ValueError("Training and validation batch IDs overlap.")

    config = feature_config or FeatureConfig()
    names = feature_names(config)
    train_features = _combine_features(batches_by_id, split.train, config)
    validation_features = _combine_features(batches_by_id, split.validation, config)

    # IndPenSim readings align at 0.2 h. Using that exact phase resolution avoids
    # confusing scheduled recipe changes with faults.
    phase_normalizer = PhaseNormalizer(bin_width_h=0.2)
    train_normalized = phase_normalizer.fit_transform(train_features, names)
    validation_normalized = phase_normalizer.transform(validation_features)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_normalized)
    validation_scaled = scaler.transform(validation_normalized)

    estimator = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples="auto",
        random_state=split.random_seed,
        n_jobs=-1,
    )
    estimator.fit(train_scaled)
    validation_scores = -estimator.score_samples(validation_scaled)
    threshold = float(np.quantile(validation_scores, threshold_quantile))

    return IsolationForestDetector(
        model_version="isolation_forest_v1",
        feature_config=config,
        phase_normalizer=phase_normalizer,
        scaler=scaler,
        estimator=estimator,
        alert_threshold=threshold,
        threshold_quantile=threshold_quantile,
        feature_names=names,
        split=split,
    )

