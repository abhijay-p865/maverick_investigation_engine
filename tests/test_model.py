from pathlib import Path

import numpy as np
import pandas as pd

from maverickx.data import load_indpensim_dataset
from maverickx.features import FeatureConfig
from maverickx.model import (
    IsolationForestDetector,
    make_stratified_batch_split,
    train_isolation_forest,
)


def load_real_batches():
    root = Path(__file__).parents[1]
    batches_path = root / "data/raw/indpensim/batches.parquet"
    stats_path = root / "data/raw/indpensim/statistics.parquet"
    if not batches_path.exists() or not stats_path.exists():
        return None
    return load_indpensim_dataset(batches_path, stats_path)


def test_batch_split_is_disjoint_and_training_is_healthy() -> None:
    batches = load_real_batches()
    if batches is None:
        return
    split = make_stratified_batch_split(batches, random_seed=42)
    sets = [set(split.train), set(split.validation), set(split.healthy_test), set(split.fault_test)]
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            assert left.isdisjoint(right)
    by_id = {batch.metadata.batch_id: batch for batch in batches}
    assert all(by_id[batch_id].metadata.is_fault_batch is False for batch_id in split.train)
    assert len(split.train) == 54
    assert len(split.validation) == 18
    assert len(split.healthy_test) == 18
    assert len(split.fault_test) == 10


def test_model_inference_output_and_round_trip(tmp_path) -> None:
    batches = load_real_batches()
    if batches is None:
        return
    split = make_stratified_batch_split(batches, random_seed=42)
    config = FeatureConfig(window_samples=6)
    detector = train_isolation_forest(
        batches, split, feature_config=config, n_estimators=20
    )
    target = next(batch for batch in batches if batch.metadata.batch_id == split.fault_test[0])
    timeline, normalized = detector.score_batch(target)
    assert list(timeline.columns) == [
        "batch_id",
        "window_start_h",
        "window_end_h",
        "anomaly_score",
        "alert_threshold",
        "is_flagged",
        "model_version",
    ]
    assert len(timeline) == len(target.measurements) - config.window_samples + 1
    assert timeline["anomaly_score"].notna().all()
    assert timeline["is_flagged"].equals(
        timeline["anomaly_score"] > timeline["alert_threshold"]
    )
    raw_scaled = detector.scaler.transform(normalized)
    np.testing.assert_allclose(
        timeline["anomaly_score"].to_numpy(),
        -detector.estimator.score_samples(raw_scaled),
    )
    assert list(normalized.columns) == detector.feature_names

    artifact = tmp_path / "detector.joblib"
    detector.save(artifact)
    loaded = IsolationForestDetector.load(artifact)
    loaded_timeline, _ = loaded.score_batch(target)
    pd.testing.assert_series_equal(
        timeline["anomaly_score"], loaded_timeline["anomaly_score"]
    )

