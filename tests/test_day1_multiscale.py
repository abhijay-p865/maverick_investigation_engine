import numpy as np
import pandas as pd

from scripts.run_day1_multiscale_experiment import (
    MultiScaleDetector,
    _healthy_percentile,
)


class FixedDetector:
    def __init__(self, times: list[float], scores: list[float]) -> None:
        self.times = times
        self.scores = scores

    def score_batch(self, _batch):
        timeline = pd.DataFrame(
            {
                "batch_id": ["test"] * len(self.times),
                "window_start_h": self.times,
                "window_end_h": self.times,
                "anomaly_score": self.scores,
                "alert_threshold": [999.0] * len(self.times),
                "is_flagged": [False] * len(self.times),
                "model_version": ["fixed"] * len(self.times),
            }
        )
        normalized = pd.DataFrame({"feature": np.zeros(len(self.times))})
        return timeline, normalized


def test_healthy_percentile_is_monotonic_and_bounded() -> None:
    result = _healthy_percentile(
        np.array([-1.0, 1.0, 2.5, 9.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert result.tolist() == [0.0, 0.25, 0.5, 1.0]


def test_multiscale_detector_aligns_long_evidence_by_window_end_time() -> None:
    detector = MultiScaleDetector(
        short_detector=FixedDetector([0.4, 0.6, 0.8], [1.0, 2.0, 3.0]),
        long_detector=FixedDetector([0.8], [15.0]),
        short_reference=np.array([1.0, 2.0, 3.0, 4.0]),
        long_reference=np.array([0.0, 5.0, 10.0, 15.0]),
        alert_threshold=0.8,
    )

    timeline, _ = detector.score_batch(object())

    assert timeline["anomaly_score"].tolist() == [0.25, 0.5, 1.0]
    assert timeline["is_flagged"].tolist() == [False, False, True]
