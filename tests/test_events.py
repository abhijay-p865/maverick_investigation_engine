import pandas as pd

from maverickx.events import group_anomaly_events


def test_consecutive_flags_are_grouped_into_events() -> None:
    timeline = pd.DataFrame(
        {
            "window_start_h": [0.0, 0.2, 0.4, 0.6, 1.4],
            "window_end_h": [1.0, 1.2, 1.4, 1.6, 2.4],
            "anomaly_score": [0.1, 0.9, 1.0, 0.2, 1.2],
            "is_flagged": [False, True, True, False, True],
        }
    )
    features = pd.DataFrame(
        {
            "ph__mean": [0, 1, 3, 0, 1],
            "temperature_k__mean": [0, 2, 1, 0, 4],
        }
    )
    events = group_anomaly_events(timeline, features)
    assert len(events) == 2
    assert events[0]["flagged_window_count"] == 2
    assert events[0]["maximum_anomaly_score"] == 1.0
    assert events[1]["affected_sensors_heuristic"][0] == "temperature_k"
