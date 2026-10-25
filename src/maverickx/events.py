"""Turn adjacent flagged windows into readable anomaly events."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def group_anomaly_events(
    timeline: pd.DataFrame,
    normalized_features: pd.DataFrame,
    *,
    maximum_gap_h: float = 0.41,
) -> list[dict[str, Any]]:
    """Group alert runs, bridging at most one 12-minute clear window.

    Rolling windows overlap heavily. A single below-threshold window between
    alerts usually represents the same process episode, so the 0.41 h default
    prevents an operator-facing event list from being needlessly fragmented.
    """
    flagged_indices = np.flatnonzero(timeline["is_flagged"].to_numpy(dtype=bool))
    if flagged_indices.size == 0:
        return []
    end_times = timeline.loc[flagged_indices, "window_end_h"].to_numpy(dtype=float)
    breaks = np.flatnonzero(np.diff(end_times) > maximum_gap_h)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, flagged_indices.size - 1]

    events: list[dict[str, Any]] = []
    for event_number, (start_pos, end_pos) in enumerate(zip(starts, ends), start=1):
        positions = flagged_indices[start_pos : end_pos + 1]
        event_windows = timeline.iloc[positions]
        peak_position = int(positions[np.argmax(event_windows["anomaly_score"].to_numpy())])
        peak_features = normalized_features.iloc[peak_position].abs()
        sensor_support: dict[str, float] = {}
        for name, value in peak_features.items():
            sensor = name.split("__", 1)[0]
            sensor_support[sensor] = max(sensor_support.get(sensor, 0.0), float(value))
        affected = [
            sensor
            for sensor, _ in sorted(
                sensor_support.items(), key=lambda item: item[1], reverse=True
            )[:3]
        ]
        representatives = (
            event_windows.nlargest(3, "anomaly_score")
            [["window_start_h", "window_end_h", "anomaly_score"]]
            .to_dict(orient="records")
        )
        event_start = float(event_windows.iloc[0]["window_start_h"])
        event_end = float(event_windows.iloc[-1]["window_end_h"])
        events.append(
            {
                "event_id": event_number,
                "start_h": event_start,
                "end_h": event_end,
                "duration_h": event_end - event_start,
                "maximum_anomaly_score": float(event_windows["anomaly_score"].max()),
                "flagged_window_count": int(len(event_windows)),
                "representative_windows": representatives,
                "affected_sensors_heuristic": affected,
            }
        )
    return events

