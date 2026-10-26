"""Transparent investigation priorities derived from model and process evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import BatchData


PROCESS_CONTEXT_CHANNELS = (
    "aeration_lph",
    "agitation_rpm",
    "substrate_feed_lph",
    "acid_flow_lph",
    "base_flow_lph",
    "coolant_flow_lph",
    "heating_flow_lph",
    "pressure_bar",
)

# Engineering-scale floors prevent essentially constant healthy commands from
# producing meaningless million-fold robust deviations from floating-point
# noise. These are comparison stabilizers, not alarm thresholds.
PROCESS_SCALE_FLOORS = {
    "aeration_lph": 0.5,
    "agitation_rpm": 5.0,
    "substrate_feed_lph": 0.1,
    "acid_flow_lph": 0.05,
    "base_flow_lph": 0.05,
    "coolant_flow_lph": 0.5,
    "heating_flow_lph": 0.05,
    "pressure_bar": 0.01,
}


@dataclass
class HealthyProcessReference:
    """Healthy median and spread for optional channels at each process phase."""

    bin_width_h: float
    medians: pd.DataFrame
    scales: pd.DataFrame

    @classmethod
    def fit(
        cls,
        healthy_batches: Iterable[BatchData],
        *,
        bin_width_h: float = 0.2,
    ) -> "HealthyProcessReference":
        batches = list(healthy_batches)
        if not batches or any(batch.metadata.is_fault_batch is not False for batch in batches):
            raise ValueError("Process reference requires explicitly healthy batches.")
        frames = []
        for batch in batches:
            available = [name for name in PROCESS_CONTEXT_CHANNELS if name in batch.measurements]
            frame = batch.measurements[["elapsed_time_h", *available]].copy()
            frame["phase_bin"] = np.rint(frame["elapsed_time_h"] / bin_width_h).astype(int)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        available = [name for name in PROCESS_CONTEXT_CHANNELS if name in combined]
        grouped = combined.groupby("phase_bin")[available]
        medians = grouped.median().sort_index()
        scales = grouped.quantile(0.75).sort_index() - grouped.quantile(0.25).sort_index()
        global_scales = combined[available].quantile(0.75) - combined[available].quantile(0.25)
        for name in available:
            floor = max(float(global_scales.get(name, 0.0)), PROCESS_SCALE_FLOORS[name])
            scales[name] = scales[name].fillna(floor).clip(lower=floor)
        return cls(bin_width_h, medians, scales)

    def event_context(
        self,
        batch: BatchData,
        *,
        start_h: float,
        end_h: float,
    ) -> list[dict[str, Any]]:
        frame = batch.measurements.loc[
            batch.measurements["elapsed_time_h"].between(start_h - 1e-8, end_h + 1e-8)
        ].copy()
        if frame.empty:
            return []
        bins = np.rint(frame["elapsed_time_h"] / self.bin_width_h).astype(int)
        medians = self.medians.reindex(bins, method="nearest").set_axis(frame.index)
        scales = self.scales.reindex(bins, method="nearest").set_axis(frame.index)
        context = []
        for name in self.medians.columns:
            if name not in frame:
                continue
            deviations = (frame[name] - medians[name]) / scales[name]
            peak_position = deviations.abs().idxmax()
            context.append(
                {
                    "channel": name,
                    "unit": batch.metadata.units.get(name, "unknown"),
                    "maximum_absolute_robust_deviation": float(deviations.abs().max()),
                    "p90_absolute_robust_deviation": float(deviations.abs().quantile(0.90)),
                    "median_absolute_robust_deviation": float(deviations.abs().median()),
                    "signed_peak_robust_deviation": float(deviations.loc[peak_position]),
                    "observed_value_at_peak": float(frame.loc[peak_position, name]),
                    "healthy_phase_median_at_peak": float(medians.loc[peak_position, name]),
                    "time_h": float(frame.loc[peak_position, "elapsed_time_h"]),
                }
            )
        return context


CATEGORY_DETAILS = {
    "aeration_oxygen_delivery": (
        "Aeration and oxygen delivery",
        "Review aeration commands, gas supply, flow controller, sparger, and line restrictions.",
    ),
    "agitation_mixing": (
        "Agitation and mixing",
        "Review agitator command versus response, motor status, and mixing limitations.",
    ),
    "dissolved_oxygen_probe": (
        "Dissolved oxygen probe",
        "Check calibration, maintenance history, fouling, drift, and agreement with off-gas evidence.",
    ),
    "acid_base_ph_control": (
        "Acid/base dosing and pH control",
        "Review dosing commands, pump delivery, reagent availability, and control-loop response.",
    ),
    "ph_probe": (
        "pH probe",
        "Check calibration, drift, fouling, maintenance, and agreement with dosing behavior.",
    ),
    "substrate_feed_system": (
        "Substrate/feed system",
        "Review feed command, pump delivery, line blockage, vessel level, and material records.",
    ),
    "cooling_heating_system": (
        "Cooling/heating system",
        "Review coolant and heating commands, valve response, utility supply, and heat removal.",
    ),
    "temperature_probe": (
        "Temperature probe",
        "Check calibration, drift, wiring, maintenance, and agreement with utility demand.",
    ),
    "pressure_regulation": (
        "Pressure regulation",
        "Review back-pressure control, vent path, pressure command, and vessel pressure records.",
    ),
    "pressure_probe": (
        "Pressure probe",
        "Check calibration, impulse path, wiring, and agreement with independent pressure records.",
    ),
    "operations_records": (
        "Sampling, transfers, handling, or maintenance records",
        "Compare the event time with batch records, sampling, additions, transfers, alarms, and maintenance.",
    ),
    "broader_contamination_review": (
        "Broader contamination investigation",
        "Only if contamination is independently suspected, review sterility, sampling, raw materials, and microbiology records.",
    ),
}


def _support_maps(event: dict[str, Any], context: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    contributions = event["explanation"]["sensor_contributions"]
    positive = {item["sensor"]: max(item["mean_positive_contribution"], 0.0) for item in contributions}
    total = sum(positive.values())
    sensor_support = {name: value / total if total > 0 else 0.0 for name, value in positive.items()}
    process_support = {
        item["channel"]: min(
            item.get("p90_absolute_robust_deviation", item["maximum_absolute_robust_deviation"]) / 4.0,
            1.0,
        )
        for item in context
    }
    return sensor_support, process_support


def _rank_scores(sensor: dict[str, float], process: dict[str, float]) -> dict[str, float]:
    do = sensor.get("dissolved_oxygen_mg_l", 0.0)
    ph = sensor.get("ph", 0.0)
    temperature = sensor.get("temperature_k", 0.0)
    pressure = sensor.get("pressure_bar", 0.0)
    aeration = process.get("aeration_lph", 0.0)
    agitation = process.get("agitation_rpm", 0.0)
    feed = process.get("substrate_feed_lph", 0.0)
    dosing = max(process.get("acid_flow_lph", 0.0), process.get("base_flow_lph", 0.0))
    thermal = max(process.get("coolant_flow_lph", 0.0), process.get("heating_flow_lph", 0.0))
    pressure_pattern = process.get("pressure_bar", 0.0)
    unexplained = max(0.0, 1.0 - max(aeration, agitation, feed, dosing, thermal, 0.0))
    multi_sensor = min(sum(value > 0.15 for value in (do, ph, temperature, pressure)) / 2.0, 1.0)
    return {
        "aeration_oxygen_delivery": 0.65 * aeration + 0.35 * do,
        "agitation_mixing": 0.65 * agitation + 0.35 * do,
        "dissolved_oxygen_probe": do * (0.70 + 0.30 * (1.0 - max(aeration, agitation))),
        "acid_base_ph_control": 0.65 * dosing + 0.35 * ph,
        "ph_probe": ph * (0.70 + 0.30 * (1.0 - dosing)),
        "substrate_feed_system": 0.75 * feed + 0.25 * max(do, ph),
        "cooling_heating_system": 0.65 * thermal + 0.35 * temperature,
        "temperature_probe": temperature * (0.70 + 0.30 * (1.0 - thermal)),
        "pressure_regulation": 0.65 * pressure + 0.30 * pressure_pattern + 0.05,
        "pressure_probe": pressure * (0.70 + 0.20 * pressure_pattern),
        "operations_records": 0.10 + 0.15 * multi_sensor,
        "broader_contamination_review": 0.05 + 0.10 * multi_sensor * unexplained,
    }


def rank_event_investigations(
    event: dict[str, Any],
    batch: BatchData,
    process_reference: HealthyProcessReference,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank transparent investigation leads; labels are intentionally absent."""

    context = process_reference.event_context(
        batch, start_h=float(event["start_h"]), end_h=float(event["end_h"])
    )
    sensor_support, process_support = _support_maps(event, context)
    scores = _rank_scores(sensor_support, process_support)
    context_by_name = {item["channel"]: item for item in context}
    sensor_names = sorted(sensor_support, key=sensor_support.get, reverse=True)
    process_names = sorted(process_support, key=process_support.get, reverse=True)
    rankings = []
    for category, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        title, verification = CATEGORY_DETAILS[category]
        supporting_sensors = [name for name in sensor_names if sensor_support[name] > 0][:3]
        supporting_context = [
            context_by_name[name]
            for name in process_names
            if process_support[name] >= 0.2
        ][:3]
        observations = [
            f"{name} supplied {sensor_support[name]:.0%} of positive SHAP sensor support."
            for name in supporting_sensors
        ]
        observations.extend(
            f"{item['channel']} had a 90th-percentile deviation of {item['p90_absolute_robust_deviation']:.1f} healthy-reference scale units during the event."
            for item in supporting_context
        )
        uncertainty = (
            "No process or operator event log was supplied; temporal coincidence with interventions could not be checked. "
            "The priority score is a relative evidence score, not a root-cause probability."
        )
        rankings.append(
            {
                "category": category,
                "investigation_title": title,
                "priority_score": float(round(score, 6)),
                "priority_score_meaning": "relative combination of SHAP sensor support and healthy-reference process-pattern agreement",
                "supporting_sensor_features": supporting_sensors,
                "confirmed_observations": observations,
                "possible_explanation": f"The observed pattern could be consistent with an issue in {title.lower()}, but does not establish that cause.",
                "relevant_process_log_events": [],
                "suggested_operator_verification": verification,
                "uncertainty": uncertainty,
            }
        )
    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank
    return rankings, context


def rank_batch_investigations(explained_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate event rankings using the strongest evidence seen in the batch.

    Each batch-level priority links to the event that produced its strongest
    score. Listing every event would be misleading because every transparent
    rule is evaluated for every anomaly event, even when its support is weak.
    """

    by_category: dict[str, dict[str, Any]] = {}
    for event in explained_events:
        for item in event.get("ranked_investigations", []):
            category = item["category"]
            candidate = {**item, "supporting_event_ids": [event["event_id"]]}
            current = by_category.get(category)
            if current is None or item["priority_score"] > current["priority_score"]:
                by_category[category] = candidate
    ranked = sorted(by_category.values(), key=lambda item: item["priority_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked
