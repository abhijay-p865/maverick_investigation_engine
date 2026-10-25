"""Label-isolated evaluation of phase-3 investigation rankings."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .data import BatchData


def _expected_categories(
    batch: BatchData,
    interval: tuple[float, float],
    mapping: dict[str, Any],
) -> list[str]:
    label = batch.metadata.fault_label
    if label == "combined_process_faults":
        key = f"{interval[0]:.1f}-{interval[1]:.1f}"
        combined = mapping["combined_process_faults_by_interval"]
        if key in combined:
            return list(combined[key])
        # The loader clips a scheduled interval to the actual batch end. Match
        # that defensible shortened interval to the predefined nominal schedule.
        for nominal, categories in combined.items():
            nominal_start, nominal_end = map(float, nominal.split("-"))
            if abs(nominal_start - interval[0]) < 1e-8 and interval[1] <= nominal_end:
                return list(categories)
        raise KeyError(f"No predefined combined-fault mapping for interval {key}.")
    return list(mapping["single_fault_categories"][label])


def evaluate_investigation_reports(
    reports_by_batch_id: dict[str, dict[str, Any]],
    fault_batches: list[BatchData],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate rankings only after label-free reports have been produced."""

    interval_results = []
    for batch in fault_batches:
        report = reports_by_batch_id[batch.metadata.batch_id]
        for start_h, end_h in batch.metadata.fault_intervals_h:
            matching_events = [
                event
                for event in report["anomaly_events"]
                if event["end_h"] >= start_h and event["start_h"] <= end_h
            ]
            expected = _expected_categories(batch, (start_h, end_h), mapping)
            top1 = {
                event["ranked_investigations"][0]["category"]
                for event in matching_events
                if event["ranked_investigations"]
            }
            top3 = {
                item["category"]
                for event in matching_events
                for item in event["ranked_investigations"][:3]
            }
            interval_results.append(
                {
                    "batch_id": batch.metadata.batch_id,
                    "fault_label": batch.metadata.fault_label,
                    "interval_start_h": start_h,
                    "interval_end_h": end_h,
                    "expected_investigation_categories": expected,
                    "detected": bool(matching_events),
                    "matching_event_ids": [event["event_id"] for event in matching_events],
                    "observed_top1_categories": sorted(top1),
                    "observed_top3_categories": sorted(top3),
                    "top1_hit": bool(set(expected) & top1),
                    "top3_hit": bool(set(expected) & top3),
                }
            )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        detected = [row for row in rows if row["detected"]]
        return {
            "known_fault_intervals": len(rows),
            "detected_intervals": len(detected),
            "detection_recall": len(detected) / len(rows) if rows else None,
            "top1_hits": sum(row["top1_hit"] for row in rows),
            "top3_hits": sum(row["top3_hit"] for row in rows),
            "top1_accuracy_given_detected": (
                sum(row["top1_hit"] for row in detected) / len(detected) if detected else None
            ),
            "top3_accuracy_given_detected": (
                sum(row["top3_hit"] for row in detected) / len(detected) if detected else None
            ),
            "end_to_end_top1_rate": sum(row["top1_hit"] for row in rows) / len(rows) if rows else None,
            "end_to_end_top3_rate": sum(row["top3_hit"] for row in rows) / len(rows) if rows else None,
        }

    single = [row for row in interval_results if row["fault_label"] != "combined_process_faults"]
    combined = [row for row in interval_results if row["fault_label"] == "combined_process_faults"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interval_results:
        by_category[row["fault_label"]].append(row)
    return {
        "evaluation_scope": "post-hoc phase-3 ranking evaluation on the existing phase-2 test split",
        "label_isolation": "Known fault labels and intervals were read only after reports and rankings were generated.",
        "accuracy_definition": "Conditional accuracy uses detected known intervals; end-to-end rates count undetected intervals as misses.",
        "single_faults": summarize(single),
        "combined_faults_reported_separately": summarize(combined),
        "all_faults": summarize(interval_results),
        "by_fault_category": {
            label: summarize(rows) for label, rows in sorted(by_category.items())
        },
        "interval_results": interval_results,
    }
