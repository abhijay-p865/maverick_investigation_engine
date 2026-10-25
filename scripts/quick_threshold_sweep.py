"""Development-only threshold sweep for the locked phase-2 SVM.

This does not refit the model. It tests whether a slightly lower healthy-score
quantile improves event recall while retaining the existing false-event cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from maverickx.data import BatchData, load_indpensim_csv, load_indpensim_dataset
from maverickx.metrics import score_batches, summarize_scored_metrics
from maverickx.phase2 import CandidateDetector


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "data" / "raw" / "phase2_indpensim" / "csv"
OUTPUT_DIR = ROOT / "outputs" / "phase2"
QUANTILES = (
    0.97,
    0.975,
    0.98,
    0.9825,
    0.983,
    0.9835,
    0.984,
    0.9845,
    0.985,
    0.9875,
    0.99,
)
FALSE_EVENT_CAP = 0.5


def load_generated(numbers: list[int]) -> list[BatchData]:
    by_number = {
        int(path.name.split("_", 2)[1]): path
        for path in GENERATED_DIR.glob("Batch_*.csv")
    }
    return [
        load_indpensim_csv(by_number[number], batch_id_prefix="phase2")
        for number in numbers
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posthoc-test",
        action="store_true",
        help="Evaluate the development-selected threshold on the already-viewed test set.",
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="Save the development-selected threshold as a separate v2.1 artifact.",
    )
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "configs" / "phase2_split_manifest.json").read_text(encoding="utf-8")
    )
    detector = CandidateDetector.load(
        ROOT / "artifacts" / "phase2" / "selected_detector.joblib"
    )
    published = load_indpensim_dataset(
        ROOT / "data" / "raw" / "indpensim" / "batches.parquet",
        ROOT / "data" / "raw" / "indpensim" / "statistics.parquet",
    )
    healthy = load_generated(manifest["development"]["healthy_batches"])
    faults = [
        *[batch for batch in published if batch.metadata.is_fault_batch is True],
        *load_generated(manifest["development"]["fault_batches"]),
    ]
    healthy_scored = score_batches(detector, healthy)
    fault_scored = score_batches(detector, faults)
    healthy_scores = pd.concat(
        [item.timeline["anomaly_score"] for item in healthy_scored], ignore_index=True
    )

    rows: list[dict] = []
    details: dict[str, dict] = {}
    for quantile in QUANTILES:
        threshold = float(healthy_scores.quantile(quantile))
        metrics = summarize_scored_metrics(
            detector.model_version, healthy_scored, fault_scored, threshold
        )
        aggregate = metrics["aggregate"]
        rows.append(
            {
                "threshold_quantile": quantile,
                "threshold": threshold,
                **aggregate,
                "within_false_event_cap": (
                    aggregate["healthy_false_alert_events_per_batch_day"]
                    <= FALSE_EVENT_CAP
                ),
            }
        )
        details[str(quantile)] = metrics

    table = pd.DataFrame(rows)
    eligible = table[table["within_false_event_cap"]].copy()
    eligible = eligible.sort_values(
        [
            "fault_event_recall",
            "healthy_false_alert_events_per_batch_day",
            "median_detection_delay_h",
        ],
        ascending=[False, True, True],
    )
    selected = eligible.iloc[0].to_dict() if not eligible.empty else None
    result = {
        "purpose": "Post-lock development-only quick threshold check",
        "test_data_used": False,
        "false_event_cap_per_batch_day": FALSE_EVENT_CAP,
        "selected_row": selected,
        "metrics_by_quantile": details,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "quick_threshold_sweep.csv", index=False)
    (OUTPUT_DIR / "quick_threshold_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print("\nBEST ELIGIBLE\n", json.dumps(selected, indent=2))

    if args.adopt and selected is not None:
        detector.with_threshold(
            float(selected["threshold"]), float(selected["threshold_quantile"])
        )
        detector.model_version = "maverickx_phase2_threshold_v2_1"
        artifact = ROOT / "artifacts" / "phase2" / "selected_detector_v2_1.joblib"
        detector.save(artifact)
        decision = {
            "model_version": detector.model_version,
            "change": "Threshold-only update; estimator, preprocessing, features, and window unchanged",
            "development_selection": selected,
            "original_artifact_preserved": "artifacts/phase2/selected_detector.joblib",
            "new_artifact": "artifacts/phase2/selected_detector_v2_1.joblib",
            "scientific_status": (
                "Selected using development data. Any comparison on the existing test set "
                "is post-hoc because its earlier results were already viewed."
            ),
        }
        (OUTPUT_DIR / "quick_threshold_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        print(f"\nSaved {artifact}")

    if args.posthoc_test and selected is not None:
        test_healthy = load_generated(manifest["untouched_test"]["healthy_batches"])
        test_faults = load_generated(manifest["untouched_test"]["fault_batches"])
        posthoc = summarize_scored_metrics(
            "maverickx_phase2_quick_threshold_candidate",
            score_batches(detector, test_healthy),
            score_batches(detector, test_faults),
            float(selected["threshold"]),
        )
        original = json.loads(
            (OUTPUT_DIR / "independent_test_comparison.json").read_text(encoding="utf-8")
        )["selected_phase2"]
        comparison = {
            "evaluation_status": (
                "post-hoc confirmation only; this test set had already been viewed "
                "before the quick threshold experiment"
            ),
            "development_selected_quantile": selected["threshold_quantile"],
            "development_selected_threshold": selected["threshold"],
            "quick_candidate": posthoc,
            "locked_phase2_model": original,
            "aggregate_delta_candidate_minus_locked": {
                key: posthoc["aggregate"][key] - original["aggregate"][key]
                for key in posthoc["aggregate"]
                if isinstance(posthoc["aggregate"][key], (int, float))
                and posthoc["aggregate"][key] is not None
                and original["aggregate"][key] is not None
            },
        }
        (OUTPUT_DIR / "quick_threshold_posthoc_test.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        print("\nPOST-HOC TEST CANDIDATE\n", json.dumps(posthoc["aggregate"], indent=2))
        print(
            "\nPOST-HOC DELTA\n",
            json.dumps(comparison["aggregate_delta_candidate_minus_locked"], indent=2),
        )


if __name__ == "__main__":
    main()
