"""Command-line entry points for training and analyzing historical batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_indpensim_csv, load_indpensim_dataset
from .evaluation import summarize_development_evaluation
from .model import IsolationForestDetector, make_stratified_batch_split, train_isolation_forest
from .reporting import analyze_and_write
from .phase2 import CandidateDetector
from .phase3 import analyze_batch_phase3


def _dataset_paths(root: Path) -> tuple[Path, Path]:
    base = root / "data/raw/indpensim"
    return base / "batches.parquet", base / "statistics.parquet"


def run_pilot(root: Path) -> None:
    batches_path, statistics_path = _dataset_paths(root)
    if not batches_path.exists() or not statistics_path.exists():
        raise FileNotFoundError(
            "IndPenSim files are missing. Run '.venv\\Scripts\\python.exe "
            "scripts\\acquire_indpensim.py' first."
        )
    batches = load_indpensim_dataset(batches_path, statistics_path)
    split = make_stratified_batch_split(batches, random_seed=42)
    detector = train_isolation_forest(batches, split)

    artifact_path = root / "artifacts/isolation_forest_v1.joblib"
    split_path = root / "artifacts/split_manifest.json"
    metadata_path = root / "artifacts/isolation_forest_v1_metadata.json"
    detector.save(artifact_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": detector.model_version,
                "model_type": "Isolation Forest",
                "model_parameters": detector.estimator.get_params(),
                "alert_threshold": detector.alert_threshold,
                "threshold_quantile": detector.threshold_quantile,
                "score_direction": "larger is more anomalous",
                "feature_config": detector.feature_config.to_dict(),
                "feature_names": detector.feature_names,
                "phase_normalization": {
                    "comparison_groups": sorted(detector.phase_normalizer.medians_),
                    "phase_bin_width_h": detector.phase_normalizer.bin_width_h,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_id = {batch.metadata.batch_id: batch for batch in batches}
    selected_ids = [split.healthy_test[0], *split.fault_test]
    summaries = []
    for batch_id in selected_ids:
        report = analyze_and_write(detector, by_id[batch_id], root / "outputs/pilot")
        summaries.append(
            {
                "batch_id": batch_id,
                "fault_label": by_id[batch_id].metadata.fault_label,
                **report["summary"],
            }
        )
        print(
            f"{batch_id}: {report['summary']['flagged_window_count']} flagged windows, "
            f"{report['summary']['anomaly_event_count']} events"
        )
    summary_path = root / "outputs/pilot/pilot_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    evaluation = summarize_development_evaluation(detector, batches)
    evaluation_path = root / "outputs/pilot/development_evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"saved model: {artifact_path}")
    print(f"saved model metadata: {metadata_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved development evaluation: {evaluation_path}")


def run_analyze(root: Path, csv_path: Path, output_dir: Path, model_path: Path) -> None:
    detector = IsolationForestDetector.load(model_path)
    batch = load_indpensim_csv(csv_path)
    report = analyze_and_write(detector, batch, output_dir)
    print(json.dumps(report["summary"], indent=2))


def _healthy_background(directory: Path, maximum_batches: int = 10):
    paths = sorted(directory.glob("Batch_*_fault_0.csv"))[:maximum_batches]
    if not paths:
        raise FileNotFoundError(
            f"No healthy Batch_*_fault_0.csv files were found in {directory}."
        )
    batches = [load_indpensim_csv(path, batch_id_prefix="phase2") for path in paths]
    if any(batch.metadata.is_fault_batch is not False for batch in batches):
        raise ValueError("The selected SHAP background contains a non-healthy batch.")
    return batches


def run_investigate(
    csv_path: Path,
    output_dir: Path,
    model_path: Path,
    background_dir: Path,
) -> None:
    detector = CandidateDetector.load(model_path)
    batch = load_indpensim_csv(csv_path, batch_id_prefix="phase3")
    report = analyze_batch_phase3(
        detector,
        batch,
        _healthy_background(background_dir),
        output_dir,
    )
    result = {
        **report["executive_summary"],
        "top_investigation": (
            report["ranked_investigations"][0]["investigation_title"]
            if report["ranked_investigations"]
            else None
        ),
        "json_report": report["files"]["analysis_json"],
        "markdown_report": report["files"]["markdown_report"],
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maverickx", description="Post-batch fermentation anomaly detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pilot", help="Train and run the verified IndPenSim pilot")
    analyze = subparsers.add_parser("analyze", help="Analyze one IndPenSim-format CSV")
    analyze.add_argument("csv_path", type=Path)
    analyze.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    analyze.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/isolation_forest_v1.joblib"),
    )
    investigate = subparsers.add_parser(
        "investigate", help="Run the complete v2.1 post-batch investigation workflow"
    )
    investigate.add_argument("csv_path", type=Path)
    investigate.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    investigate.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/phase2/selected_detector_v2_1.joblib"),
    )
    investigate.add_argument(
        "--healthy-background-dir",
        type=Path,
        default=Path("data/raw/phase2_indpensim/csv"),
    )
    return parser


def main() -> None:
    root = Path.cwd()
    args = build_parser().parse_args()
    if args.command == "pilot":
        run_pilot(root)
    elif args.command == "analyze":
        run_analyze(root, args.csv_path, args.output_dir, args.model)
    elif args.command == "investigate":
        run_investigate(
            args.csv_path,
            args.output_dir,
            args.model,
            args.healthy_background_dir,
        )


if __name__ == "__main__":
    main()

