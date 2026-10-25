"""Run the complete phase-3 workflow on the fixed phase-2 test batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maverickx.data import load_indpensim_csv
from maverickx.investigation_evaluation import evaluate_investigation_reports
from maverickx.phase2 import CandidateDetector
from maverickx.phase3 import analyze_batch_phase3


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data/raw/phase2_indpensim/csv"
OUTPUT = ROOT / "outputs/phase3"


def _path(batch_number: int) -> Path:
    matches = list(CSV_DIR.glob(f"Batch_{batch_number:02d}_fault_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV for batch {batch_number}, found {len(matches)}.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate reports instead of reusing them")
    args = parser.parse_args()
    detector = CandidateDetector.load(ROOT / "artifacts/phase2/selected_detector_v2_1.joblib")
    healthy_background = [
        load_indpensim_csv(_path(number), batch_id_prefix="phase2")
        for number in range(1, 11)
    ]
    test_faults = [
        load_indpensim_csv(_path(number), batch_id_prefix="phase2")
        for number in range(33, 49)
    ]
    reports = {}
    for batch in test_faults:
        report_dir = OUTPUT / "evaluation_batches" / batch.metadata.batch_id
        report_path = report_dir / f"{batch.metadata.batch_id}_investigation.json"
        if report_path.exists() and not args.force:
            print(f"Loading existing report for {batch.metadata.batch_id}...", flush=True)
            reports[batch.metadata.batch_id] = json.loads(report_path.read_text())
        else:
            print(f"Analyzing {batch.metadata.batch_id} ({batch.metadata.fault_label})...", flush=True)
            reports[batch.metadata.batch_id] = analyze_batch_phase3(
                detector,
                batch,
                healthy_background,
                report_dir,
                windows_per_event=1,
            )
    healthy = load_indpensim_csv(_path(11), batch_id_prefix="phase2")
    healthy_dir = OUTPUT / "examples" / "healthy"
    healthy_path = healthy_dir / f"{healthy.metadata.batch_id}_investigation.json"
    if healthy_path.exists() and not args.force:
        print(f"Loading existing healthy report {healthy.metadata.batch_id}...", flush=True)
        healthy_report = json.loads(healthy_path.read_text())
    else:
        print(f"Analyzing healthy example {healthy.metadata.batch_id}...", flush=True)
        healthy_report = analyze_batch_phase3(
            detector,
            healthy,
            healthy_background,
            healthy_dir,
            windows_per_event=1,
        )
    mapping = json.loads((ROOT / "configs/phase3_investigation_mapping.json").read_text())
    evaluation = evaluate_investigation_reports(reports, test_faults, mapping)
    evaluation["healthy_example"] = {
        "batch_id": healthy.metadata.batch_id,
        **healthy_report["executive_summary"],
        "report": healthy_report["files"]["markdown_report"],
    }
    evaluation["model_version"] = detector.model_version
    evaluation["mapping_file"] = "configs/phase3_investigation_mapping.json"
    json_path = OUTPUT / "investigation_benchmark.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps({
        "single_faults": evaluation["single_faults"],
        "combined_faults": evaluation["combined_faults_reported_separately"],
        "all_faults": evaluation["all_faults"],
        "benchmark": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
