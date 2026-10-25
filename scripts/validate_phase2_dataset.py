"""Validate and hash the completed phase-2 simulator campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from maverickx.data import FAULT_LABEL_BY_CODE, load_indpensim_csv_directory


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "raw" / "phase2_indpensim" / "csv"
OUTPUT = ROOT / "outputs" / "phase2" / "dataset_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    batches = load_indpensim_csv_directory(CSV_DIR, batch_id_prefix="phase2")
    manifest = json.loads(
        (ROOT / "configs" / "phase2_split_manifest.json").read_text(encoding="utf-8")
    )
    expected_codes = manifest["campaign"]["fault_codes_by_batch"]
    actual_codes: list[int] = []
    paths = sorted(CSV_DIR.glob("Batch_*.csv"))
    by_number = {int(path.name.split("_", 2)[1]): path for path in paths}
    for batch in batches:
        number = batch.metadata.source_batch_number
        code = int(by_number[number].stem.rsplit("_", 1)[1])
        if batch.metadata.fault_label != FAULT_LABEL_BY_CODE[code]:
            raise RuntimeError(f"Filename and CSV fault metadata disagree for batch {number}.")
        actual_codes.append(code)
    if actual_codes != expected_codes:
        raise RuntimeError("Generated fault-code sequence does not match the locked manifest.")

    simulator_commit = subprocess.run(
        ["git", "-C", str(ROOT / "IndPenSim"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "validation_passed": True,
        "batch_count": len(batches),
        "healthy_batch_count": int(sum(batch.metadata.is_fault_batch is False for batch in batches)),
        "fault_batch_count": int(sum(batch.metadata.is_fault_batch is True for batch in batches)),
        "fault_code_counts": dict(sorted(Counter(actual_codes).items())),
        "row_count": int(sum(len(batch.measurements) for batch in batches)),
        "duration_h_range": [
            min(float(batch.measurements["elapsed_time_h"].max()) for batch in batches),
            max(float(batch.measurements["elapsed_time_h"].max()) for batch in batches),
        ],
        "sampling_interval_h": sorted(
            {round(float(batch.quality.median_sampling_interval_h), 9) for batch in batches}
        ),
        "required_measurement_missing_values": int(
            sum(
                sum(
                    batch.quality.missing_counts.get(name, 0)
                    for name in (
                        "dissolved_oxygen_mg_l",
                        "ph",
                        "temperature_k",
                        "pressure_bar",
                    )
                )
                for batch in batches
            )
        ),
        "simulator_commit": simulator_commit,
        "octave_version": "11.3.0",
        "csv_sha256": {
            path.name: sha256(path) for path in sorted(paths, key=lambda item: item.name)
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "csv_sha256"}, indent=2))


if __name__ == "__main__":
    main()
