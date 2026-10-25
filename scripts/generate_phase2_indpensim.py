"""Generate the locked phase-2 IndPenSim campaign with portable Octave.

The script works in a disposable copy of the simulator so the separately
downloaded upstream checkout stays unchanged. Each batch is an independent
Octave process and therefore can be generated safely in parallel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "phase2_split_manifest.json"
SIMULATOR_SOURCE = ROOT / "IndPenSim"
WORK_ROOT = ROOT / "data" / "simulator_workspace"
OUTPUT_ROOT = ROOT / "data" / "raw"


def octave_literal(path: Path) -> str:
    """Return a path safe inside a single-quoted Octave expression."""

    return path.resolve().as_posix().replace("'", "''")


def prepare_workspace(name: str, fault_codes: list[int]) -> Path:
    workspace = WORK_ROOT / name
    if workspace.exists():
        raise FileExistsError(
            f"Workspace already exists: {workspace}. Remove this generated workspace "
            "only if you intentionally want a clean rerun."
        )
    shutil.copytree(
        SIMULATOR_SOURCE,
        workspace,
        ignore=shutil.ignore_patterns(".git", "output_*", "*.csv"),
    )
    config_path = workspace / "get_fault_batch_run_flags.m"
    source = config_path.read_text(encoding="utf-8")
    vector = "[" + " ".join(str(code) for code in fault_codes) + "]"
    replaced, count = re.subn(
        r"Batch_run_flags\.Batch_fault_order_reference\s*=\s*.*?;",
        f"Batch_run_flags.Batch_fault_order_reference = {vector};",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not locate the simulator campaign configuration line.")
    config_path.write_text(replaced, encoding="utf-8", newline="\n")
    return workspace


def generate_one(octave: Path, workspace: Path, output_dir: Path, batch_no: int) -> str:
    expression = (
        f"addpath('{octave_literal(workspace)}'); "
        f"Generate_Single_Batch_CSV({batch_no}, '{octave_literal(output_dir)}');"
    )
    completed = subprocess.run(
        [str(octave), "--quiet", "--no-gui", "--eval", expression],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Batch {batch_no} failed with exit code {completed.returncode}:\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--octave", type=Path, required=True, help="Path to octave-cli.exe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate one healthy batch in a separate smoke-test directory.",
    )
    args = parser.parse_args()
    if not args.octave.is_file():
        parser.error(f"Octave executable not found: {args.octave}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    fault_codes = [0] if args.smoke else manifest["campaign"]["fault_codes_by_batch"]
    name = "phase2_smoke" if args.smoke else "phase2_indpensim"
    output_dir = OUTPUT_ROOT / name / "csv"
    if output_dir.exists() and any(output_dir.glob("Batch_*.csv")):
        raise FileExistsError(
            f"Output already contains batch CSVs: {output_dir}. Existing generated "
            "evidence will not be overwritten automatically."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(name, fault_codes)

    print(f"Generating {len(fault_codes)} batch(es) with {args.workers} worker(s)...")
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, args.octave, workspace, output_dir, batch_no): batch_no
            for batch_no in range(1, len(fault_codes) + 1)
        }
        for future in concurrent.futures.as_completed(futures):
            batch_no = futures[future]
            try:
                future.result()
                print(f"  completed batch {batch_no}/{len(fault_codes)}", flush=True)
            except Exception as exc:  # report every failed worker before exiting
                failures.append(str(exc))
                print(f"  FAILED batch {batch_no}", file=sys.stderr, flush=True)

    generated = sorted(output_dir.glob("Batch_*.csv"))
    if failures or len(generated) != len(fault_codes):
        print("\n".join(failures), file=sys.stderr)
        print(
            f"Expected {len(fault_codes)} CSVs but found {len(generated)} in {output_dir}.",
            file=sys.stderr,
        )
        return 1
    print(f"Verified {len(generated)} CSVs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
