"""Remove an unnecessary PyArrow pickle dependency from the frozen v2.1 artifact.

Pandas 3 can store DataFrame column labels as ArrowStringArray when PyArrow is
installed. The detector's phase-reference tables are numeric, but their column
indexes inherited that optional storage type. Converting those labels to plain
Python objects makes the unchanged model loadable in a smaller hosted runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from maverickx.data import load_indpensim_csv
from maverickx.phase2 import CandidateDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "runtime_assets/model/selected_detector_v2_1.joblib"
DEMO_PATH = PROJECT_ROOT / "runtime_assets/demo/Batch_40_fault_8.csv"


def main() -> None:
    detector = CandidateDetector.load(MODEL_PATH)
    demo = load_indpensim_csv(DEMO_PATH, batch_id_prefix="portability_check")
    before, _ = detector.score_batch(demo)

    for attribute in ("medians_", "scales_"):
        frames = getattr(detector.phase_normalizer, attribute)
        for frame in frames.values():
            frame.columns = pd.Index([str(value) for value in frame.columns], dtype=object)

    temporary_path = MODEL_PATH.with_suffix(".portable.joblib")
    detector.save(temporary_path)
    portable = CandidateDetector.load(temporary_path)
    after, _ = portable.score_batch(demo)

    np.testing.assert_allclose(
        before["anomaly_score"],
        after["anomaly_score"],
        rtol=0.0,
        atol=0.0,
    )
    if b"pyarrow" in temporary_path.read_bytes():
        raise RuntimeError("Portable artifact still contains a PyArrow pickle reference.")

    temporary_path.replace(MODEL_PATH)
    print(
        "Portable v2.1 artifact written; all demo anomaly scores are exactly unchanged."
    )


if __name__ == "__main__":
    main()
