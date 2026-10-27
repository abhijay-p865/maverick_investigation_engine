from pathlib import Path

import maverickx


def test_package_has_version() -> None:
    assert maverickx.__version__ == "0.1.0"


def test_required_web_runtime_assets_are_packaged() -> None:
    root = Path(__file__).parents[1]
    assert (root / "runtime_assets/model/selected_detector_v2_1.joblib").is_file()
    assert (root / "runtime_assets/demo/Batch_40_fault_8.csv").is_file()
    assert len(list((root / "runtime_assets/healthy_background").glob("Batch_*_fault_0.csv"))) >= 5
