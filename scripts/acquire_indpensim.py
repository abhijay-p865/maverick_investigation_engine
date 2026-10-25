"""Download the small public IndPenSim compilation and verify exact hashes."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

FILES = {
    "batches.parquet": (
        "https://huggingface.co/datasets/foundation-models/golden-batch-sentinel-data/"
        "resolve/main/indpensim/batches.parquet",
        "704843412cdf370a54fb19c72f5bbeeeb7989195607930dee9a2eb062acfa124",
    ),
    "statistics.parquet": (
        "https://huggingface.co/datasets/foundation-models/golden-batch-sentinel-data/"
        "resolve/main/indpensim/statistics.parquet",
        "e6df6081c6e500784a808fdf992334f020fb5355a0a3d6ac30a872be400ffe4c",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    destination = Path("data/raw/indpensim")
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = destination / name
        if path.exists():
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"Existing {path} has SHA-256 {actual}, expected {expected}. "
                    "Remove it manually before downloading a clean copy."
                )
            print(f"verified {path} ({actual})")
            continue
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        actual = sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Hash mismatch for {name}: {actual} != {expected}")
        partial.replace(path)
        print(f"downloaded and verified {path} ({actual})")


if __name__ == "__main__":
    main()

