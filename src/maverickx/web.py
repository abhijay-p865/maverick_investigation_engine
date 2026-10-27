"""Local web interface for the MaverickX post-batch investigation pipeline."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import DataValidationError, load_indpensim_csv
from .phase2 import CandidateDetector
from .phase3 import analyze_batch_phase3


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
MODEL_PATH = PROJECT_ROOT / "runtime_assets/model/selected_detector_v2_1.joblib"
BACKGROUND_DIR = PROJECT_ROOT / "runtime_assets/healthy_background"
IS_HOSTED = bool(os.getenv("VERCEL"))
RESULTS_DIR = (
    Path(tempfile.gettempdir()) / "maverickx-results"
    if IS_HOSTED
    else PROJECT_ROOT / "outputs/web"
)
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
DEMO_BATCH_PATH = PROJECT_ROOT / "runtime_assets/demo/Batch_40_fault_8.csv"
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
ANALYSIS_LOCK = Lock()

app = FastAPI(
    title="MaverickX Investigation Engine",
    description="Upload a completed IndPenSim batch and generate a post-batch investigation report.",
    version="0.1.0",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@lru_cache(maxsize=1)
def _runtime_resources() -> tuple[CandidateDetector, list[Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Required v2.1 model is missing: {MODEL_PATH}")
    paths = sorted(BACKGROUND_DIR.glob("Batch_*_fault_0.csv"))[:10]
    if len(paths) < 5:
        raise FileNotFoundError(
            "At least five healthy Phase 2 CSVs are required for SHAP and process references."
        )
    detector = CandidateDetector.load(MODEL_PATH)
    healthy = [load_indpensim_csv(path, batch_id_prefix="web_background") for path in paths]
    return detector, healthy


def _download_url(job_id: str, path_value: str) -> str:
    return f"/results/{job_id}/{Path(path_value).name}"


def _embedded_file(path_value: str, media_type: str) -> dict[str, str]:
    """Embed a generated file so downloads survive stateless hosted requests."""

    path = Path(path_value)
    return {
        "filename": path.name,
        "media_type": media_type,
        "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _embedded_image(path_value: str) -> str:
    encoded = base64.b64encode(Path(path_value).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _web_payload(job_id: str, report: dict[str, Any]) -> dict[str, Any]:
    events = []
    for event in report["anomaly_events"]:
        explanation = event["explanation"]
        peak_window = explanation["explained_windows"][0]
        events.append(
            {
                "event_id": event["event_id"],
                "start_h": event["start_h"],
                "end_h": event["end_h"],
                "duration_h": event["duration_h"],
                "maximum_anomaly_score": event["maximum_anomaly_score"],
                "flagged_window_count": event["flagged_window_count"],
                "interpretation": explanation["plain_language_interpretation"],
                "top_features": explanation["top_features"][:5],
                "sensor_readings": peak_window["sensor_readings"],
                "investigations": event["ranked_investigations"][:3],
            }
        )
    files = report["files"]
    return {
        "job_id": job_id,
        "batch_id": report["batch_metadata"]["batch_id"],
        "summary": report["executive_summary"],
        "model": report["model"],
        "ranked_investigations": report["ranked_investigations"][:6],
        "events": events,
        "limitations": report["limitations"],
        "shap_consistency": report["shap_consistency"],
        "downloads": {
            "json": _embedded_file(files["analysis_json"], "application/json"),
            "markdown": _embedded_file(files["markdown_report"], "text/markdown"),
            "timeline_csv": _embedded_file(files["timeline_csv"], "text/csv"),
        },
        "charts": {
            "timeline": _embedded_image(files["timeline_chart_png"]),
            "shap": _embedded_image(files["shap_chart_png"]),
        },
    }


def _analyze_saved_upload(
    upload_path: Path,
    result_dir: Path,
    job_id: str,
) -> dict[str, Any]:
    try:
        detector, healthy = _runtime_resources()
        batch = load_indpensim_csv(upload_path, batch_id_prefix=f"upload_{job_id}")
        with ANALYSIS_LOCK:
            report = analyze_batch_phase3(
                detector,
                batch,
                healthy,
                result_dir,
                model_artifact_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            )
        return _web_payload(job_id, report)
    finally:
        if IS_HOSTED and result_dir.parent == RESULTS_DIR:
            shutil.rmtree(result_dir, ignore_errors=True)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    healthy_count = len(list(BACKGROUND_DIR.glob("Batch_*_fault_0.csv")))
    return {
        "status": "ready" if MODEL_PATH.exists() and healthy_count >= 5 else "not_ready",
        "model_available": MODEL_PATH.exists(),
        "healthy_background_batches": healthy_count,
        "model_version": "maverickx_phase2_threshold_v2_1",
        "deployment_mode": "hosted" if IS_HOSTED else "local",
        "maximum_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/api/example.csv", include_in_schema=False)
def example_csv() -> FileResponse:
    """Download the packaged IndPenSim-format batch used in the demo."""

    if not DEMO_BATCH_PATH.is_file():
        raise HTTPException(status_code=503, detail="The example CSV is unavailable.")
    return FileResponse(
        DEMO_BATCH_PATH,
        media_type="text/csv",
        filename="maverickx_example_batch.csv",
    )


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "batch.csv"
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Please upload a CSV file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV files are limited to 4 MB.")
    job_id = uuid4().hex[:12]
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=False)
    upload_path = result_dir / "uploaded_batch.csv"
    upload_path.write_bytes(content)
    try:
        return await run_in_threadpool(_analyze_saved_upload, upload_path, result_dir, job_id)
    except (DataValidationError, ValueError, UnicodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        LOGGER.exception("MaverickX runtime asset is missing")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive API boundary
        LOGGER.exception("Unexpected batch analysis failure")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed unexpectedly. Check the server log for details.",
        ) from error


@app.post("/api/demo")
async def analyze_demo() -> dict[str, Any]:
    """Run the packaged simulator example without requiring a user upload."""

    if not DEMO_BATCH_PATH.is_file():
        raise HTTPException(status_code=503, detail="The demonstration batch is unavailable.")
    job_id = uuid4().hex[:12]
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=False)
    upload_path = result_dir / "demo_batch.csv"
    upload_path.write_bytes(DEMO_BATCH_PATH.read_bytes())
    try:
        return await run_in_threadpool(_analyze_saved_upload, upload_path, result_dir, job_id)
    except Exception as error:  # pragma: no cover - defensive API boundary
        LOGGER.error(
            "Demonstration analysis cause: %s: %s",
            type(error).__name__,
            error,
        )
        LOGGER.exception("Demonstration analysis failed")
        raise HTTPException(
            status_code=500,
            detail="The demonstration analysis failed unexpectedly.",
        ) from error


@app.get("/results/{job_id}/{filename}", include_in_schema=False)
def result_file(job_id: str, filename: str) -> FileResponse:
    if not JOB_ID_PATTERN.fullmatch(job_id) or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Result not found.")
    result_dir = (RESULTS_DIR / job_id).resolve()
    path = (result_dir / filename).resolve()
    if path.parent != result_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found.")
    allowed_suffixes = {".json", ".md", ".csv", ".png"}
    if path.suffix.lower() not in allowed_suffixes or path.name == "uploaded_batch.csv":
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, filename=path.name)


def run() -> None:
    """Run the local web application from the installed console command."""

    import uvicorn

    uvicorn.run("maverickx.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
