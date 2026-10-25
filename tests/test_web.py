import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from maverickx.web import PROJECT_ROOT, app


client = TestClient(app)


def test_vercel_entrypoint_exposes_the_web_application() -> None:
    import app as vercel_entrypoint

    assert vercel_entrypoint.app is app


def test_home_and_health_are_available() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert "Turn batch data into an investigation plan" in home.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "maverickx_phase2_threshold_v2_1"


def test_upload_rejects_non_csv() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("batch.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 415
    assert "CSV" in response.json()["detail"]


def test_upload_rejects_files_above_hosting_limit() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("large.csv", b"x" * (4 * 1024 * 1024 + 1), "text/csv")},
    )
    assert response.status_code == 413
    assert "4 MB" in response.json()["detail"]


def test_real_csv_runs_full_web_pipeline_and_serves_report() -> None:
    csv_path = (
        PROJECT_ROOT
        / "runtime_assets/demo/Batch_40_fault_8.csv"
    )
    with csv_path.open("rb") as source:
        response = client.post(
            "/api/analyze",
            files={"file": (csv_path.name, source, "text/csv")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["model_version"] == "maverickx_phase2_threshold_v2_1"
    assert payload["summary"]["anomaly_event_count"] > 0
    assert payload["ranked_investigations"][0]["investigation_title"] == "pH probe"
    assert payload["shap_consistency"]["all_additivity_checks_passed"]
    report_file = payload["downloads"]["json"]
    assert report_file["media_type"] == "application/json"
    report = json.loads(base64.b64decode(report_file["content_base64"]))
    assert report["model"]["model_version"] == "maverickx_phase2_threshold_v2_1"
    assert payload["charts"]["timeline"].startswith("data:image/png;base64,")


def test_demo_endpoint_runs_without_an_upload() -> None:
    response = client.post("/api/demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["anomaly_event_count"] > 0
    assert payload["model"]["model_version"] == "maverickx_phase2_threshold_v2_1"


def test_result_route_blocks_path_traversal() -> None:
    response = client.get("/results/not-a-job/anything.json")
    assert response.status_code == 404
