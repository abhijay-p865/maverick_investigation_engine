import base64
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import FormatChecker, validate as validate_json_schema

from maverickx.web import PROJECT_ROOT, app


client = TestClient(app)


def test_vercel_entrypoint_exposes_the_web_application() -> None:
    import app as vercel_entrypoint

    assert vercel_entrypoint.app is app


def test_home_and_health_are_available() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert "Find unusual batch behavior" in home.text
    assert "shows the measurements behind each flag" in home.text
    assert "/assets/lab-culture-hero.jpg" in client.get("/assets/styles.css").text
    assert client.get("/assets/lab-culture-hero.jpg").status_code == 200
    assert "MAVERICK X" not in home.text
    assert "RESEARCH PROTOTYPE · V2.1" not in home.text
    assert "Evidence first" not in home.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "maverickx_phase2_threshold_v2_1"


def test_csv_template_and_schema_guide_are_available() -> None:
    home = client.get("/")
    assert "elapsed time (h)" in home.text
    assert 'href="/api/example.csv"' in home.text
    assert 'href="/assets/csv-schema-guide.html"' in home.text
    example = client.get("/api/example.csv")
    assert example.status_code == 200
    assert example.headers["content-type"].startswith("text/csv")
    assert "Time (h)" in example.text
    guide = client.get("/assets/csv-schema-guide.html")
    assert guide.status_code == 200
    assert "Vessel head pressure" in guide.text
    assert ">bar<" in guide.text


def test_priority_interface_labels_evidence_and_event_timing() -> None:
    home = client.get("/")
    script = client.get("/assets/app.js")
    assert script.status_code == 200
    assert "Priority score</strong> runs from 0 to 1" in home.text
    assert "Unusual-behavior score" in home.text
    assert "Model evidence" in home.text
    assert 'evidenceBlock("Observed evidence"' in script.text
    assert 'evidenceBlock("Possible explanation"' in script.text
    assert 'evidenceBlock("When observed"' in script.text
    assert "supporting_event_ids" in script.text
    assert "feature.display_name" in script.text
    assert "feature.plain_language_meaning" in script.text
    assert '.replaceAll("_", " ")' not in script.text
    assert "Higher = stronger relative evidence" in script.text
    assert "saved v2.1" not in home.text
    assert "saved v2.1" not in script.text


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
    assert "v2.1" not in payload["summary"]["plain_language_summary"]
    assert payload["summary"]["anomaly_event_count"] > 0
    assert payload["ranked_investigations"][0]["investigation_title"] == "pH probe"
    assert payload["shap_consistency"]["all_additivity_checks_passed"]
    first_feature = payload["events"][0]["top_features"][0]
    assert "_" not in first_feature["display_name"]
    assert first_feature["plain_language_meaning"]
    assert payload["events"][0]["sensor_readings"][0]["display_name"]
    visible_text = json.dumps(payload["ranked_investigations"])
    assert "dissolved_oxygen_mg_l contributed" not in visible_text
    assert "coolant_flow_lph differed" not in visible_text
    report_file = payload["downloads"]["json"]
    assert report_file["media_type"] == "application/json"
    report = json.loads(base64.b64decode(report_file["content_base64"]))
    assert report["model"]["model_version"] == "maverickx_phase2_threshold_v2_1"
    model_path = PROJECT_ROOT / "runtime_assets/model/selected_detector_v2_1.joblib"
    schema_path = PROJECT_ROOT / "schemas/investigation-report-v1.schema.json"
    validate_json_schema(
        report,
        json.loads(schema_path.read_text(encoding="utf-8")),
        format_checker=FormatChecker(),
    )
    assert report["analysis_metadata"]["input_sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert report["analysis_metadata"]["model_artifact_sha256"] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert report["analysis_metadata"]["analysis_timestamp_utc"].endswith("Z")
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
