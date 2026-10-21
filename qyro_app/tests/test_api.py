import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from qyro_app import main
    from qyro_app.vision import MockVisionModels

    monkeypatch.setattr(main, "vision_models", MockVisionModels())
    monkeypatch.setattr(main, "llm_provider", None)
    monkeypatch.setattr(main.config, "VISION_MODE", "mock")
    return TestClient(main.app)


def jpeg(size=640, blur=False, base=(120, 150, 200)):
    noise = np.random.default_rng(1).integers(-30, 31, (size, size, 3))
    img = np.clip(np.array(base) + noise, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (51, 51), 20)
    return cv2.imencode(".jpg", img)[1].tobytes()


def post(client, data, questionnaire=None, content_type="image/jpeg"):
    return client.post(
        "/api/v1/analyze",
        files={"image": ("face.jpg", data, content_type)},
        data={"questionnaire": json.dumps(questionnaire or {})},
    )


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["schema_version"] == "qyro.findings.v1"


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "QYRO" in res.text


def test_analyze_returns_structured_findings(client):
    body = post(client, jpeg()).json()
    assert body["schema_version"] == "qyro.findings.v1"
    assert body["image_quality"]["status"] == "pass"
    assert body["vision_findings"]["detection"]["lesion_count"] == 3
    assert "mock_mode" in {f["code"] for f in body["safety_flags"]}
    assert body["disclaimer"]
    assert body["retrieved_evidence"] and body["explanation"]["source"] == "template"
    cited = {i for s in body["explanation"]["next_steps"] for i in s["source_ids"]}
    assert cited <= {e["id"] for e in body["retrieved_evidence"]}


def test_blurry_photo_skips_models(client):
    body = post(client, jpeg(blur=True)).json()
    assert body["recommendation"] == "retake_photo"
    assert body["vision_findings"] is None


def test_non_skin_photo_is_not_analyzed(client):
    body = post(client, jpeg(base=(230, 160, 60))).json()  # blue, sky-like image
    assert body["recommendation"] == "retake_photo"
    assert body["vision_findings"] is None and body["explanation"] is None
    assert "out_of_scope_image" in {f["code"] for f in body["safety_flags"]}


def test_rejects_non_image_upload(client):
    assert post(client, b"not an image", content_type="text/plain").status_code == 415
    assert post(client, b"not an image").status_code == 422


def test_rejects_invalid_questionnaire(client):
    res = post(client, jpeg(), questionnaire={"age_years": 500})
    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "INVALID_QUESTIONNAIRE"
