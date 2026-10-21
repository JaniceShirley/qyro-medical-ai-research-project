"""QYRO demo API: photo + questionnaire in, structured findings out.

Run:  .venv/bin/uvicorn qyro_app.main:app --port 8000
Mock: QYRO_VISION_MODE=mock .venv/bin/uvicorn qyro_app.main:app --port 8000
LLM:  set GROQ_API_KEY or GEMINI_API_KEY (optionally QYRO_LLM_PROVIDER / QYRO_LLM_MODEL); without a key the
      explanation comes from the safe template.
"""
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from qyro_app import config, explain, quality, retrieval, rules
from qyro_app.llm import configured_provider_name, get_provider
from qyro_app.schemas import AnalysisResponse, QualityStatus, Questionnaire
from qyro_app.vision import get_vision_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("qyro_app")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="QYRO Acne Decision Support (demo)", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
vision_models = get_vision_models()
llm_provider = get_provider()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "schema_version": config.SCHEMA_VERSION,
        "vision_mode": config.VISION_MODE,
        "models": {
            "detector": config.DETECTOR_WEIGHTS.exists(),
            "subtype": config.SUBTYPE_WEIGHTS.exists(),
            "severity": config.SEVERITY_WEIGHTS.exists(),
        },
        "llm": {
            "provider": configured_provider_name(),
            "model": getattr(llm_provider, "model", None),
            "enabled": llm_provider is not None,
        },
    }


# Plain `def` so FastAPI runs it in a worker thread: model inference and the LLM call are blocking.
@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(image: UploadFile = File(...), questionnaire: str = Form("{}")):
    request_id = uuid.uuid4().hex[:12]
    timings = {}
    started = time.perf_counter()

    try:
        context = Questionnaire.model_validate_json(questionnaire or "{}")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_QUESTIONNAIRE", "errors": exc.errors(include_url=False)})

    if image.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_IMAGE_TYPE", "message": "Upload a JPG, PNG or WebP photo."})
    data = image.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"code": "IMAGE_TOO_LARGE", "message": "Photo must be 10 MB or smaller."})
    image_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail={"code": "INVALID_IMAGE", "message": "The file could not be read as an image."})

    t = time.perf_counter()
    image_quality = quality.assess_quality(image_bgr)
    timings["quality_ms"] = round((time.perf_counter() - t) * 1000, 1)

    findings = None
    if image_quality.status != QualityStatus.reject:
        t = time.perf_counter()
        findings = vision_models.analyze(image_bgr)
        timings["vision_ms"] = round((time.perf_counter() - t) * 1000, 1)

    flags, recommendation, summary = rules.evaluate(image_quality, findings, context, mock=config.VISION_MODE == "mock")

    evidence, explanation = [], None
    if findings is not None:
        evidence = retrieval.retrieve(findings, flags, context)
        t = time.perf_counter()
        explanation = explain.explain(findings, flags, recommendation, summary, context, evidence, llm_provider)
        timings["explanation_ms"] = round((time.perf_counter() - t) * 1000, 1)
    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

    logger.info("request=%s quality=%s recommendation=%s flags=%s explanation=%s timings=%s",
                request_id, image_quality.status.value, recommendation.value, [f.code for f in flags],
                explanation.source if explanation else None, timings)
    return AnalysisResponse(
        schema_version=config.SCHEMA_VERSION,
        request_id=request_id,
        image_quality=image_quality,
        vision_findings=findings,
        safety_flags=flags,
        recommendation=recommendation,
        summary=summary,
        retrieved_evidence=evidence,
        explanation=explanation,
        disclaimer=rules.DISCLAIMER,
        latency_ms=timings,
    )
