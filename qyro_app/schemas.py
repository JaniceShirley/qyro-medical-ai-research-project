"""Versioned request/response contracts for the QYRO findings API."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Duration(str, Enum):
    unknown = "unknown"
    under_1_month = "under_1_month"
    one_to_6_months = "1_to_6_months"
    over_6_months = "over_6_months"


class Questionnaire(BaseModel):
    """Patient context. Every field allows 'unknown' and nothing is inferred when it is missing."""
    age_years: Optional[int] = Field(default=None, ge=0, le=120)
    duration: Duration = Duration.unknown
    painful_lumps: Optional[bool] = None
    fever_or_unwell: Optional[bool] = None
    rapid_spread: Optional[bool] = None
    scarring: Optional[bool] = None
    current_treatment: Optional[str] = Field(default=None, max_length=300)


class QualityStatus(str, Enum):
    passed = "pass"
    caution = "caution"
    reject = "reject"


class QualityCheck(BaseModel):
    name: str
    status: QualityStatus
    value: float
    message: str


class ImageQuality(BaseModel):
    status: QualityStatus
    width: int
    height: int
    checks: list[QualityCheck]


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class ModelInfo(BaseModel):
    name: str
    version: str
    available: bool
    note: Optional[str] = None


class Detection(BaseModel):
    model: ModelInfo
    boxes: list[Box] = []
    lesion_count: Optional[int] = None
    mean_confidence: Optional[float] = None


class ClassProbability(BaseModel):
    label: str
    probability: float


class Classification(BaseModel):
    model: ModelInfo
    top_label: Optional[str] = None
    top_probability: Optional[float] = None
    probabilities: list[ClassProbability] = []


class VisionFindings(BaseModel):
    detection: Detection
    subtype: Classification
    severity: Classification


class FlagLevel(str, Enum):
    info = "info"
    caution = "caution"
    urgent = "urgent"


class SafetyFlag(BaseModel):
    code: str
    level: FlagLevel
    message: str


class Recommendation(str, Enum):
    retake_photo = "retake_photo"
    routine_review = "routine_dermatologist_review"
    priority_review = "priority_dermatologist_review"
    urgent_care = "seek_prompt_medical_care"


class Evidence(BaseModel):
    id: str
    title: str
    source: str
    url: str
    excerpt: str


class NextStep(BaseModel):
    text: str = Field(max_length=400)
    source_ids: list[str] = Field(min_length=1)


class ExplanationDraft(BaseModel):
    """What the LLM must return. Validated before anything reaches the user."""
    explanation: str = Field(max_length=1200)
    next_steps: list[NextStep] = Field(min_length=1, max_length=5)
    uncertainty_note: str = Field(max_length=500)


class Explanation(BaseModel):
    source: str  # "llm" or "template"
    provider: Optional[str] = None
    model: Optional[str] = None
    explanation: str
    next_steps: list[NextStep]
    uncertainty_note: str
    validation_passed: bool
    validation_issues: list[str] = []
    fallback_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class AnalysisResponse(BaseModel):
    schema_version: str
    request_id: str
    image_quality: ImageQuality
    vision_findings: Optional[VisionFindings] = None
    safety_flags: list[SafetyFlag]
    recommendation: Recommendation
    summary: str
    retrieved_evidence: list[Evidence] = []
    explanation: Optional[Explanation] = None
    disclaimer: str
    latency_ms: dict[str, float]
