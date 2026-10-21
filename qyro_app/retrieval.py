"""Layer 5: retrieve approved guidance passages that match the structured findings.

The corpus is small and curated, so retrieval is deterministic tag matching rather than a vector
database: every result is explainable and testable. Swap in embeddings only if the corpus grows.
"""
import json
from functools import lru_cache
from pathlib import Path

from qyro_app.schemas import Evidence, Questionnaire, SafetyFlag, VisionFindings

CORPUS_PATH = Path(__file__).parent / "knowledge" / "guidance.json"
MAX_PASSAGES = 5

SUBTYPE_TAGS = {
    "open_comedo": ["comedonal"], "closed_comedo": ["comedonal"],
    "papular": ["inflammatory"], "pustular": ["inflammatory"],
    "cystic": ["nodulocystic", "referral"],
}
SEVERITY_TAGS = {
    "stage_1": ["mild_moderate"], "stage_2": ["mild_moderate"],
    "stage_3": ["moderate_severe", "referral"], "stage_4": ["moderate_severe", "referral"],
}
FLAG_TAGS = {
    "fever_or_unwell": ["urgent"], "rapid_spread": ["referral"], "painful_lumps": ["nodulocystic", "referral"],
    "scarring": ["scarring", "referral"], "young_patient": ["children"],
}
# Higher weight = more important to include when space is limited
TAG_WEIGHTS = {"urgent": 100, "children": 6, "referral": 5, "nodulocystic": 4, "scarring": 4, "moderate_severe": 3,
               "mild_moderate": 3, "comedonal": 2, "inflammatory": 2, "current_treatment": 2, "general_care": 1,
               "medicine_safety": 1, "wellbeing": 1}


@lru_cache(maxsize=1)
def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)


def wanted_tags(findings: VisionFindings, flags: list[SafetyFlag], questionnaire: Questionnaire) -> set[str]:
    tags = {"general_care", "medicine_safety", "wellbeing"}
    if findings.subtype.model.available and findings.subtype.top_label:
        tags.update(SUBTYPE_TAGS.get(findings.subtype.top_label, []))
    if findings.severity.model.available and findings.severity.top_label:
        tags.update(SEVERITY_TAGS.get(findings.severity.top_label, []))
    for flag in flags:
        tags.update(FLAG_TAGS.get(flag.code, []))
    if questionnaire.current_treatment:
        tags.add("current_treatment")
    return tags


def retrieve(findings: VisionFindings, flags: list[SafetyFlag], questionnaire: Questionnaire) -> list[Evidence]:
    tags = wanted_tags(findings, flags, questionnaire)
    scored = []
    for p in load_corpus()["passages"]:
        score = sum(TAG_WEIGHTS.get(t, 1) for t in p["tags"] if t in tags)
        if score:
            scored.append((score, p["id"], p))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [
        Evidence(id=p["id"], title=p["title"], source=p["source"], url=p["url"], excerpt=p["text"])
        for _, _, p in scored[:MAX_PASSAGES]
    ]
