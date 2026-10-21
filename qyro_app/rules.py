"""Layer 4: deterministic logic. Counts, uncertainty thresholds, red flags and escalation.

No model or LLM decides the recommendation; these rules do, and they are unit-tested.
"""
from qyro_app import config
from qyro_app.schemas import (
    FlagLevel, ImageQuality, QualityStatus, Questionnaire, Recommendation, SafetyFlag, VisionFindings,
)

DISCLAIMER = (
    "QYRO is an AI-assisted decision-support prototype. It does not diagnose. "
    "All findings must be reviewed by a qualified dermatologist."
)

# Subtypes with too little training data for a reliable prediction (see subtype_manifest.csv counts)
LOW_DATA_SUBTYPES = {"mixed", "scar", "infantile", "mechanica"}

# Measured on the frozen Kurnaz test split (78 images) on 2026-10-21
DETECTOR_TEST_RECALL = 0.61

# The severity model under-grades stage 3 on the test split (10 of 14 predicted stage 2), so a
# meaningful probability of stage 3/4 escalates even when the top prediction is stage 1/2.
HIGHER_SEVERITY_PROBABILITY = 0.25

# Flags that mean "an AI result may be wrong": these never allow a routine recommendation.
UNCERTAINTY_FLAGS = {"image_quality_caution", "low_detection_confidence", "low_subtype_confidence",
                     "low_data_subtype", "low_severity_confidence", "possible_higher_severity"}


def _flag(code, level, message):
    return SafetyFlag(code=code, level=level, message=message)


def evaluate(quality: ImageQuality, findings: VisionFindings | None, questionnaire: Questionnaire, mock: bool = False):
    """Returns (flags, recommendation, summary)."""
    flags: list[SafetyFlag] = []
    priority = False
    urgent = False

    if mock:
        flags.append(_flag("mock_mode", FlagLevel.caution, "Demo mock mode: model outputs are fake placeholders."))

    # Patient-reported red flags apply even when the photo is unusable.
    q = questionnaire
    if q.fever_or_unwell:
        urgent = True
        flags.append(_flag("fever_or_unwell", FlagLevel.urgent,
                           "Fever or feeling unwell with skin lesions needs prompt in-person medical assessment."))
    if q.rapid_spread:
        priority = True
        flags.append(_flag("rapid_spread", FlagLevel.urgent, "Rapidly spreading lesions should be seen by a clinician soon."))
    if q.painful_lumps:
        priority = True
        flags.append(_flag("painful_lumps", FlagLevel.caution, "Painful deep lumps can indicate nodulocystic acne; dermatologist review advised."))
    if q.scarring:
        priority = True
        flags.append(_flag("scarring", FlagLevel.caution, "Reported scarring: early dermatologist review can help limit further scarring."))
    if q.age_years is not None and q.age_years < 12:
        priority = True
        flags.append(_flag("young_patient", FlagLevel.caution, "Patient is under 12, outside the population the models were trained on."))

    if quality.status == QualityStatus.reject:
        rejected = [c for c in quality.checks if c.status == QualityStatus.reject]
        if any(c.name == "skin_visible" for c in rejected):
            flags.append(_flag("out_of_scope_image", FlagLevel.caution,
                               "This does not look like a photo of skin, so it was not analyzed. Please upload a photo of the affected skin."))
        reasons = ", ".join(c.name for c in rejected)
        flags.append(_flag("image_rejected", FlagLevel.caution, f"Photo not usable ({reasons}). Please retake it."))
        recommendation = Recommendation.urgent_care if urgent else Recommendation.retake_photo
        summary = "The photo did not pass the quality check, so no AI analysis was run."
        return flags, recommendation, summary

    if quality.status == QualityStatus.caution:
        reasons = ", ".join(c.name for c in quality.checks if c.status == QualityStatus.caution)
        flags.append(_flag("image_quality_caution", FlagLevel.caution, f"Photo quality is borderline ({reasons}); results are less reliable."))

    det = findings.detection
    if not det.model.available:
        flags.append(_flag("detector_unavailable", FlagLevel.caution, "Lesion detector is not available; no lesion count."))
    else:
        flags.append(_flag("detector_limits", FlagLevel.info,
                           f"The detector finds about {DETECTOR_TEST_RECALL:.0%} of marked lesions on test images; "
                           "the count is likely an undercount."))
        if det.lesion_count == 0:
            flags.append(_flag("no_lesions_detected", FlagLevel.info, "No lesions detected. This does not rule out acne."))
        elif det.mean_confidence is not None and det.mean_confidence < config.LOW_DETECTION_CONFIDENCE:
            flags.append(_flag("low_detection_confidence", FlagLevel.caution,
                               f"Average detection confidence is low ({det.mean_confidence:.0%})."))

    sub = findings.subtype
    if not sub.model.available:
        flags.append(_flag("subtype_unavailable", FlagLevel.caution, "Acne type model is not available; no type prediction."))
    else:
        flags.append(_flag("subtype_closeup_training", FlagLevel.info,
                           "The acne type model learned from close-up skin photos; on full-face photos its prediction is less reliable."))
        if sub.top_probability < config.LOW_CLASSIFIER_CONFIDENCE:
            flags.append(_flag("low_subtype_confidence", FlagLevel.caution,
                               f"Acne type prediction is uncertain ({sub.top_probability:.0%})."))
        if sub.top_label in LOW_DATA_SUBTYPES:
            flags.append(_flag("low_data_subtype", FlagLevel.caution,
                               f"'{sub.top_label}' had very few training images; treat this prediction as unreliable."))
        if sub.top_label == "cystic":
            priority = True
            flags.append(_flag("cystic_pattern", FlagLevel.caution, "Cystic pattern predicted; dermatologist review advised."))

    sev = findings.severity
    if not sev.model.available:
        flags.append(_flag("severity_unavailable", FlagLevel.caution, "Severity model is not available; no severity grade."))
    else:
        if sev.top_probability < config.LOW_CLASSIFIER_CONFIDENCE:
            flags.append(_flag("low_severity_confidence", FlagLevel.caution,
                               f"Severity grade is uncertain ({sev.top_probability:.0%})."))
        if sev.top_label in ("stage_3", "stage_4"):
            priority = True
            flags.append(_flag("high_severity", FlagLevel.caution, f"Predicted {sev.top_label.replace('_', ' ')}; dermatologist review advised."))
        else:
            higher = sum(p.probability for p in sev.probabilities if p.label in ("stage_3", "stage_4"))
            if higher >= HIGHER_SEVERITY_PROBABILITY:
                flags.append(_flag("possible_higher_severity", FlagLevel.caution,
                                   f"The severity model gives a {higher:.0%} chance of stage 3 or 4 and tends to under-grade, "
                                   "so this may be more severe than shown."))

    uncertain = sorted({f.code for f in flags} & UNCERTAINTY_FLAGS)
    if uncertain and not (urgent or priority):
        priority = True
        flags.append(_flag("uncertain_result", FlagLevel.caution,
                           "Some AI results are uncertain, so a dermatologist should review this before any treatment decision."))

    if urgent:
        recommendation = Recommendation.urgent_care
    elif priority:
        recommendation = Recommendation.priority_review
    else:
        recommendation = Recommendation.routine_review

    codes = {f.code for f in flags}
    parts = []
    if det.model.available:
        note = " (uncertain)" if "low_detection_confidence" in codes else ""
        parts.append(f"{det.lesion_count} lesion{'s' if det.lesion_count != 1 else ''} detected{note}")
    if sub.model.available:
        note = " (uncertain)" if codes & {"low_subtype_confidence", "low_data_subtype"} else ""
        parts.append(f"most likely type: {sub.top_label.replace('_', ' ')} ({sub.top_probability:.0%}){note}")
    if sev.model.available:
        note = " (uncertain)" if codes & {"low_severity_confidence", "possible_higher_severity"} else ""
        parts.append(f"estimated severity: {sev.top_label.replace('_', ' ')} ({sev.top_probability:.0%}){note}")
    summary = ("; ".join(parts) + ".") if parts else "No model results available."
    summary = summary[0].upper() + summary[1:]
    return flags, recommendation, summary
