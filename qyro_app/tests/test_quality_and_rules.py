import cv2
import numpy as np
import pytest

from qyro_app import quality, rules
from qyro_app.schemas import (
    Classification, ClassProbability, Detection, ModelInfo, QualityStatus, Questionnaire, Recommendation, VisionFindings,
)
from qyro_app.vision import ordinal_to_stage_probabilities


def textured_image(size=640, brightness=128):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.5, np.full_like(img, brightness), 0.5, 0)
    return img


def test_sharp_well_lit_image_passes():
    assert quality.assess_quality(textured_image()).status == QualityStatus.passed


def test_blurry_image_is_rejected():
    blurry = cv2.GaussianBlur(textured_image(), (51, 51), 20)
    result = quality.assess_quality(blurry)
    assert result.status == QualityStatus.reject
    assert {c.name for c in result.checks if c.status == QualityStatus.reject} == {"sharpness"}


def test_dark_image_is_rejected():
    dark = (textured_image() * 0.1).astype(np.uint8)
    assert quality.assess_quality(dark).status == QualityStatus.reject


def test_tiny_image_is_rejected():
    assert quality.assess_quality(textured_image(size=150)).status == QualityStatus.reject


def test_ordinal_probabilities_sum_to_one_and_follow_thresholds():
    probs = ordinal_to_stage_probabilities(np.array([0.9, 0.2, 0.05]))
    assert probs.sum() == pytest.approx(1.0)
    assert int(np.argmax(probs)) == 1  # P(>stage1)=0.9, P(>stage2)=0.2 -> stage_2 most likely


def model(available=True):
    return ModelInfo(name="m", version="v", available=available)


def findings(subtype="papular", subtype_p=0.9, severity="stage_1", severity_p=0.9, count=5, mean_conf=0.7, available=True):
    def cls(label, p):
        if not available:
            return Classification(model=model(False))
        return Classification(model=model(), top_label=label, top_probability=p,
                              probabilities=[ClassProbability(label=label, probability=p)])
    return VisionFindings(
        detection=Detection(model=model(), boxes=[], lesion_count=count, mean_confidence=mean_conf),
        subtype=cls(subtype, subtype_p),
        severity=cls(severity, severity_p),
    )


def passing_quality():
    return quality.assess_quality(textured_image())


def codes(flags):
    return {f.code for f in flags}


def test_clean_case_is_routine_review():
    flags, rec, summary = rules.evaluate(passing_quality(), findings(), Questionnaire())
    assert rec == Recommendation.routine_review
    assert "5 lesions detected" in summary
    assert {"detector_limits", "subtype_closeup_training"} <= codes(flags)


def test_fever_escalates_to_urgent_even_with_rejected_photo():
    bad = quality.assess_quality(textured_image(size=150))
    flags, rec, _ = rules.evaluate(bad, None, Questionnaire(fever_or_unwell=True))
    assert rec == Recommendation.urgent_care
    assert {"fever_or_unwell", "image_rejected"} <= codes(flags)


def test_rejected_photo_without_red_flags_asks_for_retake():
    bad = quality.assess_quality(textured_image(size=150))
    _, rec, summary = rules.evaluate(bad, None, Questionnaire())
    assert rec == Recommendation.retake_photo
    assert "no AI analysis" in summary


@pytest.mark.parametrize("answers", [{"painful_lumps": True}, {"scarring": True}, {"rapid_spread": True}, {"age_years": 9}])
def test_reported_red_flags_raise_priority(answers):
    _, rec, _ = rules.evaluate(passing_quality(), findings(), Questionnaire(**answers))
    assert rec == Recommendation.priority_review


def test_unknown_answers_do_not_trigger_flags():
    flags, rec, _ = rules.evaluate(passing_quality(), findings(), Questionnaire())
    assert rec == Recommendation.routine_review
    assert not codes(flags) & {"painful_lumps", "scarring", "rapid_spread", "fever_or_unwell"}


def test_cystic_or_high_severity_raise_priority():
    assert rules.evaluate(passing_quality(), findings(subtype="cystic"), Questionnaire())[1] == Recommendation.priority_review
    assert rules.evaluate(passing_quality(), findings(severity="stage_4"), Questionnaire())[1] == Recommendation.priority_review


def test_low_confidence_and_low_data_predictions_are_flagged():
    flags, _, _ = rules.evaluate(passing_quality(), findings(subtype="scar", subtype_p=0.4, severity_p=0.3, mean_conf=0.3), Questionnaire())
    assert {"low_subtype_confidence", "low_data_subtype", "low_severity_confidence", "low_detection_confidence"} <= codes(flags)


def test_missing_models_report_unavailable_and_never_invent_values():
    flags, _, summary = rules.evaluate(passing_quality(), findings(available=False), Questionnaire())
    assert {"subtype_unavailable", "severity_unavailable"} <= codes(flags)
    assert "type" not in summary and "severity" not in summary
