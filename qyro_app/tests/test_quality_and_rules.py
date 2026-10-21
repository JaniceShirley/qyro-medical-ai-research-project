import cv2
import numpy as np
import pytest

from qyro_app import quality, rules
from qyro_app.schemas import (
    Classification, ClassProbability, Detection, ModelInfo, QualityStatus, Questionnaire, Recommendation, VisionFindings,
)
from qyro_app.vision import ordinal_to_stage_probabilities


SKIN_BGR = (120, 150, 200)  # inside the YCrCb skin range used by the quality gate


def textured_image(size=640, base=SKIN_BGR, gray_texture=False):
    """Sharp texture on a base colour; skin-coloured by default so it passes every quality check."""
    rng = np.random.default_rng(0)
    noise = rng.integers(-30, 31, (size, size, 1 if gray_texture else 3))
    return np.clip(np.array(base) + noise, 0, 255).astype(np.uint8)


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


@pytest.mark.parametrize("base, gray", [((230, 160, 60), False), ((40, 160, 40), False), ((200, 200, 200), True)],
                         ids=["sky", "grass", "document"])
def test_non_skin_photo_is_out_of_scope(base, gray):
    result = quality.assess_quality(textured_image(base=base, gray_texture=gray))
    assert result.status == QualityStatus.reject
    assert {c.name for c in result.checks if c.status == QualityStatus.reject} == {"skin_visible"}
    flags, rec, _ = rules.evaluate(result, None, Questionnaire())
    assert rec == Recommendation.retake_photo
    assert "out_of_scope_image" in codes(flags)


def test_ordinal_probabilities_sum_to_one_and_follow_thresholds():
    probs = ordinal_to_stage_probabilities(np.array([0.9, 0.2, 0.05]))
    assert probs.sum() == pytest.approx(1.0)
    assert int(np.argmax(probs)) == 1  # P(>stage1)=0.9, P(>stage2)=0.2 -> stage_2 most likely


def model(available=True):
    return ModelInfo(name="m", version="v", available=available)


def findings(subtype="papular", subtype_p=0.9, severity="stage_1", severity_p=0.9, count=5, mean_conf=0.7, available=True,
             severity_others=None):
    def cls(label, p, others=None):
        if not available:
            return Classification(model=model(False))
        probs = [ClassProbability(label=label, probability=p)] + [ClassProbability(label=l, probability=q) for l, q in (others or {}).items()]
        return Classification(model=model(), top_label=label, top_probability=p, probabilities=probs)
    return VisionFindings(
        detection=Detection(model=model(), boxes=[], lesion_count=count, mean_confidence=mean_conf),
        subtype=cls(subtype, subtype_p),
        severity=cls(severity, severity_p, severity_others),
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


@pytest.mark.parametrize("kwargs", [{"subtype_p": 0.4}, {"severity_p": 0.3}, {"mean_conf": 0.3}, {"subtype": "scar"}])
def test_low_confidence_lowers_routine_to_priority_review(kwargs):
    flags, rec, summary = rules.evaluate(passing_quality(), findings(**kwargs), Questionnaire())
    assert rec == Recommendation.priority_review
    assert "uncertain_result" in codes(flags)
    assert "(uncertain)" in summary


def test_possible_under_graded_severity_escalates():
    f = findings(severity="stage_2", severity_p=0.6, severity_others={"stage_3": 0.3, "stage_1": 0.1})
    flags, rec, summary = rules.evaluate(passing_quality(), f, Questionnaire())
    assert rec == Recommendation.priority_review
    assert {"possible_higher_severity", "uncertain_result"} <= codes(flags)


def test_confident_results_stay_routine_without_uncertainty_flag():
    f = findings(severity="stage_2", severity_p=0.85, severity_others={"stage_3": 0.1, "stage_1": 0.05})
    flags, rec, summary = rules.evaluate(passing_quality(), f, Questionnaire())
    assert rec == Recommendation.routine_review
    assert "uncertain_result" not in codes(flags) and "(uncertain)" not in summary


def test_borderline_photo_quality_counts_as_uncertain():
    borderline = quality.assess_quality(textured_image(size=400))  # 400 px: caution, not reject
    assert borderline.status == QualityStatus.caution
    flags, rec, _ = rules.evaluate(borderline, findings(), Questionnaire())
    assert rec == Recommendation.priority_review and "uncertain_result" in codes(flags)


def test_missing_models_report_unavailable_and_never_invent_values():
    flags, _, summary = rules.evaluate(passing_quality(), findings(available=False), Questionnaire())
    assert {"subtype_unavailable", "severity_unavailable"} <= codes(flags)
    assert "type" not in summary and "severity" not in summary
