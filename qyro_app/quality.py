"""Layer 1: image quality gate. Measurable checks only, each with a pass/caution/reject result."""
import cv2
import numpy as np

from qyro_app import config
from qyro_app.schemas import ImageQuality, QualityCheck, QualityStatus


def _grade(value, reject, caution, higher_is_worse=False):
    if higher_is_worse:
        if value > reject:
            return QualityStatus.reject
        if value > caution:
            return QualityStatus.caution
        return QualityStatus.passed
    if value < reject:
        return QualityStatus.reject
    if value < caution:
        return QualityStatus.caution
    return QualityStatus.passed


def assess_quality(image_bgr: np.ndarray) -> ImageQuality:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    min_side = float(min(width, height))
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin = float(cv2.inRange(ycrcb, config.SKIN_YCRCB_LOW, config.SKIN_YCRCB_HIGH).mean() / 255)

    checks = [
        QualityCheck(
            name="skin_visible",
            status=_grade(skin, config.SKIN_REJECT_BELOW, config.SKIN_CAUTION_BELOW),
            value=round(skin, 3),
            message=(f"Only {skin:.0%} of the photo looks like skin. Please upload a photo of the affected skin."
                     if skin < config.SKIN_CAUTION_BELOW else f"{skin:.0%} of the photo looks like skin."),
        ),
        QualityCheck(
            name="resolution",
            status=_grade(min_side, config.MIN_SIDE_REJECT_BELOW, config.MIN_SIDE_CAUTION_BELOW),
            value=min_side,
            message=f"Shortest side is {int(min_side)} px.",
        ),
        QualityCheck(
            name="sharpness",
            status=_grade(blur, config.BLUR_REJECT_BELOW, config.BLUR_CAUTION_BELOW),
            value=round(blur, 1),
            message=f"Sharpness score {blur:.1f} (below {config.BLUR_REJECT_BELOW:.0f} is too blurry).",
        ),
        QualityCheck(
            name="too_dark",
            status=_grade(brightness, config.DARK_REJECT_BELOW, config.DARK_CAUTION_BELOW),
            value=round(brightness, 1),
            message=f"Average brightness {brightness:.0f}/255.",
        ),
        QualityCheck(
            name="too_bright",
            status=_grade(brightness, config.BRIGHT_REJECT_ABOVE, config.BRIGHT_CAUTION_ABOVE, higher_is_worse=True),
            value=round(brightness, 1),
            message=f"Average brightness {brightness:.0f}/255.",
        ),
    ]

    statuses = {c.status for c in checks}
    if QualityStatus.reject in statuses:
        overall = QualityStatus.reject
    elif QualityStatus.caution in statuses:
        overall = QualityStatus.caution
    else:
        overall = QualityStatus.passed
    return ImageQuality(status=overall, width=width, height=height, checks=checks)
