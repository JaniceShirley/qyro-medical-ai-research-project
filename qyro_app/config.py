"""Central settings for the QYRO demo app. Thresholds live here so rules stay testable."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "qyro.findings.v1"

# "real" loads the trained weights; "mock" returns fixed fake outputs for UI work and tests.
VISION_MODE = os.environ.get("QYRO_VISION_MODE", "real").lower()

DETECTOR_WEIGHTS = REPO_ROOT / "models/production/qyro_acne_v1_best.pt"
SUBTYPE_WEIGHTS = REPO_ROOT / "models/production/qyro_subtype_v1_best.pt"
SEVERITY_WEIGHTS = REPO_ROOT / "models/production/qyro_severity_v1_best.pt"

# Detector settings from models/production/production_config.yaml
DETECTOR_CONF = 0.25
DETECTOR_IOU = 0.60
DETECTOR_IMGSZ = 640

# Upload limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Image quality gate (Laplacian blur variance, mean brightness 0-255, shortest side in px)
BLUR_REJECT_BELOW = 20.0  # same cutoff used by the dataset cleaning pipeline
BLUR_CAUTION_BELOW = 60.0
DARK_REJECT_BELOW = 35.0
DARK_CAUTION_BELOW = 60.0
BRIGHT_REJECT_ABOVE = 235.0
BRIGHT_CAUTION_ABOVE = 210.0
MIN_SIDE_REJECT_BELOW = 224
MIN_SIDE_CAUTION_BELOW = 480
# Out-of-scope check: share of pixels in a YCrCb skin-colour range. Measured on 2026-10-21:
# 0.5% of 3,703 real dataset photos fall below 0.05, no Fitzpatrick group (FST1-6, SCIN holdout) does,
# while sky/grass/documents/screens score 0.00 and random noise 0.09.
SKIN_REJECT_BELOW = 0.05
SKIN_CAUTION_BELOW = 0.20
SKIN_YCRCB_LOW = (0, 133, 77)
SKIN_YCRCB_HIGH = (255, 173, 127)

# Uncertainty thresholds used by the rules layer
LOW_DETECTION_CONFIDENCE = 0.40  # mean box confidence below this is flagged
LOW_CLASSIFIER_CONFIDENCE = 0.60  # top-class probability below this is flagged
