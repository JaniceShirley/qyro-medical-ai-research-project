"""Layer 2: specialist vision models behind one interface.

Each model reports its own version and availability. A model whose weights are missing returns
available=False and no values, so nothing downstream can mistake a missing model for a result.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

from qyro_app import config
from qyro_app.schemas import Box, ClassProbability, Classification, Detection, ModelInfo, VisionFindings

# Label order must match SUBTYPE_MAP / SEVERITY_MAP in scripts/training/train_subtype.py and train_severity.py
SUBTYPE_LABELS = ["open_comedo", "closed_comedo", "papular", "pustular", "cystic", "mixed", "scar", "infantile", "mechanica"]
SEVERITY_LABELS = ["stage_1", "stage_2", "stage_3", "stage_4"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _model_version(path: Path) -> str:
    return path.stem


def ordinal_to_stage_probabilities(sigmoids: np.ndarray) -> np.ndarray:
    """Turn K-1 cumulative probabilities P(stage > k) into K per-stage probabilities."""
    cumulative = np.concatenate([[1.0], np.minimum.accumulate(sigmoids), [0.0]])
    probs = np.clip(cumulative[:-1] - cumulative[1:], 0.0, None)
    total = probs.sum()
    return probs / total if total > 0 else np.full(len(probs), 1.0 / len(probs))


class VisionModels:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self.detector = self.subtype = self.severity = None
        self.device = "cpu"

    def _load(self):
        with self._lock:
            if self._loaded:
                return
            import torch
            import timm
            from ultralytics import YOLO

            self.device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
            if config.DETECTOR_WEIGHTS.exists():
                self.detector = YOLO(str(config.DETECTOR_WEIGHTS))
            for attr, path, outputs in (
                ("subtype", config.SUBTYPE_WEIGHTS, len(SUBTYPE_LABELS)),
                ("severity", config.SEVERITY_WEIGHTS, len(SEVERITY_LABELS) - 1),
            ):
                if path.exists():
                    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=outputs)
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                    setattr(self, attr, model.to(self.device).eval())
            self._loaded = True

    def _classifier_input(self, image_bgr):
        # Must match val_transform in scripts/training/train_subtype.py / train_severity.py exactly:
        # PIL antialiased resize differs a lot from cv2 resize on large photos and changes predictions.
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        return transform(pil_image).unsqueeze(0).to(self.device)

    def _detect(self, image_bgr) -> Detection:
        info = ModelInfo(name="YOLOv8s lesion detector", version=_model_version(config.DETECTOR_WEIGHTS), available=self.detector is not None)
        if self.detector is None:
            info.note = "Detector weights not found."
            return Detection(model=info)
        result = self.detector.predict(
            image_bgr, conf=config.DETECTOR_CONF, iou=config.DETECTOR_IOU, imgsz=config.DETECTOR_IMGSZ,
            device=self.device, verbose=False,
        )[0]
        boxes = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), confidence=float(c))
            for b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ]
        mean_conf = float(np.mean([b.confidence for b in boxes])) if boxes else None
        return Detection(model=info, boxes=boxes, lesion_count=len(boxes), mean_confidence=mean_conf)

    def _classify(self, image_bgr, model, name, weights, labels, ordinal) -> Classification:
        import torch

        info = ModelInfo(name=name, version=_model_version(weights), available=model is not None)
        if model is None:
            info.note = "Model weights not found; no prediction made."
            return Classification(model=info)
        with torch.no_grad():
            logits = model(self._classifier_input(image_bgr)).cpu().numpy()[0]
        if ordinal:
            sigmoids = 1.0 / (1.0 + np.exp(-logits))
            probs = ordinal_to_stage_probabilities(sigmoids)
            top = int(np.sum(sigmoids > 0.5))  # same decoding as training/evaluation
        else:
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            top = int(np.argmax(probs))
        ranked = sorted(zip(labels, probs), key=lambda p: -p[1])
        return Classification(
            model=info,
            top_label=labels[top],
            top_probability=round(float(probs[top]), 4),
            probabilities=[ClassProbability(label=l, probability=round(float(p), 4)) for l, p in ranked],
        )

    def analyze(self, image_bgr: np.ndarray) -> VisionFindings:
        self._load()
        return VisionFindings(
            detection=self._detect(image_bgr),
            subtype=self._classify(image_bgr, self.subtype, "EfficientNet-B0 subtype classifier",
                                   config.SUBTYPE_WEIGHTS, SUBTYPE_LABELS, ordinal=False),
            severity=self._classify(image_bgr, self.severity, "EfficientNet-B0 ordinal severity grader",
                                    config.SEVERITY_WEIGHTS, SEVERITY_LABELS, ordinal=True),
        )


class MockVisionModels:
    """Fixed, clearly labelled fake outputs for UI development and tests. Never used for real results."""

    def analyze(self, image_bgr: np.ndarray) -> VisionFindings:
        h, w = image_bgr.shape[:2]
        boxes = [
            Box(x1=w * 0.30, y1=h * 0.40, x2=w * 0.34, y2=h * 0.44, confidence=0.82),
            Box(x1=w * 0.55, y1=h * 0.50, x2=w * 0.59, y2=h * 0.54, confidence=0.64),
            Box(x1=w * 0.45, y1=h * 0.65, x2=w * 0.48, y2=h * 0.68, confidence=0.31),
        ]
        mock = lambda name: ModelInfo(name=name, version="mock", available=True, note="MOCK OUTPUT, not a real prediction")
        return VisionFindings(
            detection=Detection(model=mock("mock detector"), boxes=boxes, lesion_count=3,
                                mean_confidence=round(sum(b.confidence for b in boxes) / 3, 4)),
            subtype=Classification(model=mock("mock subtype"), top_label="papular", top_probability=0.71,
                                   probabilities=[ClassProbability(label="papular", probability=0.71),
                                                  ClassProbability(label="pustular", probability=0.19),
                                                  ClassProbability(label="open_comedo", probability=0.10)]),
            severity=Classification(model=mock("mock severity"), top_label="stage_2", top_probability=0.55,
                                    probabilities=[ClassProbability(label="stage_2", probability=0.55),
                                                   ClassProbability(label="stage_1", probability=0.30),
                                                   ClassProbability(label="stage_3", probability=0.15)]),
        )


def get_vision_models():
    return MockVisionModels() if config.VISION_MODE == "mock" else VisionModels()
