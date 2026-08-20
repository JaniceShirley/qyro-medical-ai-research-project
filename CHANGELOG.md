# Changelog

All notable changes to the **QYRO Medical AI** dataset pipelines, configuration schemas, and validation experiments are documented in this file.

---

## [1.0.0] - 2026-07-02

### Added
* **Dataset Factory v1.0:** Established the central Python processing suite for raw data ingestion:
  * **Clinical Mapping Engine:** Standardizes heterogeneous raw directory structures into harmonized clinical labels.
  * **Annotation Audit Engine:** Forensically audits bounding box coordinates and class indexes.
  * **Image Quality Engine:** Measures image blur (Laplacian variance) and exposure levels (brightness checks).
  * **Deduplication Engine:** Leverages MD5 matching and perceptual dHash clustering to filter out online augmentations and near-duplicates.
* **Dataset Integration:**
  * **DS001 (Kurnaz YOLOv8):** Cleaned, deduplicated, and split (364 train, 78 val, 78 test images). Added SHA256 checksum tracking.
  * **DS002 (Tiswan classification):** Cleaned and split (2,632 images).
  * **DS003 (DermNet NZ reference):** Sanitized 325 clinical reference atlas images.
  * **DS005 (Google SCIN subset):** Extracted and verified 205 images with Monk Skin Tone labels for robustness evaluation.
* **Model Calibration & Training Logs:**
  * **YOLOv8s Convergence Baseline:** Logged details of the Phase 7D.3 recovery experiment achieving validation **mAP50 = 0.694** and **Precision = 0.686**.
  * **Severity Ordinal Loss:** Integrated `ordinal_cross_entropy` loss configuration in `severity_config.yaml` for EfficientNet-B0.

### Fixed
* **Cross-Split Leakage:** Corrected 56 instances of near-duplicate image contamination across train/validation splits in the raw Kurnaz dataset. The reconstructed v1 dataset features **0.0% cross-split leakage**.
* **Annotation Coordinate Clipping:** Repaired out-of-bound coordinates in the raw YOLO bounding box labels.

---

## [0.5.0] - 2026-06-05

### Added
* **Prototype Workspace:** Initial directory layouts for Phase 1 (Acne).
* **Registry System:** Created `registry/dataset_registry.csv` and `registry/master_acne_registry.csv` schemas.
* **Baseline training scripts:** Prototyped `train_detection.py`, `train_subtype.py`, and `train_severity.py`.
