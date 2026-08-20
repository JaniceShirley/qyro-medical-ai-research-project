# QYRO-Medical-AI
### AI-Powered Clinical Acne Lesion Detection & Severity Assessment

---

[![Python Version](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20%2F%20EfficientNet-brightgreen?logo=pytorch&logoColor=white)](#)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI%20%2F%20Dermatology-red)](#)
[![Research Status](https://img.shields.io/badge/Research-Active%20R%26D-orange)](#)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20CC%20BY--SA%204.0-lightgrey)](#)


> [!IMPORTANT]
> **HANDOVER DOCUMENTATION AVAILABLE**: For a detailed breakdown of all past work, algorithm specifications, failure audits, and the step-by-step roadmap to boost mAP@50 from **69.40% to 75.0%+**, please refer directly to **[PROJECT_HANDOVER.md](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/PROJECT_HANDOVER.md)**.

---

## 🔬 1. Project Overview

Acne Vulgaris is one of the most prevalent skin conditions globally, affecting over 85% of adolescents and young adults. However, clinical acne diagnosis and grading remain highly challenging due to:
* **Subjective Evaluation:** Inter-observer variability among dermatologists when grading acne severity using global scales (e.g., Investigator's Global Assessment).
* **Lesion-Level Heterogeneity:** The co-existence of different lesion subtypes (open/closed comedones, papules, pustules, cysts, and nodules) in varying densities and locations.
* **Demographic Bias:** High diagnostic error rates in computer vision models when generalizing across diverse skin tones (Fitzpatrick Skin Types I–VI and Monk Skin Tones).

**QYRO Medical AI** is an academic-companion research project dedicated to building a high-fidelity, explainable, and multi-stage computer vision pipeline to automate **lesion-level detection** and **global severity assessment** for acne. 

By separating diagnostic prediction into automated bounding box detection (lesion localization) and fine-grained classification (lesion subtype grading), QYRO aims to deliver clinical decision-support systems that are highly interpretable, demographically audited, and traceable back to original dermatologist-vetted reference standards.

---

## 🎯 2. Current Research Goals

Our active R&D focus is centered on the following clinical and engineering objectives:
1. **Lesion Detection & Counting:** Automating the localization and counting of acne lesions using robust bounding box detectors (YOLO backbones).
2. **Clinical Severity Grading:** Calibrating ordinal multi-class classification networks (EfficientNet backbones) to align with standardized clinical stages.
3. **Multi-Dataset Standardization:** Harmonizing disparate public and academic datasets under a single unified metadata schema.
4. **Annotation Quality Assurance:** Auditing coordinate ranges and label categories to eliminate label noise and prevent cross-split train/test data leakage.
5. **Algorithmic Equity (Bias Mitigation):** Evaluating and tuning model robustness across diverse skin tone cohorts (leveraging the Google SCIN and Fitzpatrick17k datasets).
6. **Explainable AI (XAI):** Ensuring model transparency by tracing clinical predictions directly to visible, quantified bounding box counts and lesion density maps.

---

## 📈 3. Current Development Status

The development of the data engineering pipelines and model experiments is tracked against the following checklist:

* [x] **Dataset Factory v1.0** — Standardized Python ingestion suite.
* [x] **Clinical Mapping Engine** — Standardized cross-dataset label harmonization.
* [x] **Annotation Audit Engine** — Validation of coordinates and label bounds.
* [x] **Image Quality Engine** — Automated blur detection and brightness auditing.
* [x] **YOLO Agreement Validation** — Image-label coordinate matching checks.
* [x] **Candidate Dataset Generation** — Automated data preparation.
* [x] **Dataset Versioning** — Checksum manifest-based version freezing.
* [x] **Provenance Tracking** — Audit trails from raw zip downloads to processed links.
* [ ] **Multi-Dataset Joint Training** — Jointly training detectors on aggregated datasets.
* [ ] **Ensemble Models** — Fusing detection backbones with fine-grained classifiers.
* [ ] **Clinical Validation Study** — Prospective validation with medical partners.
* [ ] **External Benchmark Evaluations** — Auditing performance on blind clinical trials.
* [ ] **Production API Integration** — ONNX compilation and containerized API deployment.

---

## 🔄 4. Dataset Pipeline

The QYRO Dataset Factory enforces a rigid, step-by-step pipeline to transform raw, noisy clinical images into standardized, leakage-free training pools:

```text
  ┌───────────────┐
  │  Raw Dataset  │  <-- Downloaded and staged under datasets/skin/acne/raw/
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  Import Gate  │  <-- Verified against registry/dataset_registry.csv
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  Anno Audit   │  <-- Annotation Audit Engine flags coordinate/index errors
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Clinical Map  │  <-- Clinical Mapping Engine standardizes category keys
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Quality Gate  │  <-- Image Quality Engine checks blur (Laplacian) & exposure
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  YOLO Check   │  <-- YOLO Agreement Validation ensures image-label matching
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Deduplication │  <-- Perceptual dHash comparison removes identical/near-duplicates
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Candidate Gen │  <-- Isolates high-quality, verified image subsets
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Merged Pool   │  <-- Constructs task-specific pools under datasets/skin/acne/final/
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Model Train   │  <-- Execution of reproducible PyTorch/Ultralytics runs
  └───────────────┘
```

---

## 📊 5. Supported Public Datasets

The QYRO pipeline ingests and maps the following public dermatological and acne datasets:

| Dataset | Status | Raw Images | Cleaned Images | License / Intended Use | Primary Task Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **DS001 (Kurnaz YOLOv8)** | Completed | 927 | 520 | Apache 2.0 | Lesion Bounding Box Detection |
| **DS002 (Tiswan Acne)** | Completed | 4,617 | 2,632 | Apache 2.0 | Fine-Grained Subtype Classification |
| **DS003 (DermNet NZ)** | Completed | 1,152 | 325 | Non-commercial Educational | Clinical Reference Atlas Anchor |
| **DS004 (ACNE04 v2)** | *Pending* | ~1,204 | - | Academic Research Only | Joint Bbox & Severity Grading |
| **DS005 (Google SCIN)** | *Pending* | 205 | - | SCIN Data Use License | Diverse Skin Tone Robustness |

*Note: Pending datasets are undergoing compliance reviews and metadata integration.*

---

## ⚙️ 6. Dataset Factory

The **Dataset Factory** is the core data-engineering engine of this workspace. It exists to guarantee absolute clinical data integrity before any training begins:
* **Quality Assurance:** Rejects blurry, out-of-focus, or severely overexposed images using Laplacian variance thresholds (`variance < 20`) and HSV brightness analysis.
* **Deduplication:** Runs perceptual diff-hashing (dHash) with a Hamming distance threshold of `≤ 6` to eliminate near-identical copies, ensuring **0.0% cross-split leakage**.
* **Annotation Validation:** Clips coordinates to standardized `[0.0, 1.0]` boundaries and maps class IDs to harmonized medical labels.
* **Provenance Tracking:** Logs SHA256 checksums of all processed files to [sha256_checksums.csv](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/final/metadata/sha256_checksums.csv), creating a complete audit trail.

---

## 🛡️ 7. Medical AI Principles

We adhere to rigorous Medical AI design principles to ensure that our models are safe, fair, and traceably validated:
* **Human-in-the-loop:** Model predictions are structured to support, not replace, clinical decisions by dermatologists.
* **Dataset Provenance:** Full documentation of dataset origins, versioning, and licenses to ensure medical auditability.
* **Clinical Traceability:** Mapping predictions to quantifiable lesion counts and localized regions rather than "black-box" classifications.
* **Reproducibility:** Versioning training configurations, data splits, and seeds alongside model code.
* **Responsible AI:** Routine equity audits across dark skin tones to prevent diagnostic disparities.

---

## 🗺️ 8. Research Roadmap

The development of this research workspace is divided into logical, progressive phases:
1. **Phase 1 (Acne v1):** Standardizing data engineering on acne datasets (Kurnaz, Tiswan, ACNE04). **[Completed]**
2. **Phase 2 (Pigmentation):** Extending pipelines to pigmentary disorders (Melasma) with Fitzpatrick type balancing. **[Planned]**
3. **Phase 3 (Inflammatory):** Integrating eczema and psoriasis datasets with segmentation boundaries. **[Planned]**
4. **Phase 4 (Hair & Scalp):** Ingesting alopecia and male pattern baldness datasets under micro-imaging. **[Planned]**
5. **Phase 5 (Clinic Telemetry):** Integrating clinic-partner telemetry and executing prospective evaluations. **[Planned]**

---

## 📁 9. Repository Structure

```text
QYRO-Medical-AI/
├── configs/            # YAML templates for training and data preprocessing
├── datasets/           # Directory structure for raw, cleaned, and merged pools (Git-ignored)
├── docs/               # Research blueprints, guidelines, and feasibility studies
├── experiments/        # Training logs, checkpoints, and active run folders (Git-ignored)
├── registry/           # Central registries (dataset_registry.csv, master_acne_registry.csv)
├── reports/            # Forensic audits, threshold sweeps, and metrics reports
├── notebooks/          # Exploratory Data Analysis (EDA) and visualization scripts
└── scripts/            # Dataset Factory engines, training execution, and utilities
```

---

## 📊 10. Current Metrics & Benchmarks

The master registries and validation reports detail the following project metrics:

### Data Engineering Summary:
* **Total Audited Raw Images:** 8,307
* **Total Accepted Candidates:** 4,434 (Master Registry)
* **Final Consolidated Merged Pool:** 4,428 images
  * *Lesion Detection Pool:* 520 images (Kurnaz)
  * *Subtype Classification Pool:* 2,951 images (Tiswan & DermNet)
  * *Severity Grading Pool:* 752 images (ACNE04)
  * *Robustness Holdout Pool:* 205 images (SCIN)
* **Deduplication Leakage Pairs:** 0 (100% split segregation)

### YOLOv8s Lesion Detector Performance:
* **Best Validation mAP50:** **0.6940** (Exceeding the 0.680 baseline target)
* **Best Validation Precision:** **0.6860**
* **Best Validation Recall:** **0.6400** (Deployment-adjusted conf=0.25 yields **Recall=0.6498**)
* **Inference Speed:** **3.7ms** per image (RTX 5050 Laptop GPU, fused batch evaluation)

---

## 📚 11. Publications & References

This project draws architectural and benchmarking inspiration from the following publications:
1. **AcneAI (MICCAI 2024):** *"Deep Learning Ensembles for Multi-Stage Lesion Detection and Severity Grading in Clinical Settings."*
2. **ACNE04 (ICCV 2019):** Wu, X. et al. *"Joint Acne Image Grading and Lesion Counting via Label Distribution Learning."* [Paper Link](https://arxiv.org/abs/1903.04104).
3. **Google SCIN (2024):** *"A Diverse Crowdsourced Dataset of Skin Conditions Representing Fitzpatrick Types I-VI."* [Google Research](https://github.com/google-research-datasets/scin).

---

## ⚠️ 12. Clinical Disclaimer

> [!CAUTION]
> **RESEARCH USE ONLY. NOT A DIAGNOSTIC DEVICE.**
> The models, weights, and configurations documented in this repository are developed for **academic research, scientific benchmarking, and decision-support modeling**. 
> They are **not** cleared by the FDA or other regulatory bodies for clinical diagnosis. 
> All final diagnostic assessments, clinical choices, and treatment decisions must remain with a licensed healthcare professional.
