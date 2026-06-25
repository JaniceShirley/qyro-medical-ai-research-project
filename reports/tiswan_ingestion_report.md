# QYRO Medical AI - Tiswan Acne Dataset Ingestion Report
**Phase 2A: Controlled Dataset Ingestion & Structure Verification**

---

## 1. Executive Summary

This report documents the ingestion and verification of the **Tiswan Acne Dataset** inside `datasets/skin/acne/raw/tiswan/`. 

This dataset serves as the primary large-scale training source for QYRO's fine-grained acne classification and severity assessment backbones. The raw dataset has been successfully staged in a read-only state.

* **Total Raw Images:** 4,617
* **Discovered Categories (Classes):** 5 (`Blackheads`, `Cyst`, `Papules`, `Pustules`, `Whiteheads`)
* **Annotation Format:** One-hot multi-class folder organization & metadata CSV mapping
* **Ingestion Status:** **SUCCESSFUL (PENDING AUDIT)**

---

## 2. Ingested Folder Structure

The staged raw folder structure was verified and contains the following layout:

```
datasets/skin/acne/raw/tiswan/
├── original_download/        # Compressed archive and backup files (immutable)
├── extracted/                # Extracted dataset files (read-only)
│   └── AcneDataset/
│       ├── train/
│       │   ├── Blackheads/   # 2,778 total train images across 5 folders
│       │   ├── Cyst/
│       │   ├── Papules/
│       │   ├── Pustules/
│       │   ├── Whiteheads/
│       │   └── _train_classes.csv
│       ├── valid/
│       │   ├── Blackheads/   # 921 total validation images
│       │   ├── Cyst/
│       │   ├── Papules/
│       │   ├── Pustules/
│       │   ├── Whiteheads/
│       │   └── _valid_classes.csv
│       └── test/
│           ├── Blackheads/   # 918 total test images
│           ├── Cyst/
│           ├── Papules/
│           ├── Pustules/
│           ├── Whiteheads/
│           └── _test_classes.csv
└── notes.txt                 # Ingestion notes and context
```

---

## 3. Dataset Splits & Category Distribution

Our basic file discovery scanned 4,617 images across the pre-defined splits. Each split contains files distributed among five fine-grained acne categories:

| Acne Category (Subtype) | Train Count | Valid Count | Test Count | Total Raw Count | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Blackheads** | 2,778 total | 921 total | 918 total | 4,617 total | Open comedones |
| **Cyst** | *TBD* | *TBD* | *TBD* | *TBD* | Severe nodulocystic lesions |
| **Papules** | *TBD* | *TBD* | *TBD* | *TBD* | Inflammatory red bumps |
| **Pustules** | *TBD* | *TBD* | *TBD* | *TBD* | Pus-filled inflammatory lesions |
| **Whiteheads** | *TBD* | *TBD* | *TBD* | *TBD* | Closed comedones |

*Note: Individual category counts per split will be extracted and audited in Phase 2B (Forensic Audit). The basic file scan validated that all splits are populated and contain zero empty category folders.*

---

## 4. Annotation Format & Availability

* **Category-level Labels:** The dataset relies on folder structures for categorical labels. Placing an image in `train/Blackheads` implicitly assigns it the `Blackheads` label.
* **Metadata CSV Mapping:** Each split contains a class mapping CSV file (e.g., `_train_classes.csv`). This file maps the image filename to one-hot columns:
  * Columns: `filename`, `Blackheads`, `Cyst`, `Papules`, `Pustules`, `Whiteheads`
  * Values are binary (`1` for membership, `0` for non-membership).
* **Object Detection Coordinates:** This dataset is **classification-only** and does not contain bounding box (`.txt` or `xml`) annotations for individual lesion coordinates.

---

## 5. Immediate Quality & Integrity Risks

* **Image Formatting and Decodability:** Checked 4,617 files; 0 corrupt files or unsupported extensions found.
* **Roboflow Augmentation Naming:** Filenames exhibit Roboflow's custom string suffix schema (e.g., `_jpg.rf.[hash].jpg`). This strongly suggests that offline augmentations (rotations, brightness, color shifts) are already present and mixed into the splits.
* **Split Leakage Risk (CRITICAL):** Since the dataset contains augmented duplicates, there is a very high probability that augmented versions of the same original image are shared across `train`, `valid`, and `test` splits.
* **Demographic Bias:** Images appear to be web-scraped consumer smartphone photos, showing close-ups of facial skin. Demographic makeup, skin-type representation, and device capture consistency are unknown and require audit review.

---

## 6. Recommendation for Audit Readiness

### Status: READY FOR FORENSIC AUDIT

The Tiswan Acne Dataset has been successfully ingested with all folder pathways verified. The database registry is updated. We recommend proceeding immediately to **Phase 2B (Forensic Audit)** to:
1. Run a dHash perceptual duplicate scan to identify duplicate groups.
2. Check for train-validation-test leakage.
3. Compute image statistics (resolutions, aspect ratios, blur distribution).
4. Inspect class balance metrics.
