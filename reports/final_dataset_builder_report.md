# QYRO Medical AI - Unified Dataset Builder Report
**Phase 6B: Centralized Dataset Consolidation & Validation**

---

## 1. Executive Summary

This report documents the assembly of all cleaned acne datasets into the single train-ready target directory structure **`datasets/skin/acne/final/`**. 

Using a **hardlinks-first, copy-fallback** linking strategy, the dataset builder pooled all images and labels, preserved original file formats and extensions, computed SHA256 checksums, and built task-focused directories.

* **Total Linked/Copied Images:** 4428
* **Linking Method Utilized:** Hardlinks: 4428 | Copies: 0
* **Detection Dataset Size:** 520 images (Kurnaz)
* **Subtype Classification Dataset Size:** 2951 images (Tiswan & DermNet pooled)
* **Severity Dataset Size:** 752 images (ACNE04)
* **Robustness Holdout Dataset Size:** 205 images (Google SCIN)
* **Excluded Images:** 6 images (DermNet `primary_lesion` and `unknown` subtype images excluded from subtype classification directories)
* **Total Files Hashed for Integrity:** 4959 files (saved to `final/metadata/sha256_checksums.csv`)
* **Split Contamination / Leakage Check:** **PASSED** (0 leakages detected)
* **Registry Integrity Validation:** **PASSED** (0 missing files, 0 duplicate IDs found)
* **Readiness Assessment:** **READY FOR MODEL TRAINING**

---

## 2. Directory Structure Verification

The builder constructed the target structure successfully:

- `final/detection/` $ightarrow$ YOLOv8 structure containing train, valid, and test sets.
- `final/subtype_classification/` $ightarrow$ 9 allowed subtype subfolders across train, valid, and test splits.
- `final/severity/` $ightarrow$ 4 severity stage subfolders across train, valid, and test splits.
- `final/robustness_holdout/images/` $ightarrow$ flat folder containing the out-of-distribution evaluation set.
- `final/metadata/` $ightarrow$ manifests (`subtype_manifest.csv`, `severity_manifest.csv`, `robustness_manifest.csv`, `dataset_summary.json`, `sha256_checksums.csv`) and dataset breakdown JSONs.
- `final/registry/` $ightarrow$ snapshot copy of the master registry CSV.

---

## 3. Subtype & Severity Balance (Task 2 & 3)

### 3.1 Subtype Classification Pool (`subtype_manifest.csv`)
- **Total Images:** 2951
- **Tiswan Contribution:** 2632 images
- **DermNet Contribution:** 319 images
- **Subtype Balance Table:**
  - `open_comedo`: 751 images
  - `closed_comedo`: 168 images
  - `papular`: 634 images
  - `pustular`: 627 images
  - `cystic`: 743 images
  - `scar`: 11 images
  - `infantile`: 16 images
  - `mechanica`: 1 images

### 3.2 Severity Pool (`severity_manifest.csv`)
- **Total Images:** 752 (ACNE04)
- **Severity Balance Table:**
  - `stage_1` (Level 0): 271 images
  - `stage_2` (Level 1): 340 images
  - `stage_3` (Level 2): 89 images
  - `stage_4` (Level 3): 52 images

### 3.3 Subtype Source Proportions Verification
The split percentages for each source dataset (`tiswan` vs `dermnet`) within each subtype class are preserved:

| Subtype | Source | Train | Valid | Test | Total | Split Distribution (Train / Val / Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `closed_comedo` | `dermnet` | 23 | 5 | 6 | 34 | 67.6% / 14.7% / 17.6% |
| `closed_comedo` | `tiswan` | 93 | 20 | 21 | 134 | 69.4% / 14.9% / 15.7% |
| `cystic` | `dermnet` | 72 | 15 | 17 | 104 | 69.2% / 14.4% / 16.3% |
| `cystic` | `tiswan` | 447 | 96 | 96 | 639 | 70.0% / 15.0% / 15.0% |
| `infantile` | `dermnet` | 11 | 2 | 3 | 16 | 68.8% / 12.5% / 18.8% |
| `mechanica` | `dermnet` | 0 | 0 | 1 | 1 | 0.0% / 0.0% / 100.0% |
| `open_comedo` | `dermnet` | 46 | 9 | 11 | 66 | 69.7% / 13.6% / 16.7% |
| `open_comedo` | `tiswan` | 479 | 103 | 103 | 685 | 69.9% / 15.0% / 15.0% |
| `papular` | `dermnet` | 16 | 3 | 5 | 24 | 66.7% / 12.5% / 20.8% |
| `papular` | `tiswan` | 427 | 91 | 92 | 610 | 70.0% / 14.9% / 15.1% |
| `pustular` | `dermnet` | 44 | 9 | 10 | 63 | 69.8% / 14.3% / 15.9% |
| `pustular` | `tiswan` | 394 | 85 | 85 | 564 | 69.9% / 15.1% / 15.1% |
| `scar` | `dermnet` | 7 | 1 | 3 | 11 | 63.6% / 9.1% / 27.3% |

---

## 4. Verification Check Summary

| Check Task | Status | Failures | Description |
| :--- | :---: | :---: | :--- |
| **Split Leakage Check** | Passed | 0 | Checks that no image hashes cross-contaminate train/valid/test splits. |
| **Path Integrity Check** | Passed | 0 | Verifies all final dataset image paths exist physically on disk. |
| **Unique ID Check** | Passed | 0 | Verifies all file basenames align uniquely with registry image IDs. |

---

## 5. Dataset Readiness Assessment
The compiled backbone represents **Phase 6B Completion**. All directories, assets, manifests, and snapshots are populated, structured, and validated. The final dataset is 100% ready for model training runs in the next phase.
