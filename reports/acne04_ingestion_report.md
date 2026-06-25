# QYRO Medical AI - ACNE04 Ingestion & Structural Verification Report
**Phase 3A: Forensic Dataset Quality & Structural Audit**

---

## 1. Executive Summary

This report documents the structural verification of the **ACNE04 dataset** staged inside `datasets/skin/acne/raw/acne04/extracted/acne_1024/`.

The primary objective was to verify the relationship between the master folder (`all_1024/`) and the individual severity subfolders (`acne0_1024/` through `acne3_1024/`), locate and verify the annotation files, and assess whether the dataset supports QYRO's clinical staging and lesion localization pipelines.

### Key Audit Metrics:
* **Total Scanned Images (in `all_1024`):** 1,406
* **Original Severity Folder Count:** 1,377 images
* **Overlap rate:** **100%** (every image in `acne0` to `acne3` is present in `all_1024`)
* **Extra Images in `all_1024`:** **29 images** (not present in individual folders)
* **Annotation Availability:** **None** (this version of the download is **classification-only** and does not contain bounding box coordinates).
* **Ingestion Status:** **SUCCESSFUL (PENDING CLEANING/DECISION)**
* **Reconstruction Recommendation:** **USE ALL_1024 ONLY**

---

## 2. Folder Meaning Explanation

An analysis of the directory hierarchy reveals the following folder purposes:

1. **`all_1024/` (1,406 images):** The master folder containing the entire set of high-resolution front-facing skin images. It serves as a unified repository.
2. **`acne0_1024/` (483 images):** Mild severity (Level 0) classification subset.
3. **`acne1_1024/` (623 images):** Moderate severity (Level 1) classification subset.
4. **`acne2_1024/` (175 images):** Severe severity (Level 2) classification subset.
5. **`acne3_1024/` (96 images):** Very Severe severity (Level 3) classification subset.
6. **`acne3_512_selection/` (16 images):** A small, downsampled selection of level 3 images.
7. **`small_1024/` (200 images):** A subset of 200 images with a creator's script (`main.py`) for generating text-to-image prompts.
8. **`small_1024_renamed/` (200 images):** A duplicate folder where substrings like `acne0` are renamed to text classes (e.g. `acnezero`, `acnesmall`).
9. **`sim_acne.csv` (1,402 rows):** A longitudinal clinical simulation file tracking patient severity trends over time.

---

## 3. Overlap Verification (Scientific Proof)

We computed exact MD5 hashes and filename overlaps for all images in the dataset to verify whether the individual severity folders are subsets of the master `all_1024/` directory.

### Overlap Breakdown:
* **`acne0_1024` Overlap:** 483 / 483 files (100.0% filename match, 100.0% MD5 hash match).
* **`acne1_1024` Overlap:** 623 / 623 files (100.0% filename match, 100.0% MD5 hash match).
* **`acne2_1024` Overlap:** 175 / 175 files (100.0% filename match, 100.0% MD5 hash match).
* **`acne3_1024` Overlap:** 96 / 96 files (100.0% filename match, 100.0% MD5 hash match).

### Summary of Folder Duplication:
The four individual severity directories contain a combined total of **1,377 images**, which are all **100% duplicate copies** of files present in `all_1024/`.

The master `all_1024/` folder contains **29 extra images** not found in the severity directories:
* **8 images** with the prefix `levle0_`
* **2 images** with the prefix `levle2_`
* **19 images** with the prefix `levle3_`

These extra images are legitimate severity-labeled images that were simply excluded from the subfolders. Therefore, **`all_1024/` is a merged master folder containing the entire dataset.**

---

## 4. Annotation Forensics & Format Explanation

> [!WARNING]
> **NO LESION COORDINATE LABELS PRESENT**
> We discovered that this download does not contain bounding box or center-radius coordinate annotations. 
> The `.jsonl` files (`metadata.jsonl` in `all_1024/` and `metadata_acne3.jsonl` in `acne3_1024/`) only contain text-prompt mapping tags:
> * Format: `{"file_name": "levle0_144.jpg", "prompt": "photo of a person with acne0"}`
> * Purpose: Prepared for HuggingFace text-to-image (Stable Diffusion) training metadata.

* **Can this be converted into YOLO later?**
  **NO.** Because there are no coordinate coordinates or lesion bounding boxes to extract.
* **Global Labels:** The only labels available are the global image-level severity grades (Level 0 to Level 3) which are represented by the filename prefixes (e.g. `levle0_`, `levle1_`, etc.).

---

## 5. Sufficiency of `all_1024/`

**YES. `all_1024/` alone is completely sufficient.**
Since all images in the severity folders are identical to those in `all_1024/`, and `all_1024/` actually contains 29 extra valid images, keeping the other directories is redundant. The individual severity folders can be safely ignored, and all pipeline work should focus on the master `all_1024/` directory.

---

## 6. Medical Value Assessment

Although the dataset does not contain lesion-level annotations, it is still valuable:

### A. Lesion Localization (Object Detection):
* **Support:** **Poor.** The dataset cannot train lesion detection models because it contains no coordinate boxes.

### B. Subtype Classification:
* **Support:** **Poor.** It does not identify specific acne subtypes (e.g., papules vs. cysts) on a lesion level.

### C. Severity Estimation & Staging:
* **Support:** **Excellent.** The 4-level severity labeling (`levle0_` to `levle3_`) corresponds to global clinical severity grading.
* **QYRO Stage Alignment:**
  * **Level 0 (Mild) $\rightarrow$ QYRO Stage 1** (comedonal acne, minimal inflammatory lesions)
  * **Level 1 (Moderate) $\rightarrow$ QYRO Stage 2** (papular inflammatory acne)
  * **Level 2 (Severe) $\rightarrow$ QYRO Stage 3** (pustular inflammatory acne)
  * **Level 3 (Very Severe) $\rightarrow$ QYRO Stage 4** (cystic/nodular acne)
  This makes it highly useful as a global validation benchmark for QYRO's severity grading model.

---

## 7. Risks & Mitigation

1. **Licensing:** The dataset is licensed under `CC BY 4.0` (which allows research and testing but should be segregated from proprietary production weights).
2. **Duplication Overhead:** Keeping the severity folders wastes ~340MB of disk space.
3. **Noisy Labels:** Global severity grading is subjective and does not represent local lesion characteristics.

---

## 8. Final Ingestion Recommendation

### Decision: USE ALL_1024 ONLY

We should **discard** the individual severity subdirectories and focus all subsequent data ingestion, cleaning, and testing scripts exclusively on **`all_1024/`**. This master folder provides the complete, unabridged image list (1,406 images) and is fully self-contained.
