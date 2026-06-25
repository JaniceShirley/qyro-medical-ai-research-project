# QYRO Medical AI - Kurnaz YOLOv8 Dataset Audit Report
**Phase 1B: Forensic Dataset Quality & Integrity Audit**

---

## 1. Executive Summary

This report presents a forensic audit of the **Kurnaz YOLOv8 Acne Dataset** (`kurnaz_yolov8` version 8), located in `datasets/skin/acne/raw/kurnaz_yolov8/extracted/data-2/`. 

The primary goal of this audit was to validate the structure, image quality, annotation counts, duplication rates, and cross-split leakage to determine suitability for production-candidate model training.

### Key Audit Metrics:
* **Total Audited Images:** 927
* **Total Audited Label Files:** 927 (100% matching image-label linkage)
* **Total Bounding Boxes:** 11,807 (Class ID `0`: Acne)
* **Average BBox density:** 12.7 boxes per image
* **Estimated Unique Images:** **633** (out of 927)
* **Duplication / Augmentation Rate:** **31.7%**
* **Cross-Split Leakage Instances:** **56 instances**
* **Audit Recommendation:** **ACCEPT WITH CLEANING** (requires full deduplication and a complete, fresh re-split).

---

## 2. Dataset Structure & Integrity Verification

The dataset follows the standard YOLO object detection format. Structure verification results are as follows:

| Verification Task | Status | Details |
| :--- | :---: | :--- |
| **data.yaml Presence** | Passed | Located in root. Identifies 1 class: `['Acne']`. |
| **Directory Check** | Passed | All standard directories exist (`train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`). |
| **Corrupted Files** | Passed | 0 corrupt images found. All 927 files are fully decodable. |
| **Orphan Labels** | Passed | 0 orphan `.txt` files (every label file has a matching image). |
| **Missing Labels** | Passed | 0 missing label files (every image has a corresponding `.txt` file). |

---

## 3. Image Statistics & Quality Analysis

The dataset shows standard uniformity in format and aspect ratio, but presents significant quality variance:

* **Resolution Distribution:** 100% of images are `640x640` pixels.
* **Image Format:** 100% of images are `JPEG`.
* **Aspect Ratio:** 100% of images are square (`1.0` ratio).
* **Blurry Image Estimation:** **401 images (43.2%)** were flagged as blurry using a Laplacian variance threshold of `< 60`. 
  * *Analysis:* A high proportion of blurry images suggests that the original photos were either out-of-focus or have suffered heavy compression artifacts and scaling from Roboflow export processes.

---

## 4. Class & Annotation Audit

* **Classes:** Single class labeled `Acne` (ID `0`). No unexpected class indices were found in the label files.
* **Bounding Box Metrics:**
  * **Total BBoxes:** 11,807
  * **Invalid BBoxes:** 0 (all coordinates are correctly normalized and fall within `[0.0, 1.0]`).
  * **Tiny BBoxes:** 1 box (area `< 0.0001` relative image size).
  * **Oversized BBoxes:** 0 boxes (no box exceeds `80%` of width or height).
  * *Analysis:* Bounding box annotations are structurally sound. However, the high average density (12.7 boxes/image) indicates dense lesion clusters, which may make the model prone to classification overlap.

---

## 5. Duplicate & Augmentation Leakage Detection

We ran an exact file hash (MD5) and a perceptual diff hash (dHash) comparison to identify duplicated or near-identical images (which typically occur when a dataset is exported with offline augmentations like rotation, brightness, or blur shifts).

* **Exact Duplicates (MD5):** 10 files in 10 clusters (identical byte-for-byte copies).
* **Perceptual Duplicates (Same dHash):** 176 files in 152 clusters.
* **Near-Duplicates (Hamming Distance ≤ 6):** **294 files in 221 clusters**.
  * *Analysis:* Approximately **31.7%** of the images in the dataset are near-duplicates or augmented versions of other images.
* **Estimated Unique Images:** **633 unique original images**.

---

## 6. Split Leakage Audit (Contamination Risk)

> [!WARNING]
> **CRITICAL DATASET LEAKAGE IDENTIFIED**
> We identified 56 instances of near-identical images (Hamming Distance ≤ 6) distributed across different dataset splits:
> - **Train ↔ Test Leakage:** 35 instances
> - **Train ↔ Valid Leakage:** 19 instances
> - **Valid ↔ Test Leakage:** 2 instances

### Contamination Impact:
Having 35 instances of train-test leakage and 19 instances of train-valid leakage means that augmented variants of the training images are present in both the validation and test sets. 
Training a model on this dataset will lead to **severely inflated evaluation metrics (overfitting disguised as high performance)** because the test set is contaminated with data the model has already "seen" during training. This violates clinical validation standards.

---

## 7. Audit Recommendations & Plan

### Recommendation: ACCEPT WITH CLEANING

We should **not** use the dataset in its current split state, but we should also **not reject** it entirely, because the 11,807 bounding boxes are valuable annotations.

### Action Plan for Ingestion:
1. **Deduplicate:** Write an ingestion script that pools all 927 images, computes dHashes, and retains only **one** representative image (and corresponding label file) from each of the 221 near-duplicate clusters.
2. **Re-split:** From the resulting **633 unique images**, perform a fresh, randomized split:
   * **Train (70%):** ~443 images
   * **Validation (15%):** ~95 images
   * **Test (15%):** ~95 images
3. **Verify:** Re-run the leakage detector script on the new splits to guarantee **0 instances** of split contamination.
4. **Re-save `data.yaml`:** Generate a new configuration pointing to the sanitized folder splits.
