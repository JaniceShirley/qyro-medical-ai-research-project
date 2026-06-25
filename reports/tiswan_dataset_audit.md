# QYRO Medical AI - Tiswan Acne Dataset Audit Report
**Phase 2B: Forensic Dataset Quality & Integrity Audit**

---

## 1. Executive Summary

This report presents a forensic quality and integrity audit of the **Tiswan Acne Dataset** (`tiswan`), located in `datasets/skin/acne/raw/tiswan/extracted/AcneDataset/`.

The primary goal of this audit was to evaluate the dataset's suitability for training fine-grained acne classification models (e.g., distinguishing between Cysts, Papules, Pustules, Blackheads, and Whiteheads).

### Key Audit Metrics:
* **Total Scanned Images:** 4,617
* **Orphan Files (missing from CSV maps):** 10
* **Corrupt Images:** 0
* **Class Imbalance Ratio:** **4.15x** (Blackheads: 1,240 vs. Whiteheads: 299)
* **Duplication / Augmentation Rate:** **36.6%**
* **Estimated Unique Images:** **2,925**
* **Cross-Split Leakage Instances:** **1,535 instances**
* **Beauty-Filtered Image Rate:** **15.7%**
* **Estimated Usable Unique Images:** **2,746** (after deduplication and removing severe blur)
* **Audit Recommendation:** **ACCEPT WITH CLEANING** (critical deduplication, blur removal, and re-splitting required).
* **Production Suitability Score:** **6.5 / 10** (valuable annotations, but completely contaminated splits).

---

## 2. Dataset Structure & Integrity Verification

The dataset is organized as a classification split (Train, Valid, Test) where folders represent category labels, mirrored by one-hot `.csv` class maps.

* **Split fold integrity:** Passed. All directories (`train/`, `valid/`, `test/`) and class folders exist and are fully populated.
* **Corrupted Files:** Passed. 0 corrupt images found; all 4,617 files are fully decodable.
* **CSV Mapping Alignment:** 
  * 0 missing images (all entries in the CSVs exist in the folder structures).
  * **10 orphan files** were identified (images exist in class subdirectories but are missing from the CSV maps). These must be registered or quarantined before training.

---

## 3. Image Statistics & Subtype Balance

* **Image Properties:**
  * **Resolution:** 100% of images are `640x640`.
  * **Format:** 100% `JPEG`.
  * **Aspect Ratio:** 100% square (`1.0`).

### Class Balance Analysis:
The dataset exhibits severe class imbalance, particularly affecting the `Whiteheads` category:

```
Blackheads  ██████████████████████████ 1,240 (26.9%)
Cyst        ██████████████████████ 1,040 (22.5%)
Papules     ██████████████████████ 1,032 (22.4%)
Pustules    █████████████████████ 1,006 (21.8%)
Whiteheads  ██████ 299 (6.5%)
```

* **Imbalance Impact:** The 4.15x ratio between Blackheads (1,240) and Whiteheads (299) represents a severe risk of class bias. Models trained on this data as-is will be biased toward predicting Blackheads and will suffer from low sensitivity/recall on Whiteheads.
* *Correction Strategy:* Implement class-weight balancing in the loss function, or augment the Whitehead category during training.

---

## 4. Duplicate & Augmentation Risks

We calculated dHash codes to identify exact duplicates and near-identical augmented versions (rotations, color shifts, blur changes, cropping variants).

* **Exact Duplicates (MD5):** 45 files.
* **Perceptual Duplicates (Same dHash):** 1,478 files.
* **Near-Duplicates (Hamming Distance ≤ 6):** **1,692 files (36.6%)**.
* **Estimated Unique Images:** **2,925**.
  * *Analysis:* Over a third of the dataset consists of duplicated or augmented variations of existing images (classic Roboflow offline exports). Training on these without deduplication will lead to model overfitting.

---

## 5. Split Leakage & Contamination (CRITICAL)

> [!WARNING]
> **CRITICAL CROSS-SPLIT CONTAMINATION**
> The audit detected **1,535 cross-split leakage instances** (Hamming Distance ≤ 6) where augmented copies of the same image were scattered across splits:
> - **Train ↔ Valid Leakage:** 632 instances
> - **Train ↔ Test Leakage:** 660 instances
> - **Valid ↔ Test Leakage:** 243 instances

### Contamination Impact:
With 660 train-test leaks and 632 train-val leaks, **the validation and test sets are completely compromised.** Any model evaluated on these splits will show artificially high metrics (over-optimistic accuracy/F1) because it has seen the test images during training. The raw splits are clinically useless.

---

## 6. Image Quality & Medical Relevance Audit

Using Laplacian variance, we evaluated image blur, and checked for cosmetic skin filters using a high-resolution, low-edge skin smoothing detector:

### Quality Categories:
* **High Quality (Variance ≥ 150):** 232 images (5.0%) - Standard clinical macros.
* **Usable (60 ≤ Variance < 150):** 2,421 images (52.4%) - Acceptable phone/clinical photos.
* **Borderline (20 ≤ Variance < 60):** 1,523 images (33.0%) - Minor compression blur, but contain valid clinical features.
* **Reject-Worthy (Variance < 20):** **441 images (9.6%)** - Severe blur, motion distortion, or unreadable details.

### Medical Relevance Analysis:
* **Beauty-Filtered / Airbrushed Images:** **723 images (15.7%)** were flagged by our skin-smoothing heuristic (high resolution but extremely low local variance/edge detail in skin). 
  * *Analysis:* These images represent consumer-edited skin photos, often found on social media. They lack realistic clinical skin textures and can mislead the model.
* **Medically Realistic Images:** **3,894 images (84.3%)** contain natural skin texture and raw clinical presentations.

---

## 7. Audit Recommendation & Action Plan

### Recommendation: ACCEPT WITH CLEANING
While the raw splits are completely contaminated, the underlying 2,925 unique images are valuable assets for multi-class acne classification. The dataset should be accepted, but **rebuilt from scratch.**

### Reconstruction Action Plan (Phase 2C):
1. **Deduplicate:** Pool all 4,617 images, run a dHash deduplication pass, and retain only the sharpest representative image from each duplicate cluster (retaining the 2,925 unique images).
2. **Purge:** Reject the 441 extremely blurry images (variance < 20) and flag/quarantine beauty-filtered images.
3. **Re-split:** Perform a fresh, randomized split using the remaining **2,746 usable unique images** (70% train, 15% valid, 15% test).
4. **Imbalance Mitigation:** Incorporate loss weighting to handle the underrepresented Whitehead class.
