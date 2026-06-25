# QYRO Medical AI - DermNet Ingestion & Forensic Filtering Report
**Phase 5A: Forensic Ingestion Audit of Acne/Rosacea Dataset**

---

## 1. Executive Summary

This report documents the forensic quality and diagnostic filtering audit of the combined **DermNet NZ Acne and Rosacea Dataset** located at `datasets/skin/acne/raw/dermnet/extracted/`. 

DermNet NZ is a high-quality, dermatologist-curated clinical image library. However, the raw download bundles Acne and Rosacea classes together. For QYRO Medical AI (focused specifically on acne lesions), we must perform diagnostic filtering to separate true acne from rosacea-only cases, look-alike conditions (e.g., perioral dermatitis, hidradenitis suppurativa), and irrelevant media.

### Key Audit Findings:
* **Total Scanned Images:** 1152 (Train: 840, Test: 312)
* **True Acne Active Lesions (Target):** **468** (40.6%)
* **True Acne Scars:** **13** (1.1%)
* **Rosacea-Only (Reject):** **284** (24.7%)
* **Perioral Dermatitis (Reject):** **171** (14.8%)
* **Hidradenitis Suppurativa (Reject):** **135** (11.7%)
* **Other Non-Acne Vulgaris / Irrelevant (Reject):** **81** images
* **Estimated Usable Acne Images:** **363** unique images (75.5% retention of raw acne)
* **Audit Recommendation:** **PARTIAL ACCEPT (Extract Acne Subset & Purge Rosacea/Others)**

---

## 2. Dataset Structure and File Organization

The dataset files are stored in two folders: `acne_rosacea_train/` (840 files) and `acne_rosacea_test/` (312 files). 

There are no separate annotation/CSV files; instead, the files follow a distinct descriptive naming convention (e.g., `acne-cystic-101.jpg`, `perioral-dermatitis-122.jpg`, `rosacea-nose-29.jpg`). This allows us to separate classes programmatically with 100% precision.

---

## 3. Acne / Rosacea / Other Balance

The combined dataset contains a significant amount of clinical noise and non-acne conditions:

| Category | Train Count | Test Count | Total Count | Percentage | Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Active Acne Vulgaris** | 351 | 117 | **468** | 40.6% | **KEEP** |
| **Acne Scarring** | 11 | 2 | **13** | 1.1% | **KEEP (as auxiliary)** |
| **Rosacea & Rhinophyma** | 207 | 77 | **284** | 24.7% | **REJECT (Diagnostic Noise)** |
| **Perioral Dermatitis** | 116 | 55 | **171** | 14.8% | **REJECT (Look-alike)** |
| **Hidradenitis Suppurativa** | 96 | 39 | **135** | 11.7% | **REJECT (Non-Acne Vulgaris)** |
| **Milia** | 17 | 5 | **22** | 1.9% | **REJECT** |
| **Fordyce Spots** | 7 | 6 | **13** | 1.1% | **REJECT** |
| **Minocycline Pigmentation** | 8 | 2 | **10** | 0.9% | **REJECT** |
| **Nevus Comedonicus** | 6 | 1 | **7** | 0.6% | **REJECT** |
| **Other Conditions / Non-human** | 21 | 8 | **29** | 2.1% | **REJECT** |

---

## 4. Quality Gate & Blur Assessment

Image blur was evaluated across all valid decodable files:

* **High Quality (Variance ≥ 150):** 1104 images (95.8%) - Extremely crisp, high-magnification clinical photography.
* **Usable (60 ≤ Variance < 150):** 42 images (3.6%) - Sharp macro shots.
* **Borderline (20 ≤ Variance < 60):** 6 images (0.5%) - Mild focus softness.
* **Reject-Worthy (Variance < 20):** 0 images (0.0%) - Very low-resolution previews or severely out-of-focus crops.
* **Corrupted Files:** 0 images.

---

## 5. Risk Assessment and Contamination

1. **Diagnostic Look-alike Contamination:** Over **58.6%** of the dataset does *not* show acne vulgaris. Rosacea, perioral dermatitis, and hidradenitis suppurativa represent major challenges; if left uncleaned, they will contaminate training sets and result in poor diagnostic specificity.
2. **Educational / Histology Contamination:** There are **4** histology or diagram files (e.g. `acne-histology-4.jpg`, `Forest-2.jpg`) which contain tissue samples under a microscope or non-medical images. These must be deleted during reconstruction.
3. **Data Leakage Risk:** There are **270** files in the test split that match images in the training split. This leakage is typical of legacy splits on public Kaggle copies and must be corrected during split reconstruction.

---

## 6. Expected Acne Retention & Recommendations

### Target Acne Retention:
* **Raw Acne-related Images:** 481
* **Exact duplicates within Acne:** 118
* **Blur rejects (Variance < 20):** 0
* **Expected Unique Usable Acne Images:** **363** images

### Recommendations:
1. **ACCEPT FOR SUBSET EXTRACTION:** DermNet's true active acne images (468 files) are clinical-grade, dermatologist-verified representations of comedones, pustules, and deep cysts. They are extremely valuable for validation and fine-tuning.
2. **FILTER & REBUILD:** Do not use the raw combined folders. Build a cleaned, deduplicated, leakage-free classification subset of the **363** unique acne images under:
   `datasets/skin/acne/cleaned/dermnet_cleaned_v1/`
