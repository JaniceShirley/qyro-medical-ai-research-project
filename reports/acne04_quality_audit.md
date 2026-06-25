# QYRO Medical AI - ACNE04 Quality Audit Report
**Phase 3B: Forensic Quality Audit of `all_1024/`**

---

## 1. Executive Summary

This report documents the forensic quality audit of the master **ACNE04 dataset** (`all_1024/` split). The objective was to analyze internal image quality, duplication, and mapping alignment to assess whether ACNE04 is suitable as a reliable validation benchmark for QYRO's global acne severity grading model.

### Key Audit Metrics:
* **Total Scanned Images:** 1,406
* **Internal Duplication Rate:** **24.3%**
* **Estimated Unique Images:** **1,064**
* **Reject-Worthy Blur Rate:** **10.8%** (152 images)
* **Estimated Usable Unique Images:** **918** (after deduplication and quality gates)
* **Audit Recommendation:** **ACCEPT WITH LIGHT CLEANING**
* **Production Suitability Score:** **8.0 / 10** (for global severity validation; 0/10 for bounding box detection)

---

## 2. Duplicate Analysis

We performed exact file hash matching (MD5) and perceptual diff hash matching (dHash) to identify duplicates inside the `all_1024/` master folder.

* **Exact Duplicates:** 36 images.
* **Near-Duplicates (Hamming Distance ≤ 6):** **342 images (24.3%)**.
* **Estimated Unique Images:** **1,064**.
* *Analysis:* The duplicate rate of 24.3% is significant but lower than Tiswan (36.6%). These duplicates are primarily caused by minor compression copies, framing adjustments, or slight resizing variations. We must deduplicate the folder to ensure unbiased model evaluations.

---

## 3. Image Quality Findings

Using Laplacian variance, we evaluated image blur across the 1,406 images:

* **High Quality (Variance ≥ 150):** **900 images (64.0%)** - Standard, highly detailed clinical front-facing portraits.
* **Usable (60 ≤ Variance < 150):** **202 images (14.4%)** - Good quality photos with slight focus softening.
* **Borderline (20 ≤ Variance < 60):** **152 images (10.8%)** - Mild compression blur or smartphone texture softening.
* **Reject-Worthy (Variance < 20):** **152 images (10.8%)** - Severe focus distortion, low-resolution scaling, or camera motion artifacts.

---

## 4. Stage Distribution & Balance

ACNE04 is pre-classified into four global severity levels (Level 0 = Mild, Level 1 = Moderate, Level 2 = Severe, Level 3 = Very Severe), which we parsed from the filename prefixes in `all_1024/`:

```
Level 0 (Mild)        ████████████████████ 491 (34.9%)
Level 1 (Moderate)    ██████████████████████████ 623 (44.3%)
Level 2 (Severe)      ███████ 177 (12.6%)
Level 3 (Very Severe)  █████ 115 (8.2%)
```

* **Imbalance Analysis:** There is a strong skew towards Level 1 (623 images) and Level 0 (491 images). Together, mild-to-moderate cases make up **79.2%** of the dataset, while severe-to-very-severe cases (Levels 2 and 3) represent only **20.8%**.
* *Impact:* While this distribution matches typical real-world clinical prevalence (where mild-to-moderate acne is far more common than severe nodulocystic acne), it means our validation test set will have fewer samples for severe stages. We must avoid aggressive filtering on Level 2 and Level 3 images to preserve validation statistical power.

---

## 5. QYRO Clinical Staging Mapping

The 4 severity levels in ACNE04 map directly to QYRO's clinical Staging System, allowing for straightforward confidence scoring and reasoning calibration:

| ACNE04 Level | Clinical Severity | QYRO Stage Equivalent | Primary Lesion Indicators |
| :--- | :--- | :--- | :--- |
| **Level 0** | Mild | **Stage 1 (Comedonal)** | Predominantly non-inflammatory comedones (blackheads and whiteheads); few papules. |
| **Level 1** | Moderate | **Stage 2 (Papular)** | Moderate inflammatory papules; minimal pustules; no nodular lesions. |
| **Level 2** | Severe | **Stage 3 (Pustular)** | High density of inflammatory papules and pustules; early nodular structures. |
| **Level 3** | Very Severe | **Stage 4 (Cystic/Nodular)** | Prominent cystic lesions, deep nodules, pustular clusters, and scarring. |

### Reasoning Calibration:
ACNE04 can be used to validate the output probabilities of QYRO's severity grading heads. Since the images represent standardized clinical frontal views, we can calibrate our classification thresholds (e.g. mapping the softmax output of our classification network to clinical confidence scores).

---

## 6. Clinical Usefulness & Recommendation

### Recommendation: ACCEPT WITH LIGHT CLEANING

We should accept the dataset for **model validation** purposes. The standard of the clinical photography is excellent (1024x1024 standardized frontal shots, 64% high-quality sharp images).

### Light Cleaning Action Plan:
1. **Deduplicate:** Filter `all_1024/` using dHash to retain only the sharpest representative image from each of the 342 near-duplicate clusters (retaining 1,064 unique images).
2. **Blur Purge:** Reject the 152 extremely blurry images (variance < 20).
3. **Save Output:** Write the resulting **918 usable unique images** to `datasets/skin/acne/cleaned/acne04_cleaned_v1/`.
4. **Zero-Leakage split is not required** if this dataset is used strictly for validation/benchmarking (external test set). However, if we mix a subset of these images into training, we must apply a strict duplicate cluster check to prevent leakage.
