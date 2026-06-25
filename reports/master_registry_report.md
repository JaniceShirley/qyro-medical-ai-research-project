# QYRO Medical AI - Master Acne Registry Report
**Phase 6A: Unified Acne Metadata Registry Construction & Audit**

---

## 1. Executive Summary

This report documents the consolidation of five clinical and smartphone acne datasets into a single unified metadata registry: **`registry/master_acne_registry.csv`**. This registry acts as the single source of truth for all training, validation, out-of-distribution evaluation, and equity auditing in the QYRO Acne v1 codebase.

* **Total Consolidated Images:** 4434
* **Ingested Datasets:** Kurnaz, Tiswan, ACNE04, Google SCIN, DermNet NZ
* **Unique Bounding Box / Detection Samples:** 520 images
* **Fine-Grained Classification Samples:** 2632 images
* **Severity Grading Samples:** 752 images
* **Clinical Reference Atlas Samples:** 325 images
* **Robustness & Skin Tone Evaluation Samples:** 205 images
* **Quality Validation Check:** **PASSED**

---

## 2. Dataset Distribution & Summary Table

| Dataset | Version | Task Type | Split Profile | Annotation Type | Verified | Weight | Primary Use | Image Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiswan** | `cleaned_v1` | classification | train/valid/test | folder_classification | false | 1.0 | subtype_classification | 2632 |
| **Kurnaz** | `cleaned_v1` | detection | train/valid/test | bounding_box | false | 1.1 | lesion_detection | 520 |
| **ACNE04** | `cleaned_v1` | severity | train/valid/test | severity_stage | partial | 1.2 | severity_grading | 752 |
| **DermNet** | `dermnet_cleaned_v1` | clinical_reference | train/valid/test | clinical_reference | true | 1.3 | clinical_reference | 325 |
| **SCIN** | `cleaned_v1` | robustness | robustness_holdout | dermatologist_consensus | true | 1.4 | robustness_validation | 205 |
| **TOTAL** | - | - | - | - | - | - | - | **4434** |

---

## 3. Class and Attribute Distributions

### 3.1 Split Distribution
- **Train:** 2951 images (66.6%)
- **Validation (valid):** 631 images (14.2%)
- **Test (test):** 647 images (14.6%)
- **Robustness Holdout:** 205 images (4.6%)

### 3.2 Harmonized Subtype Distribution
- **open_comedo:** 751 images
- **closed_comedo:** 168 images
- **papular:** 634 images
- **pustular:** 627 images
- **cystic:** 743 images
- **mixed:** 752 images (from severity datasets)
- **scar:** 11 images
- **infantile:** 16 images
- **mechanica:** 1 images
- **unknown (not annotated):** 731 images

### 3.3 Harmonized Severity Distribution
- **stage_1 (Mild / Level 0):** 271 images
- **stage_2 (Moderate / Level 1):** 340 images
- **stage_3 (Severe / Level 2):** 89 images
- **stage_4 (Very Severe / Level 3):** 52 images
- **unknown (not applicable):** 3682 images

### 3.4 Clinical Quality Distribution
- **sharp (variance $\ge 150$):** 1234 images
- **usable ($60 \le$ variance $< 150$):** 2191 images
- **borderline ($20 \le$ variance $< 60$):** 962 images
- **poor (variance $< 20$):** 47 images

---

## 4. Missing Metadata Analysis

Our consolidated metadata has varying completion rates for clinical covariates:
- **Subtype Completion:** 83.5%
- **Severity Completion:** 17.0%
- **Skin Tone Completion:** 2.1% (explicitly populated for Google SCIN subset)
- **Body Region Completion:** 3.8% (explicitly populated for Google SCIN subset)

---

## 5. Quality Validation Summary

We performed a forensic validation check on the constructed registry:

| Validation Task | Status | Failures | Details |
| :--- | :---: | :---: | :--- |
| **Missing Files** | Passed | 0 | Verifies every image path exists in the workspace. |
| **Duplicate Image IDs** | Passed | 0 | Verifies all `image_id` strings are globally unique. |
| **Invalid Subtypes** | Passed | 0 | Checks mapping alignment with allowed subtypes. |
| **Invalid Severities** | Passed | 0 | Checks mapping alignment with allowed severities. |



---

## 6. Recommendations & Next Steps

1. **Modular Dataset Inclusion in Training Configurations:**
   - In subsequent development phases, build training configuration files that consume the unified registry directly.
   - Leverage `source_weight` during loss computation to account for varying annotation quality and demographic verification.
2. **Skin Tone and Body Location Imbalance:**
   - Out of 4434 images, skin tone annotations are only available for the 205 Google SCIN images.
   - For a production-ready model, look to acquire or annotate skin tones for other training datasets using automatic classifiers or dermatologist review.
3. **Clinical Validation with DermNet:**
   - Retain DermNet as a clinical evaluation anchor (`primary_use = clinical_reference`). Since it contains verified dermatologist labels, performance on the DermNet test set serves as a direct proxy for clinical accuracy on reference atlases.
