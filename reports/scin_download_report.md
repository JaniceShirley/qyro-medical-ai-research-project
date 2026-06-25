# QYRO Medical AI - Google SCIN Image Acquisition Report
**Phase 4B: Selective Image Acquisition & Verification**

---

## 1. Executive Summary

This report documents the selective download and verification of the acne-related subset of the **Google SCIN (Skin Condition Image Network) Dataset** to `datasets/skin/acne/raw/google_scin/extracted/acne_subset/`.

* **Requested Images in Manifest:** 205
* **Successfully Downloaded:** 205
* **Failed Downloads:** 0
* **Corrupt / Unreadable Images:** 0
* **Total Valid Decodable Images:** **205**
* **Acquisition Status:** **SUCCESSFUL**

---

## 2. Ingestion Status and Verification

| Metric | Count | Percentage | Details |
| :--- | :---: | :---: | :--- |
| **Requested in Manifest** | 205 | 100.0% | Core acne target |
| **Downloaded successfully** | 205 | 100.0% | Fetched via GCS HTTP |
| **Failed downloads** | 0 | 0.0% | Bucket missing objects or network failures |
| **Valid Decodable Images** | **205** | **100.0%** | Decodable by PIL and CV2 |

### Failed Downloads Log:
*None*

---

## 3. Image Characteristics

* **Resolutions:** 1080x810 (15), 607x1080 (8), 801x811 (1), 810x1080 (89), 776x810 (1), 809x699 (1), 729x828 (1), 810x1055 (1), 810x767 (1), 809x589 (1), 810x914 (1), 498x881 (1), 1080x486 (3), 1080x518 (1), 1080x883 (1), 342x415 (1), 719x737 (1), 811x1080 (2), 601x1050 (1), 498x1080 (2), 683x1080 (1), 408x999 (1), 718x743 (1), 586x627 (1), 813x1080 (6), 487x1080 (1), 608x1080 (1), 449x783 (1), 624x588 (1), 809x852 (1), 745x594 (1), 403x549 (1), 1080x811 (3), 820x810 (1), 710x593 (1), 1080x1023 (1), 509x614 (1), 680x688 (1), 1080x910 (1), 606x1036 (1), 622x850 (1), 594x885 (1), 650x1080 (1), 759x646 (1), 453x848 (1), 634x904 (1), 810x1079 (1), 443x980 (1), 810x954 (1), 810x1078 (1), 809x1080 (2), 810x631 (1), 536x781 (1), 745x783 (1), 692x455 (1), 810x1014 (1), 686x998 (1), 597x712 (1), 806x1079 (1), 497x1080 (3), 810x762 (1), 810x686 (1), 647x919 (1), 590x955 (1), 499x1080 (1), 546x796 (1), 486x1080 (2), 627x810 (1), 737x775 (1), 388x592 (1), 347x318 (1), 810x929 (1), 552x488 (1), 810x868 (1), 746x745 (1), 1080x523 (1), 806x1076 (1), 604x829 (1), 504x805 (1), 810x769 (1), 810x1037 (1)
* **Formats:** PNG (205)
* **Aspect Ratios:** 1.333 (15), 0.562 (8), 0.988 (2), 0.75 (89), 0.958 (1), 1.157 (1), 0.88 (1), 0.768 (1), 1.056 (2), 1.374 (1), 0.886 (1), 0.565 (1), 2.222 (3), 2.085 (1), 1.223 (1), 0.824 (1), 0.976 (1), 0.751 (4), 0.572 (1), 0.461 (2), 0.632 (1), 0.408 (1), 0.966 (1), 0.935 (1), 0.753 (6), 0.451 (1), 0.563 (1), 0.573 (1), 1.061 (1), 0.95 (1), 1.254 (1), 0.734 (1), 1.332 (3), 1.012 (1), 1.197 (1), 0.829 (1), 1.187 (1), 0.585 (1), 0.732 (1), 0.671 (1), 0.602 (1), 1.175 (1), 0.534 (1), 0.701 (1), 0.452 (1), 0.849 (1), 0.749 (3), 1.284 (1), 0.686 (2), 0.951 (2), 1.521 (1), 0.799 (1), 0.687 (1), 0.838 (1), 0.747 (1), 0.46 (3), 1.063 (1), 1.181 (1), 0.704 (1), 0.618 (1), 0.462 (1), 0.45 (2), 0.774 (1), 0.655 (1), 1.091 (1), 0.872 (1), 1.131 (1), 0.933 (1), 1.001 (1), 2.065 (1), 0.729 (1), 0.626 (1), 1.053 (1), 0.781 (1)

---

## 4. Blur and Quality Gate Audit

Using Laplacian variance, we evaluated image blur across the **205** valid images:

* **High Quality (Variance ≥ 150):** 75 images (36.6%)
* **Usable (60 ≤ Variance < 150):** 65 images (31.7%)
* **Borderline (20 ≤ Variance < 60):** 41 images (20.0%)
* **Reject-Worthy (Variance < 20):** 24 images (11.7%)

---

## 5. Duplicate Check
* **Exact duplicates (MD5):** 0 images.
* **Perceptual duplicates (dHash):** 5 images.
* *Analysis:* SCIN exhibits low internal duplication, which is expected for crowdsourced photo sets.

---

## 6. Training and validation readiness

The selective Google SCIN Acne subset has been successfully staged. It represents a highly realistic mobile camera dataset. Because it contains no cross-split leakage by default, we can cleanly use the **181** usable unique images for validating model robustness against real-world smartphone noise.
