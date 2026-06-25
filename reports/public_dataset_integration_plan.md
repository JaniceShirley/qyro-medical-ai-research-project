# QYRO Public Dataset Integration Plan

## Objective
Establish strict automated and manual quality control protocols before aggregating external open-source datasets (e.g., Roboflow datasets) into `acne_v2_curated`.

## 1. Automatic Dataset Compatibility Checker
Before merging any external dataset, it must pass a programmatic compatibility check enforcing:
* **Format**: Standard YOLO 1.1 normalized format (`.txt` labels).
* **Class Constraints**: All labels must map to class `0` (`acne`). Any multi-class sets must be filtered or remapped.
* **Integrity**: Flag any missing images, missing labels, corrupted image files, or bounding boxes outside the `[0.0, 1.0]` bounds.

## 2. Duplicate Detection (CRITICAL)
To prevent dataset contamination and data-leakage across splits:
* **Perceptual Hashing (pHash) / CLIP Embeddings**: Every incoming image will be fingerprinted using pHash (or a CLIP image encoder).
* **Collision Check**: If an incoming hash closely matches an image already present in `acne_v2_curated`, it will be **automatically rejected**. 

## 3. Image Normalization Policy
* Ensure filenames are globally unique across sets to prevent overwrites (e.g., prefixing `roboflow_`).
* Normalize extensions to `.jpg` for pipeline consistency.

## 4. Train/Validation Split Policy
* External datasets will primarily feed the **Train Split**. 
* The Validation and Test splits must remain strictly controlled domains (like the Kurnaz subset) to ensure metric consistency against the v1 baseline.

## 5. Licensing Checklist
* Must verify CC-BY, CC-BY-SA, or Public Domain (CC0) licenses. 
* Commercial restriction (NC) licenses must be flagged for legal review.
