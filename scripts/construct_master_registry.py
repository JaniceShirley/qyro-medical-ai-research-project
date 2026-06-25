import os
import sys
import json
import csv
import cv2
import pandas as pd
import numpy as np

# Set workspace root
WORKSPACE_ROOT = r"c:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI"

# Paths to input directories and files
DIR_KURNAZ = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "cleaned", "kurnaz_yolov8_cleaned_v1")
DIR_TISWAN = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "cleaned", "tiswan_cleaned_v1")
DIR_ACNE04 = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "cleaned", "acne04_cleaned_v1")
DIR_SCIN_IMAGES = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "raw", "google_scin", "extracted", "acne_subset")
FILE_SCIN_CASES = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "raw", "google_scin", "original_download", "scin_cases.csv")
FILE_SCIN_LABELS = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "raw", "google_scin", "original_download", "scin_labels.csv")
DIR_DERMNET = os.path.join(WORKSPACE_ROOT, "datasets", "skin", "acne", "cleaned", "dermnet_cleaned_v1")
FILE_DERMNET_LOG = os.path.join(WORKSPACE_ROOT, "reports", "dermnet_cleaning_log.csv")

# Output files
FILE_OUTPUT_CSV = os.path.join(WORKSPACE_ROOT, "registry", "master_acne_registry.csv")
FILE_OUTPUT_STATS = os.path.join(WORKSPACE_ROOT, "reports", "master_registry_stats.json")
FILE_OUTPUT_REPORT = os.path.join(WORKSPACE_ROOT, "reports", "master_registry_report.md")

# Ensure output directories exist
os.makedirs(os.path.dirname(FILE_OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(FILE_OUTPUT_STATS), exist_ok=True)

# Required columns schema
SCHEMA_COLUMNS = [
    "image_id",
    "source_dataset",
    "dataset_version",
    "image_path",
    "split",
    "task_type",
    "subtype_label",
    "severity_label",
    "annotation_type",
    "skin_tone",
    "body_region",
    "clinical_quality",
    "dermatologist_verified",
    "duplicate_cluster",
    "source_weight",
    "confidence_source",
    "primary_use",
    "condition_family",
    "notes"
]

def get_blur_score_and_quality(filepath):
    """Calculates Laplacian variance of the image and maps to quality bins."""
    if not os.path.exists(filepath):
        return -1, "unknown"
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1, "unknown"
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # clinical quality = sharp/usable/borderline/poor
        if var >= 150:
            quality = "sharp"
        elif var >= 60:
            quality = "usable"
        elif var >= 20:
            quality = "borderline"
        else:
            quality = "poor"
        return round(var, 2), quality
    except Exception:
        return -1, "unknown"

def normalize_path(absolute_path):
    """Converts absolute path to workspace-relative path with forward slashes."""
    rel = os.path.relpath(absolute_path, WORKSPACE_ROOT)
    return rel.replace("\\", "/")

def get_scin_body_region(row):
    """Identifies active body regions in SCIN cases."""
    body_parts = [
        ('body_parts_head_or_neck', 'head_or_neck'),
        ('body_parts_arm', 'arm'),
        ('body_parts_palm', 'palm'),
        ('body_parts_back_of_hand', 'back_of_hand'),
        ('body_parts_torso_front', 'torso_front'),
        ('body_parts_torso_back', 'torso_back'),
        ('body_parts_genitalia_or_groin', 'genitalia_or_groin'),
        ('body_parts_buttocks', 'buttocks'),
        ('body_parts_leg', 'leg'),
        ('body_parts_foot_top_or_side', 'foot_top_or_side'),
        ('body_parts_foot_sole', 'foot_sole'),
        ('body_parts_other', 'other'),
    ]
    regions = []
    for col, name in body_parts:
        if col in row and row[col] == 'YES':
            regions.append(name)
    if regions:
        return ','.join(regions)
    return "unknown"

def get_scin_monk_tone(row):
    """Gets monk skin tone from SCIN case."""
    if 'monk_skin_tone_label_india' in row and not pd.isna(row['monk_skin_tone_label_india']):
        return str(int(row['monk_skin_tone_label_india']))
    return "unknown"

def main():
    records = []
    
    # Keep track of indices for image_id generation
    idx_kurnaz = 1
    idx_tiswan = 1
    idx_acne04 = 1
    idx_scin = 1
    idx_dermnet = 1
    
    # ----------------------------------------------------
    # 1. KURNAZ YOLOv8 INGESTION
    # ----------------------------------------------------
    print("Ingesting Kurnaz dataset...")
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = os.path.join(DIR_KURNAZ, split, "images")
        if not os.path.exists(img_dir):
            continue
        for filename in sorted(os.listdir(img_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            abs_path = os.path.join(img_dir, filename)
            rel_path = normalize_path(abs_path)
            
            var, quality = get_blur_score_and_quality(abs_path)
            image_id = f"kurnaz_{idx_kurnaz:06d}"
            idx_kurnaz += 1
            
            records.append({
                "image_id": image_id,
                "source_dataset": "kurnaz",
                "dataset_version": "cleaned_v1",
                "image_path": rel_path,
                "split": split,
                "task_type": "detection",
                "subtype_label": "unknown",
                "severity_label": "unknown",
                "annotation_type": "bounding_box",
                "skin_tone": "unknown",
                "body_region": "unknown",
                "clinical_quality": quality,
                "dermatologist_verified": "false",
                "duplicate_cluster": "none",
                "source_weight": 1.1,
                "confidence_source": "non_expert",
                "primary_use": "lesion_detection",
                "condition_family": "acne",
                "notes": f"Cleaned object detection image. Blur score: {var}."
            })
            
    # ----------------------------------------------------
    # 2. TISWAN INGESTION
    # ----------------------------------------------------
    print("Ingesting Tiswan dataset...")
    subtype_map = {
        "Blackheads": "open_comedo",
        "Whiteheads": "closed_comedo",
        "Papules": "papular",
        "Pustules": "pustular",
        "Cyst": "cystic"
    }
    for split in splits:
        split_dir = os.path.join(DIR_TISWAN, split)
        if not os.path.exists(split_dir):
            continue
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path) or folder not in subtype_map:
                continue
            subtype = subtype_map[folder]
            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                abs_path = os.path.join(folder_path, filename)
                rel_path = normalize_path(abs_path)
                
                var, quality = get_blur_score_and_quality(abs_path)
                image_id = f"tiswan_{idx_tiswan:06d}"
                idx_tiswan += 1
                
                records.append({
                    "image_id": image_id,
                    "source_dataset": "tiswan",
                    "dataset_version": "cleaned_v1",
                    "image_path": rel_path,
                    "split": split,
                    "task_type": "classification",
                    "subtype_label": subtype,
                    "severity_label": "unknown",
                    "annotation_type": "folder_classification",
                    "skin_tone": "unknown",
                    "body_region": "unknown",
                    "clinical_quality": quality,
                    "dermatologist_verified": "false",
                    "duplicate_cluster": "none",
                    "source_weight": 1.0,
                    "confidence_source": "non_expert",
                    "primary_use": "subtype_classification",
                    "condition_family": "acne",
                    "notes": f"Cleaned fine-grained classification image. Class: {folder}. Blur score: {var}."
                })

    # ----------------------------------------------------
    # 3. ACNE04 INGESTION
    # ----------------------------------------------------
    print("Ingesting ACNE04 dataset...")
    severity_map = {
        "Level 0": "stage_1",
        "Level 1": "stage_2",
        "Level 2": "stage_3",
        "Level 3": "stage_4"
    }
    for split in splits:
        split_dir = os.path.join(DIR_ACNE04, split)
        if not os.path.exists(split_dir):
            continue
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path) or folder not in severity_map:
                continue
            severity = severity_map[folder]
            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                abs_path = os.path.join(folder_path, filename)
                rel_path = normalize_path(abs_path)
                
                var, quality = get_blur_score_and_quality(abs_path)
                image_id = f"acne04_{idx_acne04:06d}"
                idx_acne04 += 1
                
                records.append({
                    "image_id": image_id,
                    "source_dataset": "acne04",
                    "dataset_version": "cleaned_v1",
                    "image_path": rel_path,
                    "split": split,
                    "task_type": "severity",
                    "subtype_label": "mixed",  # ACNE04 subtype = mixed
                    "severity_label": severity,
                    "annotation_type": "severity_stage",
                    "skin_tone": "unknown",
                    "body_region": "unknown",
                    "clinical_quality": quality,
                    "dermatologist_verified": "partial",
                    "duplicate_cluster": "none",
                    "source_weight": 1.2,
                    "confidence_source": "partial_expert",
                    "primary_use": "severity_grading",
                    "condition_family": "acne",
                    "notes": f"Cleaned severity grading image. Class: {folder}. Blur score: {var}."
                })

    # ----------------------------------------------------
    # 4. GOOGLE SCIN INGESTION
    # ----------------------------------------------------
    print("Ingesting Google SCIN dataset...")
    if os.path.exists(FILE_SCIN_CASES) and os.path.exists(DIR_SCIN_IMAGES):
        df_cases = pd.read_csv(FILE_SCIN_CASES)
        
        # Build image filename mapping to row indices
        img_map_1 = {os.path.basename(p): idx for idx, p in df_cases['image_1_path'].dropna().items()}
        img_map_2 = {os.path.basename(p): idx for idx, p in df_cases['image_2_path'].dropna().items()}
        img_map_3 = {os.path.basename(p): idx for idx, p in df_cases['image_3_path'].dropna().items()}
        
        for filename in sorted(os.listdir(DIR_SCIN_IMAGES)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            abs_path = os.path.join(DIR_SCIN_IMAGES, filename)
            rel_path = normalize_path(abs_path)
            
            # Find the case row index
            row_idx = img_map_1.get(filename)
            if row_idx is None:
                row_idx = img_map_2.get(filename)
            if row_idx is None:
                row_idx = img_map_3.get(filename)
                
            if row_idx is not None:
                row = df_cases.iloc[row_idx]
                case_id = row['case_id']
                fst = row['fitzpatrick_skin_type']
                skin_tone = fst if (isinstance(fst, str) and fst.startswith("FST")) else "unknown"
                body_region = get_scin_body_region(row)
                mst_india = get_scin_monk_tone(row)
            else:
                case_id = "unknown"
                skin_tone = "unknown"
                body_region = "unknown"
                mst_india = "unknown"
                
            var, quality = get_blur_score_and_quality(abs_path)
            image_id = f"scin_{idx_scin:06d}"
            idx_scin += 1
            
            records.append({
                "image_id": image_id,
                "source_dataset": "scin",
                "dataset_version": "cleaned_v1",
                "image_path": rel_path,
                "split": "robustness_holdout",  # SCIN split -> robustness_holdout
                "task_type": "robustness",
                "subtype_label": "unknown",
                "severity_label": "unknown",
                "annotation_type": "dermatologist_consensus",
                "skin_tone": skin_tone,
                "body_region": body_region,
                "clinical_quality": quality,
                "dermatologist_verified": "true",
                "duplicate_cluster": "none",
                "source_weight": 1.4,
                "confidence_source": "dermatologist_consensus",
                "primary_use": "robustness_validation",
                "condition_family": "acne",
                "notes": f"Google SCIN acne subset. Case: {case_id}. Monk Skin Tone (India): {mst_india}. Blur score: {var}."
            })

    # ----------------------------------------------------
    # 5. DERMNET INGESTION
    # ----------------------------------------------------
    print("Ingesting DermNet dataset...")
    if os.path.exists(FILE_DERMNET_LOG) and os.path.exists(DIR_DERMNET):
        df_log = pd.read_csv(FILE_DERMNET_LOG)
        
        # Build dictionary from cleaning log to get duplicate clusters
        # Map original_filename to duplicate_cluster
        dup_map = {}
        for _, r in df_log.iterrows():
            orig_fn = r['original_filename']
            cluster = r['duplicate_cluster']
            dup_map[orig_fn] = str(cluster) if (pd.notna(cluster) and cluster != "N/A") else "none"
            
        for split in splits:
            split_dir = os.path.join(DIR_DERMNET, split)
            if not os.path.exists(split_dir):
                continue
            for filename in sorted(os.listdir(split_dir)):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                abs_path = os.path.join(split_dir, filename)
                rel_path = normalize_path(abs_path)
                
                # Determine subtype from filename prefix
                subtype = "unknown"
                if filename.startswith("acne-closed-comedo-") or filename.startswith("acne-Closed-Comedo"):
                    subtype = "closed_comedo"
                elif filename.startswith("acne-cystic-"):
                    subtype = "cystic"
                elif filename.startswith("acne-excoriated-"):
                    subtype = "papular"  # DermNet excoriated -> papular
                elif filename.startswith("acne-infantile-"):
                    subtype = "infantile"
                elif filename.startswith("acne-mechanica-"):
                    subtype = "mechanica"
                elif filename.startswith("acne-open-comedo-"):
                    subtype = "open_comedo"
                elif filename.startswith("acne-primary-lesion-"):
                    subtype = "unknown"
                elif filename.startswith("acne-pustular-"):
                    subtype = "pustular"
                elif filename.startswith("acne-scar-") or filename.startswith("07AcnePittedScars"):
                    subtype = "scar"
                
                # Look up duplicate cluster in log
                cluster = dup_map.get(filename, "none")
                
                var, quality = get_blur_score_and_quality(abs_path)
                image_id = f"dermnet_{idx_dermnet:06d}"
                idx_dermnet += 1
                
                records.append({
                    "image_id": image_id,
                    "source_dataset": "dermnet",
                    "dataset_version": "dermnet_cleaned_v1",
                    "image_path": rel_path,
                    "split": split,
                    "task_type": "clinical_reference",
                    "subtype_label": subtype,
                    "severity_label": "unknown",
                    "annotation_type": "clinical_reference",
                    "skin_tone": "unknown",
                    "body_region": "unknown",
                    "clinical_quality": quality,
                    "dermatologist_verified": "true",
                    "duplicate_cluster": cluster,
                    "source_weight": 1.3,
                    "confidence_source": "dermatologist_expert",
                    "primary_use": "clinical_reference",
                    "condition_family": "acne",
                    "notes": f"DermNet clean acne atlas image. Subtype: {subtype}. Blur score: {var}."
                })

    # Convert to DataFrame
    df_master = pd.DataFrame(records, columns=SCHEMA_COLUMNS)
    
    # Save Master Registry CSV
    df_master.to_csv(FILE_OUTPUT_CSV, index=False)
    print(f"Saved master registry with {len(df_master)} rows to {FILE_OUTPUT_CSV}")
    
    # ----------------------------------------------------
    # QUALITY VALIDATION & STATS GENERATION
    # ----------------------------------------------------
    print("Validating master registry and compiling statistics...")
    
    missing_files = []
    broken_paths = []
    duplicate_ids = df_master['image_id'].duplicated().sum()
    invalid_subtypes = []
    invalid_severities = []
    
    allowed_subtypes = {"open_comedo", "closed_comedo", "papular", "pustular", "cystic", "mixed", "scar", "infantile", "mechanica", "unknown"}
    allowed_severities = {"stage_1", "stage_2", "stage_3", "stage_4", "unknown"}
    
    for idx, r in df_master.iterrows():
        abs_path = os.path.join(WORKSPACE_ROOT, r['image_path'].replace("/", os.sep))
        if not os.path.exists(abs_path):
            missing_files.append((r['image_id'], r['image_path']))
            broken_paths.append(r['image_path'])
            
        if r['subtype_label'] not in allowed_subtypes:
            invalid_subtypes.append((r['image_id'], r['subtype_label']))
            
        if r['severity_label'] not in allowed_severities:
            invalid_severities.append((r['image_id'], r['severity_label']))
            
    # Compile stats JSON
    dataset_dist = df_master['source_dataset'].value_counts().to_dict()
    subtype_dist = df_master['subtype_label'].value_counts().to_dict()
    severity_dist = df_master['severity_label'].value_counts().to_dict()
    split_dist = df_master['split'].value_counts().to_dict()
    quality_dist = df_master['clinical_quality'].value_counts().to_dict()
    
    missing_metadata = {
        "unknown_subtype": int((df_master['subtype_label'] == "unknown").sum()),
        "unknown_severity": int((df_master['severity_label'] == "unknown").sum()),
        "unknown_skin_tone": int((df_master['skin_tone'] == "unknown").sum()),
        "unknown_body_region": int((df_master['body_region'] == "unknown").sum())
    }
    
    stats = {
        "total_image_count": len(df_master),
        "dataset_distribution": {k: int(v) for k, v in dataset_dist.items()},
        "subtype_distribution": {k: int(v) for k, v in subtype_dist.items()},
        "severity_distribution": {k: int(v) for k, v in severity_dist.items()},
        "split_distribution": {k: int(v) for k, v in split_dist.items()},
        "quality_distribution": {k: int(v) for k, v in quality_dist.items()},
        "missing_metadata_analysis": missing_metadata,
        "quality_validation_summary": {
            "duplicate_image_ids_count": int(duplicate_ids),
            "missing_files_count": len(missing_files),
            "invalid_subtype_labels_count": len(invalid_subtypes),
            "invalid_severity_labels_count": len(invalid_severities)
        }
    }
    
    with open(FILE_OUTPUT_STATS, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {FILE_OUTPUT_STATS}")
    
    # Generate Markdown Report
    print("Generating Master Registry Report...")
    report_md = f"""# QYRO Medical AI - Master Acne Registry Report
**Phase 6A: Unified Acne Metadata Registry Construction & Audit**

---

## 1. Executive Summary

This report documents the consolidation of five clinical and smartphone acne datasets into a single unified metadata registry: **`registry/master_acne_registry.csv`**. This registry acts as the single source of truth for all training, validation, out-of-distribution evaluation, and equity auditing in the QYRO Acne v1 codebase.

* **Total Consolidated Images:** {len(df_master)}
* **Ingested Datasets:** Kurnaz, Tiswan, ACNE04, Google SCIN, DermNet NZ
* **Unique Bounding Box / Detection Samples:** {dataset_dist.get('kurnaz', 0)} images
* **Fine-Grained Classification Samples:** {dataset_dist.get('tiswan', 0)} images
* **Severity Grading Samples:** {dataset_dist.get('acne04', 0)} images
* **Clinical Reference Atlas Samples:** {dataset_dist.get('dermnet', 0)} images
* **Robustness & Skin Tone Evaluation Samples:** {dataset_dist.get('scin', 0)} images
* **Quality Validation Check:** **{"PASSED" if len(missing_files) == 0 and duplicate_ids == 0 else "WARNING (Quality Failures Found)"}**

---

## 2. Dataset Distribution & Summary Table

| Dataset | Version | Task Type | Split Profile | Annotation Type | Verified | Weight | Primary Use | Image Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiswan** | `cleaned_v1` | classification | train/valid/test | folder_classification | false | 1.0 | subtype_classification | {dataset_dist.get('tiswan', 0)} |
| **Kurnaz** | `cleaned_v1` | detection | train/valid/test | bounding_box | false | 1.1 | lesion_detection | {dataset_dist.get('kurnaz', 0)} |
| **ACNE04** | `cleaned_v1` | severity | train/valid/test | severity_stage | partial | 1.2 | severity_grading | {dataset_dist.get('acne04', 0)} |
| **DermNet** | `dermnet_cleaned_v1` | clinical_reference | train/valid/test | clinical_reference | true | 1.3 | clinical_reference | {dataset_dist.get('dermnet', 0)} |
| **SCIN** | `cleaned_v1` | robustness | robustness_holdout | dermatologist_consensus | true | 1.4 | robustness_validation | {dataset_dist.get('scin', 0)} |
| **TOTAL** | - | - | - | - | - | - | - | **{len(df_master)}** |

---

## 3. Class and Attribute Distributions

### 3.1 Split Distribution
- **Train:** {split_dist.get('train', 0)} images ({(split_dist.get('train', 0)/len(df_master)*100):.1f}%)
- **Validation (valid):** {split_dist.get('valid', 0)} images ({(split_dist.get('valid', 0)/len(df_master)*100):.1f}%)
- **Test (test):** {split_dist.get('test', 0)} images ({(split_dist.get('test', 0)/len(df_master)*100):.1f}%)
- **Robustness Holdout:** {split_dist.get('robustness_holdout', 0)} images ({(split_dist.get('robustness_holdout', 0)/len(df_master)*100):.1f}%)

### 3.2 Harmonized Subtype Distribution
- **open_comedo:** {subtype_dist.get('open_comedo', 0)} images
- **closed_comedo:** {subtype_dist.get('closed_comedo', 0)} images
- **papular:** {subtype_dist.get('papular', 0)} images
- **pustular:** {subtype_dist.get('pustular', 0)} images
- **cystic:** {subtype_dist.get('cystic', 0)} images
- **mixed:** {subtype_dist.get('mixed', 0)} images (from severity datasets)
- **scar:** {subtype_dist.get('scar', 0)} images
- **infantile:** {subtype_dist.get('infantile', 0)} images
- **mechanica:** {subtype_dist.get('mechanica', 0)} images
- **unknown (not annotated):** {subtype_dist.get('unknown', 0)} images

### 3.3 Harmonized Severity Distribution
- **stage_1 (Mild / Level 0):** {severity_dist.get('stage_1', 0)} images
- **stage_2 (Moderate / Level 1):** {severity_dist.get('stage_2', 0)} images
- **stage_3 (Severe / Level 2):** {severity_dist.get('stage_3', 0)} images
- **stage_4 (Very Severe / Level 3):** {severity_dist.get('stage_4', 0)} images
- **unknown (not applicable):** {severity_dist.get('unknown', 0)} images

### 3.4 Clinical Quality Distribution
- **sharp (variance $\\ge 150$):** {quality_dist.get('sharp', 0)} images
- **usable ($60 \\le$ variance $< 150$):** {quality_dist.get('usable', 0)} images
- **borderline ($20 \\le$ variance $< 60$):** {quality_dist.get('borderline', 0)} images
- **poor (variance $< 20$):** {quality_dist.get('poor', 0)} images

---

## 4. Missing Metadata Analysis

Our consolidated metadata has varying completion rates for clinical covariates:
- **Subtype Completion:** {((len(df_master) - missing_metadata['unknown_subtype'])/len(df_master)*100):.1f}%
- **Severity Completion:** {((len(df_master) - missing_metadata['unknown_severity'])/len(df_master)*100):.1f}%
- **Skin Tone Completion:** {((len(df_master) - missing_metadata['unknown_skin_tone'])/len(df_master)*100):.1f}% (explicitly populated for Google SCIN subset)
- **Body Region Completion:** {((len(df_master) - missing_metadata['unknown_body_region'])/len(df_master)*100):.1f}% (explicitly populated for Google SCIN subset)

---

## 5. Quality Validation Summary

We performed a forensic validation check on the constructed registry:

| Validation Task | Status | Failures | Details |
| :--- | :---: | :---: | :--- |
| **Missing Files** | {"Passed" if len(missing_files) == 0 else "Failed"} | {len(missing_files)} | Verifies every image path exists in the workspace. |
| **Duplicate Image IDs** | {"Passed" if duplicate_ids == 0 else "Failed"} | {duplicate_ids} | Verifies all `image_id` strings are globally unique. |
| **Invalid Subtypes** | {"Passed" if len(invalid_subtypes) == 0 else "Failed"} | {len(invalid_subtypes)} | Checks mapping alignment with allowed subtypes. |
| **Invalid Severities** | {"Passed" if len(invalid_severities) == 0 else "Failed"} | {len(invalid_severities)} | Checks mapping alignment with allowed severities. |

{"### Details of Failures:" if len(missing_files) > 0 or len(invalid_subtypes) > 0 or len(invalid_severities) > 0 or duplicate_ids > 0 else ""}
"""
    if len(missing_files) > 0:
        report_md += "\n#### Missing Files:\n"
        for img_id, path in missing_files[:20]:
            report_md += f"- `{img_id}`: `{path}` (File not found)\n"
        if len(missing_files) > 20:
            report_md += f"- ... and {len(missing_files)-20} more.\n"
            
    if len(invalid_subtypes) > 0:
        report_md += "\n#### Invalid Subtypes:\n"
        for img_id, label in invalid_subtypes[:20]:
            report_md += f"- `{img_id}`: `{label}` (Invalid subtype)\n"
            
    if len(invalid_severities) > 0:
        report_md += "\n#### Invalid Severities:\n"
        for img_id, label in invalid_severities[:20]:
            report_md += f"- `{img_id}`: `{label}` (Invalid severity)\n"

    report_md += """
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
"""

    with open(FILE_OUTPUT_REPORT, 'w') as f:
        f.write(report_md)
    print(f"Saved report to {FILE_OUTPUT_REPORT}")
    print("Done!")

if __name__ == "__main__":
    main()
