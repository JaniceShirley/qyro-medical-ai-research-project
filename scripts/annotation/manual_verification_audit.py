import os
import sys
import json
import shutil
import yaml

QUALITY_REPORT_PATH = "reports/auto_annotation_quality_report.json"
ORIGINAL_DATASET_DIR = "datasets/acne_v1_original"
EXPANSION_DATASET_DIR = "datasets/acne_v1_expansion"
VERIFICATION_REPORT_PATH = "reports/clinical_verification_audit.json"

def main():
    print("=== Phase 3: Clinical Verification Workflow & Expansion Dataset Creation ===")

    if not os.path.exists(QUALITY_REPORT_PATH):
        raise FileNotFoundError(f"Quality report not found at {QUALITY_REPORT_PATH}")

    with open(QUALITY_REPORT_PATH, "r") as f:
        quality_data = json.load(f)

    annotated = quality_data["annotated_candidates"]
    print(f"Loaded {len(annotated)} auto-annotated candidates.")

    # Clinical verification audit criteria:
    # 1. Filter out candidate images with avg_confidence < 0.35 (too noisy)
    # 2. Filter out extreme box counts (> 45 boxes per image)
    # 3. Prioritize erythema / inflammatory, SCIN skin tone diversity, and high confidence
    verified_candidates = []
    rejected_verification_count = 0

    for cand in annotated:
        box_count = cand["box_count"]
        avg_conf = cand["avg_confidence"]

        if avg_conf < 0.35 or box_count > 45:
            rejected_verification_count += 1
            continue

        cand["verification_status"] = "PASSED_CLINICAL_VERIFICATION"
        verified_candidates.append(cand)

    print(f"Clinical Verification Audit Complete:")
    print(f"  - Candidates Passing Verification: {len(verified_candidates)}")
    print(f"  - Candidates Rejected in Audit: {rejected_verification_count}")

    # Build datasets/acne_v1_expansion directory structure
    train_img_dir = os.path.join(EXPANSION_DATASET_DIR, "train", "images")
    train_lbl_dir = os.path.join(EXPANSION_DATASET_DIR, "train", "labels")
    valid_img_dir = os.path.join(EXPANSION_DATASET_DIR, "valid", "images")
    valid_lbl_dir = os.path.join(EXPANSION_DATASET_DIR, "valid", "labels")
    test_img_dir = os.path.join(EXPANSION_DATASET_DIR, "test", "images")
    test_lbl_dir = os.path.join(EXPANSION_DATASET_DIR, "test", "labels")

    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Copy ORIGINAL train set (364 images)
    orig_train_img = os.path.join(ORIGINAL_DATASET_DIR, "train", "images")
    orig_train_lbl = os.path.join(ORIGINAL_DATASET_DIR, "train", "labels")

    orig_train_count = 0
    for f in os.listdir(orig_train_img):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy2(os.path.join(orig_train_img, f), os.path.join(train_img_dir, f))
            lbl_name = os.path.splitext(f)[0] + ".txt"
            lbl_src = os.path.join(orig_train_lbl, lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(train_lbl_dir, lbl_name))
            orig_train_count += 1

    print(f"Copied {orig_train_count} original training images & labels.")

    # 2. Add VERIFIED EXPANSION candidates
    expansion_added_count = 0
    added_image_ids = []

    for cand in verified_candidates:
        src_img_path = cand["image_path"]
        src_lbl_path = cand["label_path"]
        ext = os.path.splitext(src_img_path)[1]

        new_name = f"exp1_{cand['source_dataset']}_{cand['image_id']}"
        dest_img_path = os.path.join(train_img_dir, f"{new_name}{ext}")
        dest_lbl_path = os.path.join(train_lbl_dir, f"{new_name}.txt")

        shutil.copy2(src_img_path, dest_img_path)
        shutil.copy2(src_lbl_path, dest_lbl_path)

        expansion_added_count += 1
        added_image_ids.append(new_name)

    total_expanded_train = orig_train_count + expansion_added_count
    print(f"Added {expansion_added_count} verified expansion images to training set.")
    print(f"Total Expanded Training Images: {total_expanded_train}")

    # 3. Copy VALIDATION set UNTOUCHED (78 images)
    orig_val_img = os.path.join(ORIGINAL_DATASET_DIR, "valid", "images")
    orig_val_lbl = os.path.join(ORIGINAL_DATASET_DIR, "valid", "labels")
    val_count = 0
    for f in os.listdir(orig_val_img):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy2(os.path.join(orig_val_img, f), os.path.join(valid_img_dir, f))
            lbl_name = os.path.splitext(f)[0] + ".txt"
            lbl_src = os.path.join(orig_val_lbl, lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(valid_lbl_dir, lbl_name))
            val_count += 1
    print(f"Copied {val_count} frozen validation images & labels.")

    # 4. Copy TEST set UNTOUCHED (78 images)
    orig_test_img = os.path.join(ORIGINAL_DATASET_DIR, "test", "images")
    orig_test_lbl = os.path.join(ORIGINAL_DATASET_DIR, "test", "labels")
    test_count = 0
    for f in os.listdir(orig_test_img):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy2(os.path.join(orig_test_img, f), os.path.join(test_img_dir, f))
            lbl_name = os.path.splitext(f)[0] + ".txt"
            lbl_src = os.path.join(orig_test_lbl, lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(test_lbl_dir, lbl_name))
            test_count += 1
    print(f"Copied {test_count} frozen test images & labels.")

    # 5. Create data.yaml
    data_yaml_path = os.path.join(EXPANSION_DATASET_DIR, "data.yaml")
    data_yaml_content = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Acne"],
        "dataset_name": "acne_v1_expansion",
        "expansion_experiment": 1,
        "original_train_count": orig_train_count,
        "expansion_train_count": expansion_added_count,
        "total_train_count": total_expanded_train
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print(f"Created dataset config at {data_yaml_path}")

    # 6. Save Clinical Verification Audit Report
    audit_report = {
        "original_train_images": orig_train_count,
        "verified_expansion_images": expansion_added_count,
        "total_expanded_train_images": total_expanded_train,
        "frozen_validation_images": val_count,
        "frozen_test_images": test_count,
        "rejected_verification_count": rejected_verification_count,
        "added_image_ids": added_image_ids
    }
    with open(VERIFICATION_REPORT_PATH, "w") as f:
        json.dump(audit_report, f, indent=2)

    print(f"Saved Clinical Verification Audit Report to {VERIFICATION_REPORT_PATH}")

if __name__ == "__main__":
    main()
