"""
QYRO Phase R1 — Per-Dataset Cleaning & Preprocessing Script

Cleans and preprocesses each dataset individually using the original QYRO-Medical-AI 
cleaning pipeline policies:
  1. Resolution Gate: Min 640x640 pixels
  2. Focus/Blur Check: Laplacian variance >= 20
  3. Exposure/Lighting Checks:
     - Grayscale histogram check:
     - Overexposure: < 25% of pixels have luminance > 245
     - Underexposure: < 30% of pixels have luminance < 20
  4. Perceptual Deduplication:
     - dHash signature generation (Hamming distance threshold <= 4)
     - Exact MD5 matches
     - Keep highest blur score image in each duplicate cluster
  5. Annotation Standardization:
     - Clip box coordinates to [0.0, 1.0]
     - Force class IDs to 0 (Acne)
  6. Split Partitioning:
     - Train (70%), Val (15%), Test (15%) using seed 42

Output directories: datasets/evaluation/cleaned/QYRO_DS00X/
"""
import os
import sys
import json
import shutil
import hashlib
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# Paths
TEMP_QYRO_ROOT = r"C:\Users\KARTHIK V\OneDrive\Desktop\TEMP-QYRO\workspace\datasets"
QYRO_ROOT = r"C:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI"
OUTPUT_CLEANED_ROOT = os.path.join(QYRO_ROOT, "datasets", "evaluation", "cleaned")
REPORT_PATH = os.path.join(QYRO_ROOT, "reports", "dataset_cleaning_report.md")

DATASETS = ["DS001", "DS002", "DS004", "DS005"]

def calculate_dhash(img_path, hash_size=8):
    """Computes Difference Hash (dHash) for an image."""
    try:
        with Image.open(img_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(pixel_left > pixel_right)
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].zfill(2))
                    decimal_value = 0
            return ''.join(hex_string)
    except Exception:
        import hashlib
        return hashlib.md5(img_path.encode()).hexdigest()[:16]

def hamming_distance(hash1, hash2):
    """Calculates Hamming distance between two hex hashes."""
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        return bin(val1 ^ val2).count('1')
    except Exception:
        return 999

def get_md5(filepath):
    """Calculates exact MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_and_standardize_label(src_path, dest_path):
    """Standardizes YOLO label by clipping coords and forcing class 0."""
    if not os.path.exists(src_path):
        return 0
    
    retained_boxes = []
    with open(src_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            # Force class 0 (Acne)
            cls = 0
            # Clip coords to [0, 1]
            x = max(0.0, min(1.0, float(parts[1])))
            y = max(0.0, min(1.0, float(parts[2])))
            w = max(0.0, min(1.0, float(parts[3])))
            h = max(0.0, min(1.0, float(parts[4])))
            
            # Filter boxes with near-zero area
            if w * h >= 0.0001:
                retained_boxes.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception:
            continue
            
    if retained_boxes:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'w') as f:
            f.writelines(retained_boxes)
        return len(retained_boxes)
    return 0

def clean_dataset(ds_id):
    print(f"\n==================== Cleaning Dataset {ds_id} ====================")
    
    # 1. Pool raw images and standardized label paths
    raw_images = []
    
    if ds_id == "DS001":
        # DS001 has split dirs in standardized
        std_ds_dir = os.path.join(TEMP_QYRO_ROOT, "standardized", ds_id)
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(std_ds_dir, split, "images")
            lbl_dir = os.path.join(std_ds_dir, split, "labels")
            if os.path.exists(img_dir):
                for f in os.listdir(img_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        stem = os.path.splitext(f)[0]
                        img_path = os.path.join(img_dir, f)
                        lbl_path = os.path.join(lbl_dir, stem + ".txt")
                        raw_images.append({
                            "stem": stem,
                            "img_path": img_path,
                            "lbl_path": lbl_path if os.path.exists(lbl_path) else None
                        })
    else:
        # DS002/DS004/DS005 has renamed images in raw, labels in standardized
        raw_ds_dir = os.path.join(TEMP_QYRO_ROOT, "raw", ds_id)
        std_ds_dir = os.path.join(TEMP_QYRO_ROOT, "standardized", ds_id)
        
        for f in os.listdir(raw_ds_dir):
            if os.path.isfile(os.path.join(raw_ds_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(f)[0]
                img_path = os.path.join(raw_ds_dir, f)
                lbl_path = os.path.join(std_ds_dir, stem + ".txt")
                raw_images.append({
                    "stem": stem,
                    "img_path": img_path,
                    "lbl_path": lbl_path if os.path.exists(lbl_path) else None
                })
                
    initial_count = len(raw_images)
    print(f"Pooled {initial_count} candidate images.")
    
    # 2. Run Image Quality Gate Checks
    passed_quality = []
    rejected_corrupted = 0
    rejected_res = 0
    rejected_blur = 0
    rejected_exposure = 0
    
    for item in raw_images:
        img_path = item["img_path"]
        
        # Corrosion Check
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            rejected_corrupted += 1
            continue
            
        # OpenCV Quality analysis
        img = cv2.imread(img_path)
        if img is None:
            rejected_corrupted += 1
            continue
            
        h, w = img.shape[:2]
        
        # Resolution Gate
        if w < 640 or h < 640:
            rejected_res += 1
            continue
            
        # Blur Gate (Laplacian Focus Variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 20.0:
            rejected_blur += 1
            continue
            
        # Exposure Gate (Grayscale Histogram checks)
        # Underexposure: if > 30% of pixels have luminance < 20
        # Overexposure: if > 25% of pixels have luminance > 245
        total_pixels = w * h
        under_pixels = np.sum(gray < 20)
        over_pixels = np.sum(gray > 245)
        
        if (under_pixels / total_pixels) > 0.30 or (over_pixels / total_pixels) > 0.25:
            rejected_exposure += 1
            continue
            
        # Pass visual quality check
        item["width"] = w
        item["height"] = h
        item["blur_score"] = blur_score
        item["md5"] = get_md5(img_path)
        passed_quality.append(item)
        
    print(f"Passed Quality Gate: {len(passed_quality)}")
    print(f"  Rejected: Corrupted: {rejected_corrupted}, Low-Res: {rejected_res}, Blurry: {rejected_blur}, Exposure: {rejected_exposure}")
    
    # 3. Near-Duplicate Detection & Perceptual Deduplication
    # Generate dHash signatures
    for item in passed_quality:
        item["dhash"] = calculate_dhash(item["img_path"])
        
    # Perceptual grouping
    processed_duplicates = set()
    cleaned_unique = []
    
    for i in range(len(passed_quality)):
        img1 = passed_quality[i]
        id1 = img1["stem"]
        if id1 in processed_duplicates:
            continue
            
        cluster = [img1]
        
        for j in range(i + 1, len(passed_quality)):
            img2 = passed_quality[j]
            id2 = img2["stem"]
            if id2 in processed_duplicates:
                continue
                
            # Exact MD5 match or perceptual dHash match (Hamming distance <= 4)
            is_dup = False
            if img1["md5"] == img2["md5"]:
                is_dup = True
            else:
                dist = hamming_distance(img1["dhash"], img2["dhash"])
                if dist <= 4:
                    is_dup = True
                    
            if is_dup:
                cluster.append(img2)
                processed_duplicates.add(id2)
                
        # Retain the one with the highest blur score (sharpest focus)
        cluster.sort(key=lambda x: x["blur_score"], reverse=True)
        cleaned_unique.append(cluster[0])
        
    removed_duplicates = len(processed_duplicates)
    print(f"Retained Unique Clean Images: {len(cleaned_unique)} (Removed duplicates: {removed_duplicates})")
    
    # 4. Shuffle and Split Partitioning (70/15/15 using seed 42)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(cleaned_unique))
    
    num_train = int(len(cleaned_unique) * 0.70)
    num_val = int(len(cleaned_unique) * 0.15)
    
    train_set = [cleaned_unique[idx] for idx in shuffled_indices[:num_train]]
    val_set = [cleaned_unique[idx] for idx in shuffled_indices[num_train:num_train+num_val]]
    test_set = [cleaned_unique[idx] for idx in shuffled_indices[num_train+num_val:]]
    
    # 5. Export to evaluation folder
    dest_dir = os.path.join(OUTPUT_CLEANED_ROOT, f"QYRO_{ds_id}")
    shutil.rmtree(dest_dir, ignore_errors=True)
    
    splits_data = {"train": train_set, "val": val_set, "test": test_set}
    stats = {
        "dataset_id": ds_id,
        "initial_raw_images": initial_count,
        "rejected_corrupted": rejected_corrupted,
        "rejected_low_res": rejected_res,
        "rejected_blurry": rejected_blur,
        "rejected_exposure": rejected_exposure,
        "removed_duplicates": removed_duplicates,
        "final_cleaned_images": len(cleaned_unique),
        "splits": {}
    }
    
    for split_name, dataset_split in splits_data.items():
        dest_img_dir = os.path.join(dest_dir, "images", split_name)
        dest_lbl_dir = os.path.join(dest_dir, "labels", split_name)
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        split_bboxes = 0
        split_labels = 0
        
        for item in dataset_split:
            # Copy Image
            img_file = os.path.basename(item["img_path"])
            shutil.copy2(item["img_path"], os.path.join(dest_img_dir, img_file))
            
            # Clean and Copy Label
            if item["lbl_path"]:
                dest_lbl_path = os.path.join(dest_lbl_dir, item["stem"] + ".txt")
                box_count = clean_and_standardize_label(item["lbl_path"], dest_lbl_path)
                if box_count > 0:
                    split_bboxes += box_count
                    split_labels += 1
                    
        stats["splits"][split_name] = {
            "images": len(dataset_split),
            "labels": split_labels,
            "bboxes": split_bboxes
        }
        print(f"  Exported {split_name}: {len(dataset_split)} images, {split_labels} labels, {split_bboxes} bboxes")
        
    # Write dataset.yaml
    yaml_content = f"""path: {dest_dir}
train: images/train
val: images/val
test: images/test
names:
  0: acne
"""
    with open(os.path.join(dest_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
        
    # Save stats
    with open(os.path.join(dest_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    return stats

def main():
    print("Starting QYRO Dataset Curation and Cleaning Pipeline...")
    os.makedirs(OUTPUT_CLEANED_ROOT, exist_ok=True)
    
    summary_stats = {}
    for ds_id in DATASETS:
        summary_stats[ds_id] = clean_dataset(ds_id)
        
    # Save combined report
    with open(os.path.join(OUTPUT_CLEANED_ROOT, "cleaned_summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)
        
    # Generate dataset_cleaning_report.md
    print("\nGenerating Curation Report Markdown...")
    report_lines = [
        "# QYRO Medical AI - Incremental Dataset Curation & Cleaning Report",
        "**Phase R1: Standardized Curation & Zero-Leakage Preprocessing Pipeline**",
        "",
        "---",
        "",
        "## 1. Curation Pipeline Summary Table",
        "",
        "| Dataset ID | Ingested | Corrupt Drop | Low-Res Drop | Blurry Drop | Exposure Drop | Duplicate Drop | Final Cleaned | Retained % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for ds in DATASETS:
        s = summary_stats[ds]
        pct = (s["final_cleaned_images"] / s["initial_raw_images"] * 100) if s["initial_raw_images"] > 0 else 0
        report_lines.append(
            f"| **{ds}** | {s['initial_raw_images']} | {s['rejected_corrupted']} | {s['rejected_low_res']} | "
            f"{s['rejected_blurry']} | {s['rejected_exposure']} | {s['removed_duplicates']} | "
            f"**{s['final_cleaned_images']}** | {pct:.1f}% |"
        )
        
    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Reconstructed Training Split Breakdown",
        "",
        "All cleaned unique images were randomly partitioned into 70/15/15 training splits using seed `42`:",
        "",
        "| Dataset ID | Train Set (Imgs/Bboxes) | Val Set (Imgs/Bboxes) | Test Set (Imgs/Bboxes) | Total Cleaned Bboxes |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])
    
    for ds in DATASETS:
        s = summary_stats[ds]
        t = s["splits"]["train"]
        v = s["splits"]["val"]
        te = s["splits"]["test"]
        total_boxes = t["bboxes"] + v["bboxes"] + te["bboxes"]
        report_lines.append(
            f"| **{ds}** | {t['images']} / {t['bboxes']} | {v['images']} / {v['bboxes']} | {te['images']} / {te['bboxes']} | **{total_boxes}** |"
        )
        
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Cleaning & Curation Methodology Details",
        "",
        "- **Resolution Gate**: Rejects all images with width or height below **640 pixels**.",
        "- **Focus Assessment**: Computes Laplacian variance. Rejects images with variance **< 20** (severe blur).",
        "- **Luminance Histograms Check**: Rejects images where **> 30% of pixels < 20** (extreme dark) or **> 25% of pixels > 245** (extreme bright).",
        "- **Deduplication Engine**: Uses Difference Hashing (dHash size 8x8) and exact MD5 hashes to identify duplicates. Hamming distance threshold is **<= 4**.",
        "- **Annotation Bounds Enforcement**: Coordinates (x, y, w, h) clipped to `[0.0, 1.0]` boundaries, class index forced to `0` (Acne), and boxes smaller than `0.0001` area dropped.",
        "- **Split Seeding**: Random shuffle and partition utilizing seed `42` to prevent train/valid/test cross-split leakage.",
        "",
        "All cleaned sets are saved under [datasets/evaluation/cleaned/](file:///C:/Users/KARTHIK%20V/OneDrive/Desktop/QYRO-Medical-AI/datasets/evaluation/cleaned) and are ready for independent training."
    ])
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Curation report saved to {REPORT_PATH}")
    print("[SUCCESS] Dataset cleaning pipeline completed.")

if __name__ == "__main__":
    main()
