"""
QYRO Phase R1 — Per-Dataset Extraction Script

Extracts individual datasets from TEMP-QYRO standardized/raw directories
into isolated YOLO-ready training directories for independent evaluation.

Source data layout:
  DS001: standardized/ has full YOLO structure (train/valid/test with images+labels)
  DS002/DS004/DS005: 
    - raw/ top-level has renamed images (DS00X_NNNNN.jpg)
    - standardized/ has renamed labels (DS00X_NNNNN.txt)  
    - raw/ train/valid/test has original Roboflow filenames
    - DB maps DS00X_NNNNN -> original_filename for split assignment

Output: datasets/evaluation/QYRO_DS00X/{images,labels}/{train,val,test}/
"""
import os
import sys
import json
import shutil
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
TEMP_QYRO_ROOT = r"C:\Users\KARTHIK V\OneDrive\Desktop\TEMP-QYRO\workspace\datasets"
QYRO_ROOT = r"C:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI"
OUTPUT_ROOT = os.path.join(QYRO_ROOT, "datasets", "evaluation")
DB_PATH = os.path.join(os.path.dirname(TEMP_QYRO_ROOT), "database", "dataset_index.sqlite")

DATASETS = ["DS001", "DS002", "DS004", "DS005"]

# =============================================================================
# Helper Functions
# =============================================================================
def count_bboxes(label_path):
    """Count non-empty lines (bounding boxes) in a YOLO label file."""
    if not os.path.exists(label_path):
        return 0
    with open(label_path, 'r') as f:
        return len([line for line in f if line.strip()])

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def write_dataset_yaml(output_dir, dataset_id):
    """Write a dataset.yaml file for YOLOv8."""
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test
names:
  0: acne
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

# =============================================================================
# DS001 Extraction (standardized has full YOLO structure)
# =============================================================================
def extract_ds001():
    """
    DS001 standardized directory already has full YOLO structure.
    Copy images and labels preserving the train/valid/test splits.
    """
    ds_id = "DS001"
    src_dir = os.path.join(TEMP_QYRO_ROOT, "standardized", ds_id)
    dest_dir = os.path.join(OUTPUT_ROOT, f"QYRO_{ds_id}")
    
    print(f"\n{'='*60}")
    print(f"Extracting {ds_id} from standardized directory")
    print(f"{'='*60}")
    
    # Map standardized split names to output split names
    split_map = {"train": "train", "valid": "val", "test": "test"}
    
    stats = {"dataset_id": ds_id, "source": "standardized", "splits": {}}
    
    for src_split, dest_split in split_map.items():
        src_img_dir = os.path.join(src_dir, src_split, "images")
        src_lbl_dir = os.path.join(src_dir, src_split, "labels")
        dest_img_dir = os.path.join(dest_dir, "images", dest_split)
        dest_lbl_dir = os.path.join(dest_dir, "labels", dest_split)
        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        # Copy images
        img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in img_files:
            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(dest_img_dir, f))
        
        # Copy labels
        lbl_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]
        total_bboxes = 0
        non_empty_labels = 0
        for f in lbl_files:
            src_lbl = os.path.join(src_lbl_dir, f)
            shutil.copy2(src_lbl, os.path.join(dest_lbl_dir, f))
            bboxes = count_bboxes(src_lbl)
            total_bboxes += bboxes
            if bboxes > 0:
                non_empty_labels += 1
        
        stats["splits"][dest_split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "non_empty_labels": non_empty_labels,
            "bboxes": total_bboxes
        }
        print(f"  {dest_split}: {len(img_files)} images, {len(lbl_files)} labels, {total_bboxes} bboxes")
    
    # Write dataset.yaml
    yaml_path = write_dataset_yaml(dest_dir, ds_id)
    print(f"  Created: {yaml_path}")
    
    # Compute totals
    stats["total_images"] = sum(s["images"] for s in stats["splits"].values())
    stats["total_labels"] = sum(s["labels"] for s in stats["splits"].values())
    stats["total_bboxes"] = sum(s["bboxes"] for s in stats["splits"].values())
    
    # Save stats
    stats_path = os.path.join(dest_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {stats_path}")
    
    return stats

# =============================================================================
# DS002/DS004/DS005 Extraction (raw top-level images + standardized labels)
# =============================================================================
def extract_dataset_with_db_mapping(ds_id):
    """
    For DS002/DS004/DS005:
    - Raw top-level has renamed images (DS00X_NNNNN.jpg)
    - Standardized has renamed labels (DS00X_NNNNN.txt)
    - DB has original_filename mapping to determine which split each image belongs to
    - Raw train/valid/test has original filenames for split determination
    """
    raw_dir = os.path.join(TEMP_QYRO_ROOT, "raw", ds_id)
    std_dir = os.path.join(TEMP_QYRO_ROOT, "standardized", ds_id)
    dest_dir = os.path.join(OUTPUT_ROOT, f"QYRO_{ds_id}")
    
    print(f"\n{'='*60}")
    print(f"Extracting {ds_id} with DB mapping")
    print(f"{'='*60}")
    
    # Step 1: Build original_filename -> split mapping from raw directory
    orig_to_split = {}
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(raw_dir, split, "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                orig_to_split[f] = split
    
    # Step 2: Query DB for DS00X_NNNNN -> original_filename mapping
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_id, original_filename FROM images WHERE dataset_id = ?",
        (ds_id,)
    )
    id_to_orig = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    
    # Step 3: Build image_id -> split mapping
    id_to_split = {}
    for image_id, orig_fn in id_to_orig.items():
        split = orig_to_split.get(orig_fn, None)
        if split:
            # Normalize split names
            if split == "valid":
                id_to_split[image_id] = "val"
            else:
                id_to_split[image_id] = split
    
    print(f"  DB has {len(id_to_orig)} images, {len(id_to_split)} mapped to splits")
    
    # Step 4: Get list of available standardized label files
    std_labels = {os.path.splitext(f)[0]: f for f in os.listdir(std_dir) if f.endswith('.txt')}
    
    # Step 5: Get list of available raw top-level image files
    raw_images = {}
    for f in os.listdir(raw_dir):
        if os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = os.path.splitext(f)[0]
            raw_images[stem] = f
    
    print(f"  {len(std_labels)} standardized labels, {len(raw_images)} raw images")
    
    # Step 6: Create output directories and copy files
    stats = {"dataset_id": ds_id, "source": "raw+standardized", "splits": {}}
    split_map = {"train": "train", "val": "val", "test": "test"}
    
    for split_name in split_map.values():
        os.makedirs(os.path.join(dest_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split_name), exist_ok=True)
        stats["splits"][split_name] = {"images": 0, "labels": 0, "non_empty_labels": 0, "bboxes": 0}
    
    # Copy files maintaining split assignment
    unmapped = 0
    for stem, img_file in raw_images.items():
        split = id_to_split.get(stem)
        if not split:
            unmapped += 1
            continue
        
        # Copy image
        src_img = os.path.join(raw_dir, img_file)
        dest_img = os.path.join(dest_dir, "images", split, img_file)
        shutil.copy2(src_img, dest_img)
        stats["splits"][split]["images"] += 1
        
        # Copy label if it exists
        if stem in std_labels:
            lbl_file = std_labels[stem]
            src_lbl = os.path.join(std_dir, lbl_file)
            dest_lbl = os.path.join(dest_dir, "labels", split, lbl_file)
            shutil.copy2(src_lbl, dest_lbl)
            stats["splits"][split]["labels"] += 1
            bboxes = count_bboxes(src_lbl)
            stats["splits"][split]["bboxes"] += bboxes
            if bboxes > 0:
                stats["splits"][split]["non_empty_labels"] += 1
    
    if unmapped > 0:
        print(f"  Warning: {unmapped} images could not be mapped to a split")
    
    for split_name, s in stats["splits"].items():
        print(f"  {split_name}: {s['images']} images, {s['labels']} labels, {s['bboxes']} bboxes")
    
    # Write dataset.yaml
    yaml_path = write_dataset_yaml(dest_dir, ds_id)
    print(f"  Created: {yaml_path}")
    
    # Compute totals
    stats["total_images"] = sum(s["images"] for s in stats["splits"].values())
    stats["total_labels"] = sum(s["labels"] for s in stats["splits"].values())
    stats["total_bboxes"] = sum(s["bboxes"] for s in stats["splits"].values())
    stats["unmapped_images"] = unmapped
    
    # Save stats
    stats_path = os.path.join(dest_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {stats_path}")
    
    return stats

# =============================================================================
# Main
# =============================================================================
def main():
    print("QYRO Phase R1 — Per-Dataset Extraction")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    all_stats = {}
    
    # Extract DS001 (standardized has full YOLO structure)
    all_stats["DS001"] = extract_ds001()
    
    # Extract DS002, DS004, DS005 (raw images + standardized labels + DB mapping)
    for ds_id in ["DS002", "DS004", "DS005"]:
        all_stats[ds_id] = extract_dataset_with_db_mapping(ds_id)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Images':<10} {'Labels':<10} {'BBoxes':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 70)
    for ds_id in DATASETS:
        s = all_stats[ds_id]
        t = s["splits"].get("train", {})
        v = s["splits"].get("val", {})
        te = s["splits"].get("test", {})
        print(f"{ds_id:<10} {s['total_images']:<10} {s['total_labels']:<10} {s['total_bboxes']:<10} "
              f"{t.get('images',0):<10} {v.get('images',0):<10} {te.get('images',0):<10}")
    
    # Save combined stats
    combined_path = os.path.join(OUTPUT_ROOT, "extraction_summary.json")
    with open(combined_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved combined summary to {combined_path}")
    
    print("\n[SUCCESS] All datasets extracted successfully.")

if __name__ == "__main__":
    main()
