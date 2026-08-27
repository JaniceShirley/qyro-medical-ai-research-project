import os
import glob
import json
import hashlib
import cv2
import numpy as np

ACNE_V1_DIR = "datasets/acne_v1_original"
OUTPUT_REPORT = "reports/candidate_dataset_report.json"

CANDIDATE_SOURCES = [
    {
        "name": "ACNE04_severity",
        "path": "datasets/skin/acne/final/severity",
        "trait": "erythema_and_inflammatory_lesions"
    },
    {
        "name": "Tiswan_DermNet_subtypes",
        "path": "datasets/skin/acne/final/subtype_classification",
        "trait": "papular_pustular_low_contrast"
    },
    {
        "name": "Google_SCIN_robustness",
        "path": "datasets/skin/acne/final/robustness_holdout/images",
        "trait": "skin_tone_diversity"
    }
]

def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image_path, hash_size=8):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

def hamming_distance(h1, h2):
    if h1 is None or h2 is None:
        return 999
    return bin(h1 ^ h2).count('1')

def get_image_quality_metrics(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)
    return blur_score, mean_brightness

def main():
    print("=== Phase 1: Candidate Image Selection & Data-Leakage Audit ===")
    
    # Step 1: Collect all original acne_v1 images and compute hashes
    original_hashes = {"sha256": set(), "dhash": []}
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(ACNE_V1_DIR, split, "images")
        if os.path.exists(split_dir):
            for img_name in os.listdir(split_dir):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    p = os.path.join(split_dir, img_name)
                    s256 = compute_sha256(p)
                    dh = compute_dhash(p)
                    original_hashes["sha256"].add(s256)
                    if dh is not None:
                        original_hashes["dhash"].append((dh, p, split))

    print(f"Loaded {len(original_hashes['sha256'])} reference sha256 hashes from acne_v1_original.")

    # Step 2: Scan candidate datasets
    candidates = []
    rejected_count = 0
    duplicate_count = 0

    for source in CANDIDATE_SOURCES:
        source_name = source["name"]
        source_path = source["path"]
        trait = source["trait"]

        if not os.path.exists(source_path):
            print(f"[WARNING] Path {source_path} does not exist. Skipping.")
            continue

        image_files = []
        for root, dirs, files in os.walk(source_path):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_files.append(os.path.join(root, f))

        print(f"Scanning {source_name}: found {len(image_files)} potential images.")

        for img_p in image_files:
            s256 = compute_sha256(img_p)
            # Exact SHA256 check
            if s256 in original_hashes["sha256"]:
                duplicate_count += 1
                continue

            dh = compute_dhash(img_p)
            # Perceptual dHash distance check (threshold <= 4)
            is_near_dup = False
            for ref_dh, ref_p, ref_split in original_hashes["dhash"]:
                if hamming_distance(dh, ref_dh) <= 4:
                    is_near_dup = True
                    break

            if is_near_dup:
                duplicate_count += 1
                continue

            # Quality Check
            blur_score, mean_brightness = get_image_quality_metrics(img_p)
            if blur_score is None or blur_score < 15.0 or mean_brightness < 15.0 or mean_brightness > 240.0:
                rejected_count += 1
                continue

            # Selected as valid candidate
            candidates.append({
                "source_dataset": source_name,
                "image_path": img_p,
                "image_id": os.path.splitext(os.path.basename(img_p))[0],
                "trait_category": trait,
                "blur_score": round(blur_score, 2),
                "mean_brightness": round(mean_brightness, 2),
                "sha256": s256
            })

    print(f"Candidate Scan Complete:")
    print(f"  - Total Candidates Selected: {len(candidates)}")
    print(f"  - Duplicates / Leakage Rejected: {duplicate_count}")
    print(f"  - Poor Quality Rejected: {rejected_count}")

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    report_data = {
        "total_selected": len(candidates),
        "duplicates_rejected": duplicate_count,
        "quality_rejected": rejected_count,
        "candidates": candidates
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Saved Candidate Report to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
