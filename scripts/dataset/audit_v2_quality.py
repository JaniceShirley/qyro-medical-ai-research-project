import os
import cv2
import numpy as np
import csv

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_brightness(image):
    # Returns average brightness and standard deviation (for glare detection)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return np.mean(v_channel), np.std(v_channel)

def parse_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return lines

def main():
    dataset_dir = 'datasets/acne_v2_curated'
    splits = ['valid', 'train', 'test']
    
    audit_results = []
    
    print("Starting comprehensive image & annotation quality audit...")
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, split, 'images')
        lbl_dir = os.path.join(dataset_dir, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            base = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, base + '.txt')
            
            # Read Image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            height, width, _ = image.shape
            resolution = f"{width}x{height}"
            
            # Image Analytics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_val = variance_of_laplacian(gray)
            avg_bright, std_bright = check_brightness(image)
            
            # Annotation Analytics
            labels = parse_labels(lbl_path)
            lesion_density = len(labels)
            
            box_areas = []
            for line in labels:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w = float(parts[3])
                    h = float(parts[4])
                    box_areas.append(w * h)
            
            size_variance = np.var(box_areas) if len(box_areas) > 1 else 0
            
            # Basic Scoring Logic
            score = "Excellent"
            issues = []
            
            if blur_val < 30:
                issues.append("Severe Blur")
                score = "Poor"
            elif blur_val < 100 and score != "Poor":
                issues.append("Moderate Blur")
                score = "Questionable"
                
            if avg_bright < 30:
                issues.append("Very Dark")
                score = "Poor"
            elif avg_bright > 225:
                issues.append("Extreme Glare/Overexposed")
                score = "Poor"
                
            if lesion_density == 0:
                issues.append("No Annotations")
                if score not in ["Poor", "Questionable"]:
                    score = "Questionable"
                    
            if len(issues) == 0:
                if blur_val > 500 and 50 < avg_bright < 200:
                    score = "Excellent"
                else:
                    score = "Good"
                    
            if not issues:
                issues.append("None")
                
            audit_results.append({
                'image': img_name,
                'split': split,
                'resolution': resolution,
                'blur_score': round(blur_val, 2),
                'brightness': round(avg_bright, 2),
                'lesion_density': lesion_density,
                'quality_score': score,
                'issues': " | ".join(issues)
            })

    # Save full audit
    os.makedirs('reports', exist_ok=True)
    with open('reports/dataset_quality_audit.csv', 'w', newline='') as f:
        fieldnames = ['image', 'split', 'resolution', 'blur_score', 'brightness', 'lesion_density', 'quality_score', 'issues']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_results)
        
    # Generate Markdown Summary
    scores = {"Excellent": 0, "Good": 0, "Questionable": 0, "Poor": 0}
    for r in audit_results:
        scores[r['quality_score']] += 1
        
    with open('reports/dataset_quality_summary.md', 'w') as f:
        f.write("# QYRO Dataset v2 Quality Audit Summary\n\n")
        f.write("## Overview\n")
        f.write(f"Total Images Audited: {len(audit_results)}\n\n")
        f.write("## Score Distribution\n")
        for k, v in scores.items():
            f.write(f"- **{k}**: {v}\n")
            
        f.write("\n## Next Steps\n")
        f.write("Images marked as 'Poor' have been flagged in the candidate ignore manifest for manual review before removal.\n")
        
    # Generate Candidate Ignore Manifest
    review_dir = os.path.join(dataset_dir, 'review')
    os.makedirs(review_dir, exist_ok=True)
    
    with open(os.path.join(review_dir, 'candidate_ignore_manifest.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'split', 'issues', 'blur_score', 'brightness'])
        writer.writeheader()
        for r in audit_results:
            if r['quality_score'] == 'Poor':
                writer.writerow({
                    'image': r['image'],
                    'split': r['split'],
                    'issues': r['issues'],
                    'blur_score': r['blur_score'],
                    'brightness': r['brightness']
                })
                
    print("Audit complete! Reports and manifest generated.")

if __name__ == '__main__':
    main()
