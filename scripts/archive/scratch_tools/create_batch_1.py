import os
import csv
import shutil

def main():
    csv_path = 'reports/detection_annotation_risk_queue.csv'
    dataset_dir = 'datasets/skin/acne/final/detection'
    
    batch_dir = 'annotation_batches/batch_1'
    images_dir = os.path.join(batch_dir, 'images')
    labels_dir = os.path.join(batch_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get unique images from CSV, preserving order
    unique_images = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['image']
            if img not in unique_images:
                unique_images.append(img)
                if len(unique_images) == 50:
                    break
                    
    # Find and copy files
    copied_count = 0
    missing = []
    
    # The dataset could be in train, valid, or test
    splits = ['valid', 'train', 'test']
    
    for img_name in unique_images:
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + '.txt'
        
        found = False
        for split in splits:
            src_img = os.path.join(dataset_dir, split, 'images', img_name)
            src_lbl = os.path.join(dataset_dir, split, 'labels', label_name)
            
            if os.path.exists(src_img) and os.path.exists(src_lbl):
                dst_img = os.path.join(images_dir, img_name)
                dst_lbl = os.path.join(labels_dir, label_name)
                
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                found = True
                copied_count += 1
                break
                
        if not found:
            missing.append(img_name)
            
    # Generate README
    readme_path = os.path.join(batch_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('# Annotation Refinement Batch 1\n\n')
        f.write(f'**Total Images:** {copied_count}\n\n')
        if missing:
            f.write(f'**Missing Images (Not Found):** {len(missing)}\n\n')
            
        f.write('## Included Images\n')
        for img in unique_images:
            if img not in missing:
                f.write(f'- `{img}`\n')
                
        if missing:
            f.write('\n## Missing Images\n')
            for img in missing:
                f.write(f'- `{img}`\n')
                
    print(f"Successfully copied {copied_count} images and labels.")
    if missing:
        print(f"Failed to find {len(missing)} images.")

if __name__ == '__main__':
    main()
