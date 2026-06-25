import os
import zipfile
import shutil

def main():
    zip_path = 'annotation_exports/batch_1_refined_yolo.zip'
    dataset_dir = 'datasets/skin/acne/final/detection'
    
    splits = ['valid', 'train', 'test']
    
    report_lines = []
    report_lines.append("# Phase 7D.6 — Annotation Refinement Report")
    report_lines.append("\n## Objective")
    report_lines.append("Integrate manually refined bounding box annotations from CVAT back into the primary dataset, replacing the original labels for Batch 1 images.")
    report_lines.append("\n## Diff Summary\n")
    report_lines.append("| Image | Original Boxes | Refined Boxes | Net Change |")
    report_lines.append("|-------|----------------|---------------|------------|")
    
    total_original = 0
    total_refined = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        txt_files = [f for f in zf.namelist() if f.endswith('.txt') and not f.endswith('train.txt') and not f.endswith('obj.names') and not f.endswith('obj.data')]
        
        for txt_file in txt_files:
            file_name = os.path.basename(txt_file)
            
            refined_content = zf.read(txt_file).decode('utf-8').strip().split('\n')
            refined_content = [line for line in refined_content if line.strip()]
            refined_count = len(refined_content)
            
            original_path = None
            backup_dir = None
            
            for split in splits:
                candidate = os.path.join(dataset_dir, split, 'labels', file_name)
                if os.path.exists(candidate):
                    original_path = candidate
                    backup_dir = os.path.join(dataset_dir, split, 'labels_backup_batch1')
                    break
                    
            if not original_path:
                print(f"Warning: original label for {file_name} not found in dataset.")
                continue
                
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, file_name)
            
            with open(original_path, 'r', encoding='utf-8') as f:
                original_content = f.read().strip().split('\n')
            original_content = [line for line in original_content if line.strip()]
            original_count = len(original_content)
            
            total_original += original_count
            total_refined += refined_count
            
            diff = refined_count - original_count
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            
            report_lines.append(f"| `{file_name.replace('.txt', '.jpg')}` | {original_count} | {refined_count} | {diff_str} |")
            
            shutil.copy2(original_path, backup_path)
            
            with open(original_path, 'wb') as f:
                f.write(zf.read(txt_file))
                
    report_lines.append(f"\n**Total Original Boxes:** {total_original}")
    report_lines.append(f"**Total Refined Boxes:** {total_refined}")
    
    net_total = total_refined - total_original
    net_str = f"+{net_total}" if net_total > 0 else str(net_total)
    report_lines.append(f"**Net Change:** {net_str} boxes")
    
    report_lines.append("\n## Next Steps")
    report_lines.append("The dataset is now updated with the refined labels. The original labels are safely backed up in `labels_backup_batch1` directories. The dataset is ready for the next detection retraining experiment.")
    
    with open('reports/phase7d6_annotation_refinement_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Processed refined labels. Backed up originals. Net change: {net_str} boxes.")

if __name__ == '__main__':
    main()
