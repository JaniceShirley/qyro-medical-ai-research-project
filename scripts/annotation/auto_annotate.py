import os
import shutil
import zipfile
import argparse
from ultralytics import YOLO

CHECKPOINT_PATH = 'experiments/detection/checkpoints/yolov8s_qyro_acne_v1_convergence_20260624_150252/best.pt'

def auto_annotate(input_folder, output_zip):
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = YOLO(CHECKPOINT_PATH)
    
    export_dir = 'annotation_batches/auto_annotate_temp'
    obj_train_data = os.path.join(export_dir, 'obj_train_data')
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(obj_train_data)
    
    # Write metadata files for CVAT
    with open(os.path.join(export_dir, 'obj.names'), 'w') as f:
        f.write('acne\n')
        
    with open(os.path.join(export_dir, 'obj.data'), 'w') as f:
        f.write('classes = 1\n')
        f.write('train = train.txt\n')
        f.write('names = obj.names\n')
        f.write('backup = backup/\n')
        
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_txt_lines = []
    
    stats = {}
    
    print(f"Found {len(image_files)} images. Starting inference...")
    
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + '.txt'
        
        # Copy image to export dir
        shutil.copy2(img_path, os.path.join(obj_train_data, img_name))
        
        # Run inference
        results = model.predict(source=img_path, conf=0.25, iou=0.60, verbose=False)
        boxes = results[0].boxes
        
        # Save predictions in YOLO format
        label_path = os.path.join(obj_train_data, label_name)
        with open(label_path, 'w') as f:
            for box in boxes:
                # box format: normalized cx, cy, w, h
                # box.xywhn returns a tensor
                xywhn = box.xywhn[0].tolist()
                cls_id = int(box.cls[0].item())
                # ensure we output class 0 (acne)
                f.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                
        stats[img_name] = len(boxes)
        train_txt_lines.append(f'obj_train_data/{img_name}\n')
        
    # Write train.txt
    with open(os.path.join(export_dir, 'train.txt'), 'w') as f:
        f.writelines(train_txt_lines)
        
    # Create ZIP
    print(f"Creating ZIP archive at {output_zip}...")
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, export_dir)
                zipf.write(file_path, arcname)
                
    # Cleanup temp
    shutil.rmtree(export_dir)
    
    print("Done!")
    
    # Print stats
    print("\n--- Prediction Stats ---")
    for img, count in stats.items():
        print(f"{img}: {count} boxes predicted")
        
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto-annotate images using production YOLOv8s model.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing images to annotate.")
    parser.add_argument('--output_zip', type=str, required=True, help="Path to output CVAT-compatible ZIP.")
    
    args = parser.parse_args()
    auto_annotate(args.input_folder, args.output_zip)
