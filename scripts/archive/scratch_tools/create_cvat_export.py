import os
import shutil
import zipfile

def create_cvat_export():
    batch_dir = 'annotation_batches/batch_1'
    export_dir = 'annotation_batches/cvat_export_temp'
    zip_path = 'annotation_batches/batch_1_cvat_import.zip'
    
    # Create export structure
    obj_train_data = os.path.join(export_dir, 'obj_train_data')
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(obj_train_data)
    
    # Write obj.names
    with open(os.path.join(export_dir, 'obj.names'), 'w') as f:
        f.write('acne\n')
        
    # Write obj.data
    with open(os.path.join(export_dir, 'obj.data'), 'w') as f:
        f.write('classes = 1\n')
        f.write('train = train.txt\n')
        f.write('names = obj.names\n')
        f.write('backup = backup/\n')
        
    # Get images and copy
    images_src = os.path.join(batch_dir, 'images')
    labels_src = os.path.join(batch_dir, 'labels')
    
    image_files = [f for f in os.listdir(images_src) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    train_txt_lines = []
    for img in image_files:
        base = os.path.splitext(img)[0]
        lbl = base + '.txt'
        
        # Copy image
        shutil.copy2(os.path.join(images_src, img), os.path.join(obj_train_data, img))
        # Copy label
        shutil.copy2(os.path.join(labels_src, lbl), os.path.join(obj_train_data, lbl))
        
        # Add to train.txt (CVAT convention usually requires this structure)
        train_txt_lines.append(f'obj_train_data/{img}\n')
        
    # Write train.txt
    with open(os.path.join(export_dir, 'train.txt'), 'w') as f:
        f.writelines(train_txt_lines)
        
    # Create ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, export_dir)
                zipf.write(file_path, arcname)
                
    # Cleanup temp
    shutil.rmtree(export_dir)
    
    print(f"Created CVAT export at {zip_path}")

if __name__ == '__main__':
    create_cvat_export()
