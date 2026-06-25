import os
import shutil

def main():
    src_dir = 'datasets/acne_v1_original'
    dst_dir = 'datasets/acne_v2_curated'
    
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        curr_dst = os.path.join(dst_dir, rel_path)
        os.makedirs(curr_dst, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(curr_dst, file)
            
            # Hard link image files, copy label files and other metadata
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.link(src_file, dst_file)
            else:
                shutil.copy2(src_file, dst_file)
                
    print(f"Successfully cloned {src_dir} to {dst_dir} using hard links for images.")

if __name__ == '__main__':
    main()
