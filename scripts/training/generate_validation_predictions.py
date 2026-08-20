import os
import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Generate validation prediction overlays for QYRO Detector")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model weights (.pt)")
    parser.add_argument("--val-dir", type=str, default=r"C:\Users\KARTHIK V\OneDrive\Desktop\TEMP-QYRO\workspace\datasets\curated\dataset_v2_export\images\val", help="Path to validation images directory")
    parser.add_argument("--out-dir", type=str, default=r"C:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI\experiments\detection\QYRO_Acne_v2\sanity_predictions", help="Output directory for annotated predictions")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to predict and save")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights not found at: {args.model}")
        return
        
    if not os.path.exists(args.val_dir):
        print(f"[ERROR] Validation images directory not found at: {args.val_dir}")
        return
        
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Get first N validation images
    valid_exts = ('.jpg', '.jpeg', '.png')
    img_files = [f for f in os.listdir(args.val_dir) if f.lower().endswith(valid_exts)]
    img_files.sort()
    target_imgs = img_files[:args.num_images]
    
    print(f"Running predictions on {len(target_imgs)} validation images...")
    
    for idx, img_name in enumerate(target_imgs):
        img_path = os.path.join(args.val_dir, img_name)
        results = model(img_path)
        for r_idx, result in enumerate(results):
            # result.plot() returns a numpy array representing the annotated image in BGR format
            annotated_img = result.plot()
            out_path = os.path.join(args.out_dir, img_name)
            cv2.imwrite(out_path, annotated_img)
            
    print(f"[SUCCESS] Saved {len(target_imgs)} validation predictions to: {args.out_dir}")

if __name__ == "__main__":
    main()
