import os
import sys
import argparse
from ultralytics import YOLO

# Add root folder to import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 Segmentation model metrics.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model weights (.pt)")
    parser.add_argument("--data-yaml", type=str, default="datasets/skin/acne/AcneSCU_YOLO/dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate on ('train', 'val', 'test')")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights not found at: {args.model}")
        return

    if not os.path.exists(args.data_yaml):
        print(f"[ERROR] Dataset configuration not found at: {args.data_yaml}")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Running validation on '{args.split}' split with IoU={args.iou}, Conf={args.conf}...")
    metrics = model.val(
        data=args.data_yaml,
        split=args.split,
        iou=args.iou,
        conf=args.conf,
        save_json=False,
        plots=True
    )

    # Extract Bounding Box metrics
    box_p = float(metrics.box.mp)
    box_r = float(metrics.box.mr)
    box_map50 = float(metrics.box.map50)
    box_map50_95 = float(metrics.box.map)

    # Extract Mask Segmentation metrics
    mask_p = 0.0
    mask_r = 0.0
    mask_map50 = 0.0
    mask_map50_95 = 0.0
    if hasattr(metrics, 'seg') and metrics.seg is not None:
        mask_p = float(metrics.seg.mp)
        mask_r = float(metrics.seg.mr)
        mask_map50 = float(metrics.seg.map50)
        mask_map50_95 = float(metrics.seg.map)
    elif hasattr(metrics, 'masks') and metrics.masks is not None:
        mask_p = float(metrics.masks.mp)
        mask_r = float(metrics.masks.mr)
        mask_map50 = float(metrics.masks.map50)
        mask_map50_95 = float(metrics.masks.map)

    print("\n" + "=" * 40)
    print("      YOLO11 SEGMENTATION EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data_yaml} ({args.split} split)")
    print("-" * 40)
    print("Bounding Box Metrics:")
    print(f"  Precision: {box_p:.4f}")
    print(f"  Recall:    {box_r:.4f}")
    print(f"  mAP50:     {box_map50:.4f}")
    print(f"  mAP50-95:  {box_map50_95:.4f}")
    print("-" * 40)
    print("Mask Segmentation Metrics:")
    print(f"  Precision: {mask_p:.4f}")
    print(f"  Recall:    {mask_r:.4f}")
    print(f"  mAP50:     {mask_map50:.4f}")
    print(f"  mAP50-95:  {mask_map50_95:.4f}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
