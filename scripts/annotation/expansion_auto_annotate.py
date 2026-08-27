import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from scripts.annotation.quality_filter import filter_bounding_boxes

CHECKPOINT_PATH = "models/production/qyro_acne_v1_best.pt"
CANDIDATE_REPORT = "reports/candidate_dataset_report.json"
OUTPUT_DIR = "datasets/acne_v1_expansion/auto_annotations"
QUALITY_REPORT_PATH = "reports/auto_annotation_quality_report.json"

def main():
    print("=== Phase 2: Auto-Annotation & Quality Filtering Pipeline ===")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    if not os.path.exists(CANDIDATE_REPORT):
        raise FileNotFoundError(f"Candidate report not found at {CANDIDATE_REPORT}")

    with open(CANDIDATE_REPORT, "r") as f:
        candidate_data = json.load(f)

    candidates = candidate_data["candidates"]
    print(f"Loaded {len(candidates)} candidate images from report.")

    print(f"Loading production YOLOv8s model from {CHECKPOINT_PATH}...")
    model = YOLO(CHECKPOINT_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = {
        "total_candidate_images": len(candidates),
        "images_successfully_annotated": 0,
        "images_rejected_no_boxes": 0,
        "total_generated_boxes": 0,
        "boxes_rejected": 0,
        "boxes_accepted": 0,
        "annotated_candidates": []
    }

    print("Running auto-annotation and quality filtering...")
    for idx, cand in enumerate(candidates):
        if (idx + 1) % 500 == 0 or (idx + 1) == len(candidates):
            print(f"  Processed {idx + 1}/{len(candidates)} images...")

        img_path = cand["image_path"]
        if not os.path.exists(img_path):
            continue

        # Predict
        results = model.predict(source=img_path, conf=0.25, iou=0.60, verbose=False)
        boxes = results[0].boxes

        raw_boxes = []
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                raw_boxes.append({
                    "cls": int(clss[i]),
                    "cx": float(xywhn[i][0]),
                    "cy": float(xywhn[i][1]),
                    "w": float(xywhn[i][2]),
                    "h": float(xywhn[i][3]),
                    "conf": float(confs[i])
                })

        stats["total_generated_boxes"] += len(raw_boxes)

        # Apply Quality Filter
        accepted_boxes, rej_count = filter_bounding_boxes(raw_boxes, min_size=0.005, max_area=0.80, min_conf=0.25)
        stats["boxes_rejected"] += rej_count
        stats["boxes_accepted"] += len(accepted_boxes)

        if len(accepted_boxes) == 0:
            stats["images_rejected_no_boxes"] += 1
            continue

        stats["images_successfully_annotated"] += 1

        # Save annotation file
        img_id = cand["image_id"]
        label_filename = f"{cand['source_dataset']}_{img_id}.txt"
        label_path = os.path.join(OUTPUT_DIR, label_filename)

        with open(label_path, "w") as f:
            for b in accepted_boxes:
                f.write(f"0 {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

        cand_info = dict(cand)
        cand_info["label_filename"] = label_filename
        cand_info["label_path"] = label_path
        cand_info["box_count"] = len(accepted_boxes)
        cand_info["avg_confidence"] = round(float(sum(b["conf"] for b in accepted_boxes) / len(accepted_boxes)), 4)
        cand_info["accepted_boxes"] = accepted_boxes
        stats["annotated_candidates"].append(cand_info)

    avg_boxes_per_img = round(stats["boxes_accepted"] / max(1, stats["images_successfully_annotated"]), 2)
    stats["average_boxes_per_image"] = avg_boxes_per_img

    print("\n=== Auto-Annotation Summary ===")
    print(f"Total Candidate Images: {stats['total_candidate_images']}")
    print(f"Images Successfully Annotated: {stats['images_successfully_annotated']}")
    print(f"Images Rejected (No valid boxes): {stats['images_rejected_no_boxes']}")
    print(f"Total Generated Boxes: {stats['total_generated_boxes']}")
    print(f"Boxes Rejected: {stats['boxes_rejected']}")
    print(f"Boxes Accepted: {stats['boxes_accepted']}")
    print(f"Average Boxes / Image: {avg_boxes_per_img}")

    os.makedirs(os.path.dirname(QUALITY_REPORT_PATH), exist_ok=True)
    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved Auto-Annotation Quality Report to {QUALITY_REPORT_PATH}")

if __name__ == "__main__":
    main()
