import os
import csv
from ultralytics import YOLO

CHECKPOINT_PATH = 'experiments/detection/checkpoints/yolov8s_qyro_acne_v1_convergence_20260624_150252/best.pt'
DATA_YAML = 'datasets/skin/acne/final/detection/data.yaml'

def get_f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def main():
    model = YOLO(CHECKPOINT_PATH)
    os.makedirs('reports', exist_ok=True)
    
    # --- TASK A: NMS IoU Threshold Sweep ---
    print("Starting Task A: NMS IoU Threshold Sweep")
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    baseline_conf = 0.001 # Default val confidence
    
    task_a_results = []
    for iou in iou_thresholds:
        print(f"Evaluating IoU={iou} (conf={baseline_conf})")
        metrics = model.val(data=DATA_YAML, split='val', iou=iou, conf=baseline_conf, save_json=False, plots=False, verbose=False)
        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        f1 = get_f1(p, r)
        task_a_results.append({
            'iou': iou,
            'precision': p,
            'recall': r,
            'map50': map50,
            'map50_95': map50_95,
            'f1': f1
        })
        
    with open('reports/nms_threshold_sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iou', 'precision', 'recall', 'map50', 'map50_95', 'f1'])
        writer.writeheader()
        writer.writerows(task_a_results)
        
    # Find top 3 IoU thresholds based on F1
    sorted_iou = sorted(task_a_results, key=lambda x: x['f1'], reverse=True)
    top_3_iou = [x['iou'] for x in sorted_iou[:3]]
    
    # --- TASK B: Confidence Threshold Sweep ---
    print("Starting Task B: Confidence Threshold Sweep")
    conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    baseline_iou = 0.60 # Standard baseline IoU for YOLO
    
    task_b_results = []
    for conf in conf_thresholds:
        print(f"Evaluating conf={conf} (iou={baseline_iou})")
        metrics = model.val(data=DATA_YAML, split='val', iou=baseline_iou, conf=conf, save_json=False, plots=False, verbose=False)
        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50
        f1 = get_f1(p, r)
        task_b_results.append({
            'conf': conf,
            'precision': p,
            'recall': r,
            'map50': map50,
            'f1': f1
        })
        
    with open('reports/confidence_threshold_sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['conf', 'precision', 'recall', 'map50', 'f1'])
        writer.writeheader()
        writer.writerows(task_b_results)
        
    # Find top 3 Conf thresholds based on F1
    sorted_conf = sorted(task_b_results, key=lambda x: x['f1'], reverse=True)
    top_3_conf = [x['conf'] for x in sorted_conf[:3]]
    
    # --- TASK C: Combined Optimization Grid ---
    print("Starting Task C: Combined Threshold Grid")
    print(f"Top 3 IoU: {top_3_iou}")
    print(f"Top 3 Conf: {top_3_conf}")
    
    task_c_results = []
    for iou in top_3_iou:
        for conf in top_3_conf:
            print(f"Evaluating grid: iou={iou}, conf={conf}")
            metrics = model.val(data=DATA_YAML, split='val', iou=iou, conf=conf, save_json=False, plots=False, verbose=False)
            p = metrics.box.mp
            r = metrics.box.mr
            map50 = metrics.box.map50
            f1 = get_f1(p, r)
            task_c_results.append({
                'iou': iou,
                'conf': conf,
                'precision': p,
                'recall': r,
                'map50': map50,
                'f1': f1
            })
            
    with open('reports/combined_threshold_grid.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iou', 'conf', 'precision', 'recall', 'map50', 'f1'])
        writer.writeheader()
        writer.writerows(task_c_results)
        
    print("Done generating all sweep CSVs.")

if __name__ == "__main__":
    main()
