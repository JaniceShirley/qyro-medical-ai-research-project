from ultralytics import YOLO

CHECKPOINT_PATH = 'experiments/detection/checkpoints/yolov8s_qyro_acne_v1_convergence_20260624_150252/best.pt'
DATA_YAML = 'datasets/skin/acne/final/detection/data.yaml'

def get_f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def main():
    model = YOLO(CHECKPOINT_PATH)
    
    iou = 0.60
    conf = 0.25
    
    print(f"Running final validation with IoU={iou}, Conf={conf}")
    metrics = model.val(data=DATA_YAML, split='val', iou=iou, conf=conf, save_json=False, plots=True)
    
    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    f1 = get_f1(p, r)
    
    print("--- Final Metrics ---")
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"mAP50: {map50}")
    print(f"mAP50-95: {map50_95}")
    print(f"F1 Score: {f1}")

if __name__ == '__main__':
    main()
