# QYRO Acne v1 — YOLOv8s Upgrade Plan
**Phase 7D.2: Detection Backbone Scaling & Controlled Baseline Comparison**

---

## 1. Objective

The objective of this run is to scale up the lesion detector's parameter capacity by upgrading the backbone from **`YOLOv8n` (nano)** to **`YOLOv8s` (small)**. This controlled experiment keeps the data splits, image resolutions ($640 \times 640$), optimizer settings, and augmentation parameters identical to isolate the performance gains of the scaled backbone.

---

## 2. Model Specifications & Comparison

| Specification | YOLOv8n (Previous) | YOLOv8s (Upgraded) | Impact |
| :--- | :---: | :---: | :--- |
| **Model Weight** | `yolov8n.pt` | `yolov8s.pt` | Larger capacity weights |
| **Parameters** | 3.2 Million | 11.2 Million | **3.5x capacity scaling** |
| **GFLOPs** | 8.7 | 28.6 | Higher computation capability |
| **Backbone Channels** | 64 (max) | 128 (max) | Higher dimensional feature maps |
| **Output Layer Nodes**| Anchor-free regression | Anchor-free regression | Unchanged architecture style |

---

## 3. Hardware & Compatibility Verification

### 3.1 VRAM & Memory Estimations (ASUS TUF F16 - RTX 5050 Laptop GPU)
* **RTX 5050 VRAM Available**: 7.96 GB (8,151 MB)
* **YOLOv8n Baseline VRAM Used**: **3.29 GB** (at batch size 16)
* **YOLOv8s Expected VRAM Consumption**: **5.2 GB – 5.8 GB** (at batch size 16)
  * *Safety Profile*: Safe. The 5.8 GB peak leaves a 2.1 GB headroom under the 8GB limit, preventing CUDA out-of-memory (OOM) faults during batch loading.
  * *Fallback*: If random background allocations (like Epic Games Launcher WDDM threads) trigger OOM, the custom `oom_guard.py` wrapper in `train_detection.py` will automatically decay the batch size to 12 or 8 and retry execution gracefully.

### 3.2 Compatibility Matrix
* **Checkpoint Manager**: **COMPATIBLE**. The script `train_detection.py` copies YOLO's internal outputs (`best.pt` and `last.pt`) from the run folder directly. The checkpoint structure remains unchanged.
* **TensorBoard Logging**: **COMPATIBLE**. YOLOv8 steps write scalar values and confusion matrix plots automatically to the TensorBoard directory linked via the run ID.
* **Experiment Logger**: **COMPATIBLE**. Config snapshots, system hardware telemetry, and run statuses are parsed and logged identically.
* **Resume Functionality**: **COMPATIBLE**. Specifying the `--resume` flag will automatically load the latest `last.pt` checkpoint for the `yolov8s_qyro_acne_v1` project and pick up training.

---

## 4. Expected Performance Projections

Based on our Phase 7D.1 failure analysis, here are the estimated gains for the scaled backbone:

1. **Resolution of Overlapping Detections (Precision + Recall)**:
   * Upgrading from 3.2M parameters to 11.2M parameters enables the bounding box regression head to align borders more tightly.
   * This is projected to reduce the rate of offset bounding boxes (overlapping lesions FP category), yielding **+3.0% mAP50**.
2. **Detection of Low-Contrast Comedones (Recall)**:
   * The 128-channel deep feature layers in YOLOv8s capture minor color contrast shifts (grayscale differences $< 12$ levels) more effectively than the nano network.
   * This is projected to resolve flat, low-contrast FNs, yielding **+2.0% Recall**.
3. **Crowded Breakout Resolution**:
   * YOLOv8s maintains higher resolution spatial feature activation grids, preventing Non-Maximum Suppression (NMS) from merging adjacent spots.
   * This is projected to reduce cluster FN counts, yielding **+1.5% mAP50**.

* **Overall mAP50 Gain Projection**: **+5.5% to +7.0%** (expected final validation mAP50 of **`0.69 – 0.71`**, crossing the success target threshold of **`0.68`**).

---

## 5. Projections & Resource Checklist

* **Expected VRAM Consumption**: **5.2 GB – 5.8 GB** (RTX 5050 Laptop GPU)
* **Expected Training Time**: **~18 to 22 minutes** (approx. 15 seconds per epoch * 75 epochs total)
* **Expected Performance Gain**: **+5.5% to +7.0% mAP50** (Target validation mAP50: **`0.69 – 0.71`**)

---

## 6. Execution Command

To launch the controlled baseline experiment run using the newly created `YOLOv8s` configuration:

```powershell
python scripts/training/train_detection.py --config configs/detection_config_yolov8s.yaml
```
