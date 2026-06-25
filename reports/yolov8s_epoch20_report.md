# QYRO YOLOv8s Controlled Upgrade — Epoch 20 Training Status Report

## 1. Run Identification
* **Experiment Name**: `yolov8s_qyro_acne_v1`
* **Run ID**: `yolov8s_qyro_acne_v1_20260624_144501`
* **Current Epoch**: 20 / 75
* **Execution Device**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB profile)

## 2. Key Metrics Summary (Epoch 20)

| Metric | Training Value | Validation Value |
| :--- | :---: | :---: |
| **Box Loss** | 2.1082 | 2.2330 |
| **Class Loss** | 1.7257 | 1.9716 |
| **Precision (B)** | — | 0.3532 |
| **Recall (B)** | — | 0.3886 |
| **mAP50 (B)** | — | 0.3233 |
| **mAP50-95 (B)** | — | 0.1099 |
| **VRAM Usage** | — | 3.72 GB |

## 3. Convergence & Anomaly Analysis

### Loss Analysis
* **Training Box Loss**: Continues to decline steadily, dropping from `2.1838` (Epoch 10) to `2.1082` (Epoch 20).
* **Training Class Loss**: Dropped from `1.8261` (Epoch 10) to `1.7257` (Epoch 20).
* **Validation Box Loss**: Decreased significantly from `3.0576` (Epoch 10) to `2.2330` (Epoch 20), indicating strong generalization of spatial coordinates.
* **Validation Class Loss**: Cleared all initial instability (`nan`/`inf`) and dropped from `4.6150` (Epoch 10) to `1.9716` (Epoch 20), proving the classification capability of YOLOv8s is converging properly.

### Performance Analysis
* **Metric Growth**:
  * Precision increased from `0.0230` to `0.3532` (+0.3302).
  * Recall increased from `0.0882` to `0.3886` (+0.3004).
  * mAP50 increased from `0.0116` to `0.3233` (+0.3117).
  * mAP50-95 increased from `0.0043` to `0.1099` (+0.1056).
* **Anomaly Detection**: 
  * *Overfitting*: Validation losses are decreasing along with training losses. No overfitting detected.
  * *Plateauing*: Learning rate is active. Performance is climbing rapidly. No plateau.
  * *Exploding Loss*: None.
  * *OOM*: None. VRAM usage remains flat at `3.72 GB`.
  * *Early Stopping*: Not triggered.

## 4. Next Phase Action Items
* Continue monitoring training run up to Epoch 30.
* Monitor metric rise. If convergence trends hold, we expect `mAP50` to cross `0.50` near Epoch 30-35.
