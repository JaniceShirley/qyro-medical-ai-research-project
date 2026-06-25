# QYRO YOLOv8s Controlled Upgrade — Epoch 30 Training Report (Not Applicable)

## 1. Run Identification
* **Experiment Name**: `yolov8s_qyro_acne_v1`
* **Run ID**: `yolov8s_qyro_acne_v1_20260624_144501`
* **Current Epoch**: N/A (Run completed at Epoch 26)

## 2. Early Stopping Trigger Details
The training run was terminated early at **Epoch 26** due to Ultralytics YOLOv8's `EarlyStopping` criterion.
* **Stop Reason**: No validation metric improvement (`mAP50` and validation losses) observed for `10` consecutive epochs.
* **Best Epoch**: **Epoch 16**
* **Best validation mAP50**: `0.4309`

## 3. Analysis of Terminal State
* **Training Box Loss at Epoch 26**: `2.0360` (steady decline from `2.0720` at Epoch 21)
* **Training Class Loss at Epoch 26**: `1.6550` (steady decline from `1.7258` at Epoch 21)
* **Validation Box Loss**: Stagnated around `2.05 - 2.14` from Epoch 16 onwards.
* **Validation Class Loss**: Showed signs of slight overfitting, rising from `1.7857` (Epoch 16) to `1.8685` (Epoch 26).
* **Early Term Summary**: The model plateaued early due to the combined effect of a high initial learning rate (`lr0 = 0.01`) and small early stopping patience (`patience = 10`) under a larger model capacity (`YOLOv8s`).

No training or validation data exists for Epoch 30.
