# QYRO YOLOv8s Convergence Recovery Plan (Phase 7D.3)
**Scientific Experiment Design & Recovery Roadmap**

---

## 1. Objective & Hypothesis

### Objective:
Investigate whether the underperformance of the initial `YOLOv8s` detector baseline (`mAP50 = 0.4309` vs `0.6354` for `YOLOv8n`) was due to structural capacity limitations or premature convergence failure caused by training hyperparameter constraints.

### Hypothesis:
The `YOLOv8s` model was severely undertrained. Because it contains 3.5x more parameters than `YOLOv8n` (11.2M vs 3.2M), it has a more complex loss surface and requires a slower optimization step and longer convergence runway. 
* The previous run was stopped at **Epoch 26** (peaking at Epoch 16) due to an aggressive patience setting (`patience = 10` epochs) combined with a high initial learning rate (`lr0 = 0.01`).
* Calibrating the learning rate to `lr0 = 0.002` (5x smaller) and expanding the early stopping patience to `30` epochs over a `100` epoch budget will stabilize gradient descent and allow the model to fully adapt, outperforming the `YOLOv8n` baseline.

---

## 2. Quantitative Metric Comparison

The table below contrasts the historic baseline performance, the failed YOLOv8s baseline, and the projected target metrics for the Phase 7D.3 Convergence Recovery experiment:

| Metric / Parameter | YOLOv8n Baseline (Stage 2) | YOLOv8s Failed Baseline | **YOLOv8s Convergence (Expected)** | Target / Success Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **mAP50** | 0.6354 | 0.4309 | **0.685 – 0.720** | **> 0.68** |
| **Precision** | 0.6388 | 0.4620 | **0.670 – 0.710** | **> 0.65** |
| **Recall** | 0.6225 | 0.4508 | **0.660 – 0.690** | **> 0.65** |
| **Initial LR (lr0)**| 0.01 | 0.01 | **0.002** | — |
| **Patience** | 10 | 10 | **30** | — |
| **Epochs** | 50 (resumed) | 26 (early stopped) | **100** | — |
| **Duration (secs)** | 544.6s | 270.0s | **~1,040s (17.3 mins)**| — |
| **Peak VRAM** | 3.29 GB | 4.20 GB | **4.20 GB** | **< 8.00 GB** |

---

## 3. Convergence Rationale

### Learning Rate Calibration:
`YOLOv8s` features a deeper, wider architecture. Fine-tuning it with a large learning rate of `0.01` under AdamW causes gradient steps to oscillate aggressively around narrow local minima. Reducing `lr0` to `0.002` slows down the updates, enabling fine-grained weight adjustments that are crucial for capturing detailed facial lesion shapes.

### Patience Expansion:
In the failed run, early stopping triggered at Epoch 26. This was only 10 epochs past the peak at Epoch 16, which itself was only 13 epochs past the learning rate warmup phase (3 epochs). By increasing the patience to `30` epochs, we provide a sufficient window for the model to bypass temporary metric plateaus (which are common in multi-scale object detection optimization) and continue optimizing.

---

## 4. Hardware and VRAM Profile

* **Target GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB profile, 7.96 GB addressable)
* **Expected VRAM Consumption**:
  * **Training Phase**: `3.72 GB`
  * **Validation Phase (Peak)**: `4.20 GB`
  * **Safety Margin**: `3.76 GB` headroom (extremely safe; zero OOM risk under standard batch size 16).
* **Expected Training Duration**:
  * Standard epoch duration is `10.4 seconds`.
  * For 100 epochs, total training time is estimated at **1,040 seconds (~17.3 minutes)**.

---

## 5. Risks and Rollback Plan

### Risk 1: Stagnant Convergence
* *Description*: The lower learning rate of `0.002` might lead to extremely slow optimization, causing the model to underfit within the 100-epoch budget.
* *Mitigation*: The `ExperimentLogger` will monitor validation mAP50. If mAP50 remains below `0.30` by Epoch 30, we will halt training and restart with `lr0 = 0.005`.

### Risk 2: Overfitting on Small Dataset
* *Description*: Scaling model size and epochs on a small dataset (Kurnaz split has 364 training images) increases the risk of overfitting.
* *Mitigation*: The patience parameter is capped at `30` epochs. If validation loss begins to diverge while training loss continues to fall, early stopping will automatically trigger and save the best checkpoint prior to overfit.

### Risk 3: Windows PyTorch Shared Memory OOM
* *Description*: Multi-processing workers on Windows can lead to memory leakages or page file errors.
* *Mitigation*: Enforce `workers: 0` in the configuration. The training script has an integrated `OOM Guard` which will intercept runtime CUDA OOM events and automatically decay the batch size (`16 -> 12 -> 8`) to preserve state.

---

## 6. Execution Specifications

### Terminal Command:
```powershell
python scripts/training/train_detection.py --config configs/detection_config_yolov8s_convergence.yaml
```

### Go/No-Go Recommendation:
**GO**.
The hardware profiles confirm that we have massive memory headroom (`3.76 GB` free) and the training duration is very short (`17.3 minutes`). Given that YOLOv8s has already demonstrated superior structural behavior (reducing overlapping boxes and pore false positives), it is highly likely that granting the model a proper learning rate and convergence budget will enable it to cross the `0.68` mAP50 success threshold.
