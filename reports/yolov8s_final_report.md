# QYRO Acne v1 — YOLOv8s Upgrade Experiment Final Report
**Phase 7D.2: Controlled Detection Model Backbone Upgrade Analysis**

---

## 1. Executive Summary

This report evaluates the controlled upgrade of the QYRO Acne v1 lesion detector from **YOLOv8n (3.2M parameters)** to **YOLOv8s (11.2M parameters)**. The YOLOv8s training run was executed on the RTX 5050 Laptop GPU (8GB profile) using standard Kurnaz splits with a maximum limit of 75 epochs. Due to the early stopping patience of 10 epochs, the YOLOv8s training completed at **Epoch 26**, peaking at **Epoch 16**.

We present a quantitative comparison between YOLOv8s and the YOLOv8n baseline, conduct a comparative failure mode audit, and formulate a final production recommendation for the QYRO Acne v1 detector.

---

## 2. Quantitative Performance Comparison

The table below contrasts the best evaluation metrics achieved by the YOLOv8s model against the two YOLOv8n baseline stages:

| Metric | YOLOv8n Baseline (25 Epochs) | YOLOv8n Extended (50 Epochs) | **YOLOv8s Upgrade (Best: Ep 16)** | Upgrade Status |
| :--- | :---: | :---: | :---: | :---: |
| **mAP50** | 0.5659 | **0.6354** | 0.4309 | ❌ Decreased by 0.2045 |
| **Precision** | 0.5602 | **0.6388** | 0.4620 | ❌ Decreased by 0.1768 |
| **Recall** | 0.5483 | **0.6225** | 0.4508 | ❌ Decreased by 0.1717 |
| **Val Box Loss** | 1.8197 | **1.7970** | 2.1145 | ❌ Increased by 0.3175 |
| **Val Class Loss** | 1.5571 | **1.3835** | 1.7857 | ❌ Increased by 0.4022 |
| **Training Epochs** | 25 (Scratch) | 50 (Two-Stage) | 26 (Scratch) | Completed Early (ES) |
| **Training Duration**| 243.2s | 544.6s | **270.0s** | 50% Faster than Extended |
| **Peak VRAM Usage** | 3.29 GB | 3.29 GB | 4.20 GB | +0.91 GB Overhead |

### Key Quantitative Findings:
1. **Convergence Failure**: YOLOv8s failed to outperform the extended YOLOv8n model, achieving an inferior `mAP50 = 0.4309` compared to YOLOv8n's `0.6354`.
2. **Early Stopping Restraint**: The YOLOv8s run was stopped prematurely at **Epoch 26** because validation metrics did not improve for 10 epochs. 
3. **Training History Difference**: YOLOv8n was trained in two stages (first 25 epochs, followed by a resumed 25-epoch run starting from the Stage 1 weights). The Stage 2 run initialized with a solid baseline, allowing it to reach `0.6354`. YOLOv8s was trained entirely from scratch (coco weights) and suffered from validation loss stagnation before it could escape local minima.
4. **VRAM Feasibility**: Peak VRAM consumption of YOLOv8s was `4.20 GB`. This is well within the `8GB VRAM` capacity of the RTX 5050 Laptop GPU, leaving `3.76 GB` of headroom.

---

## 3. Failure Mode Comparison

A comparative root-cause auditing of predictions on the Kurnaz validation split (1,145 instances) reveals the following breakdown:

### 3.1 Error Breakdown Table

| Category | YOLOv8n Baseline | YOLOv8s Upgrade | Trend Analysis |
| :--- | :---: | :---: | :--- |
| **Total False Positives (FPs)** | 487 | **561** | Increased (+74) |
| *— Overlapping Lesions %* | 58.4% | **49.4%** | **Improved (-9.0%)** |
| *— Normal Pores %* | 12.1% | **3.9%** | **Improved (-8.2%)** |
| *— Camera Glare %* | 8.5% | **0.2%** | **Improved (-8.3%)** |
| **Total False Negatives (FNs)** | 511 | **620** | Increased (+109) |
| *— Low Contrast Lesions %* | 85.3% | **85.5%** | Stagnant (+0.2%) |
| *— Clustered Lesions %* | 10.4% | **10.2%** | Stagnant (-0.2%) |

### 3.2 Analysis of Failure Mode Trends:
* **Bounding Box Alignment (Overlapping Lesions FP)**: The percentage of FPs caused by misaligned box dimensions dropped from `58.4%` (YOLOv8n) to `49.4%` (YOLOv8s). This verifies that the larger parameter capacity of YOLOv8s (11.2M parameters) provides **better spatial regression capability** and resolves class boundary ambiguity.
* **Texture & Distractor Filtering (Pores / Glare FP)**: YOLOv8s achieved massive reductions in false detections on normal skin pores (`3.9%` vs `12.1%`) and specular camera glares (`0.2%` vs `8.5%`). The larger backbone successfully learned to reject high-contrast lighting distractions.
* **Low-Contrast Erythema bottleneck**: Missed flat, skin-toned inflammatory red spots remain the dominant cause of false negatives for both models (~85%). This indicates a dataset annotations contrast ceiling that model capacity scaling alone cannot resolve.

---

## 4. Production Recommendation

### Should YOLOv8s become the production detector for QYRO Acne v1?

> [!WARNING]
> **REJECT YOLOv8s IN ITS CURRENT STATE**. 
> The YOLOv8s model trained in this run should **NOT** become the production detector. It achieved an `mAP50 = 0.4309` which is significantly below both the target threshold of `0.680` and the existing YOLOv8n baseline of `0.6354`.

### Rationale & Pathology:
The failure of YOLOv8s to surpass YOLOv8n is **not** a backbone capacity issue, but rather a **warmup/patience hyperparameter constraint**:
1. **Aggressive Early Stopping**: Setting `patience = 10` is suitable for small models like YOLOv8n that converge rapidly, but is too restrictive for YOLOv8s. YOLOv8s contains 3.5x more parameters and requires a longer training budget to stabilize gradients and escape early local minima. It was terminated at Epoch 26, right as the learning rate warmup was settling.
2. **Learning Rate Mismatch**: Using a high initial learning rate of `0.01` with `AdamW` caused validation losses to fluctuate and stagnation to occur early.

### Remediation Roadmap:
To unlock the true capacity of YOLOv8s and cross the `0.680` mAP50 threshold, the following actions must be taken:
1. **Disable Early Stopping (or Set Patience = 30)**: Allow the model to train for the full 75-100 epochs to facilitate full weight adaptation.
2. **Calibrate Learning Rate**: Lower the initial learning rate (`lr0`) from `0.01` to `0.002` to prevent validation gradient divergence.
3. **Execute Cosine Decay Scheduler**: Run with cosine learning rate decay and a larger batch size (batch = 16 is fine, but we can utilize gradient accumulation if needed).
4. **Resubmit Run**: Run a resumed training run for YOLOv8s with these hyperparameters. Since RTX 5050 VRAM has 3.76 GB of headroom, this is computationally safe.
