# QYRO Acne v1 - Training Pipeline Report
**Phase 7B: Training Pipeline Engineering & Smoke-Test Verification**

---

## 1. Executive Summary

This report documents the implementation and validation of the training pipelines for **QYRO Acne v1**. 

To verify end-to-end code compatibility, dataset pipelines, and logging before launching full-scale runs, we successfully executed a **1-epoch GPU smoke test** for all three core pipelines on the target hardware.

* **Pipeline Verification Status:** **ALL PASSED ✅**
* **Target Hardware Profile:** ASUS TUF F16 (RTX 5050 Laptop GPU 8GB VRAM)
* **Local Run Device:** GPU Mode (CUDA 13.0, PyTorch 2.12.0+cu130, AMP Enabled)
- **Baseline Batch Sizes**: YOLOv8n Detection = 16, EfficientNet-B0 Subtype = 32, Severity = 32
- **Checkpoints Serials**: Standardized on saving `best.pt`, `last.pt`, `metrics.json`, `training_log.csv`, and `config_snapshot.yaml` inside each run directory.
- **Windows Compatibility Fix**: Set `workers = 0` (YOLO) and `num_workers = 0` (EfficientNet) to prevent Windows shared memory file mapping errors (`error code 2`).

---

## 2. Smoke-Test Verification Results

### 2.1 Lesion Detection Pipeline (`train_detection.py`)
* **Backbone**: YOLOv8n (nano)
* **Smoke-Test Execution**: **PASSED** (on GPU `RTX 5050`)
* **Epochs Run**: 1 epoch
* **Final Bounding Box Loss**: 2.436
* **Final Class Loss**: 2.656
* **Final DFL Loss**: 1.623
* **Validation mAP50**: 0.0274
* **Validation mAP50-95**: 0.00756
* **Weights Saved**: `best.pt` and `last.pt` copied to [experiments/detection/checkpoints/](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/experiments/detection/checkpoints/)
* **Logging Location**: [experiments/detection/runs/](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/experiments/detection/runs/)

### 2.2 Subtype Classification Pipeline (`train_subtype.py`)
* **Backbone**: EfficientNet-B0 (via `timm`, loaded with ImageNet pretrained weights)
* **Smoke-Test Execution**: **PASSED** (on GPU `RTX 5050`)
* **Epochs Run**: 1 epoch (classification head training)
* **Training Loss**: 2.9970
* **Validation Loss**: 2.6474
* **Validation Accuracy**: 23.92%
* **Macro F1-Score**: 12.82% (untrained baseline)
* **Class Weights**: Applied class weights dynamically calculated from training split.
* **Weights Saved**: `best.pt`, `last.pt`, and periodic `epoch_0.pt` written to [experiments/subtype/checkpoints/](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/experiments/subtype/checkpoints/)

### 2.3 Severity Grading Pipeline (`train_severity.py`)
* **Backbone**: EfficientNet-B0 Ordinal Regression (via `timm`, loaded with ImageNet pretrained weights)
* **Smoke-Test Execution**: **PASSED** (on GPU `RTX 5050`)
* **Epochs Run**: 1 epoch
* **Training Loss (Ordinal BCE)**: 1.3958
* **Validation Loss**: 1.0206
* **Validation Accuracy**: 31.58%
* **Macro F1-Score**: 21.54%
* **Mean Absolute Error (MAE)**: 0.8947
* **Quadratic Weighted Kappa (QWK)**: 0.1149
* **Weights Saved**: `best.pt`, `last.pt`, and periodic `epoch_0.pt` written to [experiments/severity/checkpoints/](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/experiments/severity/checkpoints/)

---

## 3. Implementation of Approved Safeguards

1. **Dependency Checker**:
   - Implemented a check at the beginning of all scripts to ensure critical modules (`timm`, `torch`, `torchvision`, `ultralytics`, `tensorboard`, etc.) are installed and version-compatible, immediately halting execution with clear logs if a dependency is missing.
2. **NaN-Loss Protection**:
   - Integrated `torch.isnan(loss)` checking at each training step in classification and severity pipelines. Training immediately halts and raises a `ValueError` if a NaN loss is detected, protecting model weights from corruption.
3. **Gradient Clipping**:
   - Enforced L2 gradient clipping using `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` during backward steps in EfficientNet pipelines to prevent gradient explosion and stabilize convergence.
4. **VRAM and Caching monitoring**:
   - Logged CUDA allocated and reserved VRAM at each epoch boundary directly to TensorBoard to track memory leaks.
5. **SSD Cache Strategy**:
   - Enabled dataloaders to stream images directly from the TUF laptop's SSD to RAM on-demand, caching label tensors only, keeping RAM footprint under 120 MB.
6. **Graceful OOM Fallback**:
   - Wrapped dataloader and training loop initialization in exception handlers. If CUDA OOM is raised:
     - Clear CUDA allocator cache (`torch.cuda.empty_cache()`).
     - Decays batch size (16 $\rightarrow$ 12 $\rightarrow$ 8 $\rightarrow$ 4 for YOLO; 32 $\rightarrow$ 24 $\rightarrow$ 16 $\rightarrow$ 8 for EfficientNet).
     - Retries execution.

---

## 4. Expected Bottlenecks & Recommendations

1. **Windows Multiprocessing Dataloader Workaround**:
   - Setting `num_workers > 0` on Windows with PyTorch causes random shared memory allocation failures (`error code 2`). To guarantee pipeline stability, keep `num_workers = 0` (or `workers = 0`) on Windows machines. Since the files are streamed from SSD, CPU dataloading bottleneck is minimal.
2. **Thermal Throttling on TUF Laptop**:
   - ASUS TUF F16 laptop GPU/CPU can experience thermal throttling under heavy continuous load. Developers must ensure Armoury Crate is set to "Turbo" or "Manual" with 100% fan speed curves.
3. **HuggingFace Hub Model Download cold-start**:
   - First-time loading of `efficientnet_b0` from timm requires internet connection to download weights. Verified that download works and models load successfully.
