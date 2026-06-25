# QYRO Acne v1 - Training Readiness Optimization Report
**Phase 7A.5: Pre-Training Hardware Tuning & Success Calibration**

---

## 1. Laptop Optimization Profile: ASUS TUF F16 (RTX 5050)

To prevent thermal throttling and ensure stable multi-hour model training runs on the ASUS TUF F16, the following profiles are recommended:

### 1.1 ASUS Armoury Crate & Fan Profiles
* **Recommended Mode**: **Turbo Mode** (or **Manual Mode** with aggressive fan curves).
  - **Turbo Mode** increases the TGP (Total Graphics Power) limit of the RTX 5050, allowing higher stable core clocks.
  - Fans must be set to run at $\ge 85\%$ capacity when GPU temperature exceeds $75^{\circ}$C.
* **GPU Mode (MUX Switch)**: Set to **Ultimate** (or **Discrete GPU only**).
  - Bypasses integrated AMD/Intel graphics processing paths, reducing memory copy latencies and system RAM overhead.

### 1.2 Thermal Safeguard Recommendations
* **Core Limit Alert**: The training runner includes utility functions to poll GPU core temperature via `nvidia-smi` (or PyTorch CUDA interfaces).
* **Safeguard Interruption**: If the GPU core temperature exceeds **$85^{\circ}$C** or CPU package exceeds **$92^{\circ}$C** for longer than 3 minutes:
  - **Action**: Pause training execution, clear CUDA cache, and sleep for 5 minutes to allow fans to cool the laptop down before resuming.
* **Cooling Recommendations**: Elevate the back of the laptop by 1-2 cm (or use an active cooling pad) to maximize intake airflow.

### 1.3 Windows OS & CUDA Optimizations
* **Hardware-Accelerated GPU Scheduling (HAGS)**: **ENABLE** in Windows Graphics Settings. HAGS offloads scheduling tasks to the dedicated GPU scheduling processor, reducing CPU driver overhead.
* **NVIDIA Control Panel**: Set Power Management Mode to **"Prefer maximum performance"** for PyTorch execution.
* **CUDA recommendations**: Use **CUDA 12.1+** to compile/run PyTorch. CUDA 12 features a more efficient memory allocator with reduced fragmentation overheads, maximizing the usable 8GB VRAM limit.

---

## 2. Baseline Success Targets (QYRO Acne v1)

Below are the realistic, locked-in performance targets for our initial training run.

### 2.1 Model 1: Lesion Detection (YOLOv8n)
* **Target mAP@0.5**: $\ge \mathbf{78.0\%}$
* **Target Recall**: $\ge \mathbf{72.0\%}$
* **Target Precision**: $\ge \mathbf{75.0\%}$
* *Clinical Context*: Detection needs to minimize false-positive comedones caused by specular reflections, while maintaining high recall for active inflammatory papules and pustules.

### 2.2 Model 2: Subtype Classification (EfficientNet-B0)
* **Target Macro F1-Score**: $\ge \mathbf{70.0\%}$
* **Target Overall Accuracy**: $\ge \mathbf{75.0\%}$
* **Expected Per-Class Metric Ranges**:
  - `open_comedo` & `cystic` (High representation) $\rightarrow$ F1 $\ge 82\%$
  - `pustular` & `papular` (Moderate representation) $\rightarrow$ F1 $\ge 75\%$
  - `scar` & `mechanica` (Severely imbalanced, $<20$ samples) $\rightarrow$ F1 $\ge 30-40\%$ (Due to class weight limitations).

### 2.3 Model 3: Severity Grading (EfficientNet-B0 Ordinal)
* **Target Macro F1-Score**: $\ge \mathbf{72.0\%}$
* **Target Quadratic Weighted Kappa (QWK)**: $\ge \mathbf{0.75}$
* **Acceptable Error Range**: Mean Absolute Error (MAE) must be $< \mathbf{0.35}$.
  - Any classification error must fall within $\pm 1$ adjacent stage (e.g. predicting Stage 2 for Stage 1 is acceptable, but predicting Stage 3 for Stage 1 is a clinical validation failure).

### 2.4 Google SCIN Robustness Validation (OOD Holdout)
* **Maximum Acceptable Metric Degradation**: $\le \mathbf{12.0\%}$
  - The drop in accuracy/F1-score when evaluating models on the out-of-distribution (OOD) `robustness_holdout` split (Google SCIN) compared to the internal test split must not exceed 12%.
  - Metrics must remain stable across Fitzpatrick Skin Types V-VI (within 8% variance of FST I-IV performance).
