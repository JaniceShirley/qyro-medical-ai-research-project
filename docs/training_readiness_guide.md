# QYRO Acne v1 - Training Readiness Guide

This guide details the failure recovery procedures, reproducibility settings, and memory-saving techniques configured for the **QYRO Acne v1** training pipeline.

---

## 1. Reproducibility & Performance Tuning

To guarantee stable experiment tracking, the system includes a dedicated seeding utility: [seed_everything.py](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/scripts/utils/seed_everything.py).

### 1.1 Seeding Modes
Developers can toggle between two modes using the `deterministic` flag:
1. **Strict Determinism Mode (`deterministic=True`)**:
   * **Seeding**: Python `random`, `numpy`, and PyTorch `manual_seed` are locked to the target seed.
   * **CuDNN Constraints**: Sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
   * **Workspace Settings**: Sets `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'` and runs `torch.use_deterministic_algorithms(True)` to enforce deterministic algorithms for operations like convolutions and grid sampling.
   * **Tradeoff**: Guarantees identical floating-point outputs on identical hardware, but reduces convolution performance by up to $15-20\%$.
2. **Performance Mode (`deterministic=False`)**:
   * **Seeding**: Basic random and NumPy states remain seeded.
   * **CuDNN Search**: Sets `torch.backends.cudnn.deterministic = False` and `torch.backends.cudnn.benchmark = True`. This enables the CuDNN auto-tuner to dynamically benchmark and select the fastest kernel for the current tensor layouts.
   * **Tradeoff**: Maximizes GPU core utilization and speeds up training, but minor differences in floating-point aggregation may cause small, non-reproducible metric variations across runs.

---

## 2. Checkpoint Policy & Automatic Resume Strategy

### 2.1 Standard Saved Artifacts
Every training run must write a standardized checkpoint payload to its unique run directory under `experiments/{task_type}/runs/{run_id}/`:
* **`best.pt`**: Weights that achieved the optimal evaluation score (mAP50 for detection, Macro F1 for classification, QWK for severity).
* **`last.pt`**: State dictionary at the most recent epoch (weights, optimizer state, scheduler state, current epoch index).
* **`metrics.json`**: Static JSON containing overall final evaluation metrics.
* **`training_log.csv`**: Epoched progress CSV capturing loss, val_loss, and learning rate at each step.
* **`config_snapshot.yaml`**: Full execution snapshot including git revision, registry SHA256, and hardware profiles.

### 2.2 Automatic Resume Logic
Before starting a training run, the training wrapper script will execute the following checks:
```python
def check_and_resume(run_dir):
    last_checkpoint = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_checkpoint):
        print(f"Detected interrupted run. Resuming training from: {last_checkpoint}")
        # Load weights, optimizer, scheduler state and epoch counter
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    return 0
```

---

## 3. SSD Disk Cache Strategy

The ASUS TUF F16 features a fast NVMe SSD and 16GB system RAM. Because 16GB CPU memory can easily saturate if we load the entire uncompressed image dataset into RAM, QYRO v1 uses a **hybrid SSD cache strategy**:
* **Label Preloading**: Dataloaders load all label strings, bounding box text vectors, and path indexes into CPU memory during initialization.
* **Disk Streaming**: Dataloaders stream image files directly from the SSD on-demand during mini-batch generation.
* **Prefetching**: PyTorch dataloaders utilize `num_workers = 4`, `pin_memory = True`, and `prefetch_factor = 2` to fetch subsequent batches in parallel CPU threads, hiding disk I/O latency behind GPU computation.
* **YOLOv8 SSD cache**: YOLOv8 utilizes the `cache: "disk"` parameter, caching resized and structured images to the local SSD partition rather than RAM.

---

## 4. Automatic GPU Out-of-Memory (OOM) Recovery

If PyTorch throws a `RuntimeError: CUDA out of memory` during the first training epoch:
1. **Catch OOM exception**: Wrap the training loop initializer in a try-except block.
2. **Reduce Batch Size**: Cut the active batch size by half (e.g. from 32 to 16 for EfficientNet).
3. **Enable Gradient Accumulation**: Multiply the gradient accumulation step count by 2 to maintain the same effective batch size:
   $$\text{Effective Batch Size} = \text{Batch Size} \times \text{Accumulation Steps}$$
4. **Flush Cache**: Call `torch.cuda.empty_cache()` to free allocated fragmentation blocks.
5. **Resume**: Restart the training step with modified parameters.
