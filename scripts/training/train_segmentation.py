import os
import sys
import argparse
import json
import yaml
import shutil
import ultralytics
from datetime import datetime
from pathlib import Path

# Add root folder to import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_dependencies():
    """Verifies that all required packages are installed."""
    required = ["torch", "pandas", "numpy", "yaml", "ultralytics"]
    print("Checking segmentation training pipeline dependencies...")
    import importlib
    for lib in required:
        try:
            importlib.import_module(lib)
            print(f"  [OK] '{lib}' is available.")
        except ImportError:
            print(f"  [ERROR] Missing required library: '{lib}'")
            sys.exit(1)

check_dependencies()

import torch
import cv2
from ultralytics import YOLO
from scripts.utils.seed_everything import seed_everything
from scripts.utils.experiment_logger import ExperimentLogger
from scripts.training.utils.config_loader import load_config
from scripts.training.utils.oom_guard import oom_clean, decay_batch_size, get_vram_usage

def generate_learning_curves(results_csv_path, output_dirs):
    """
    Generates training_loss.png, validation_loss.png, box_map50.png,
    mask_map50.png, and learning_rate.png from results.csv.
    Saves them in the specified output directories.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not os.path.exists(results_csv_path):
        print(f"Warning: results.csv not found at {results_csv_path}. Skipping plot generation.")
        return
        
    try:
        df = pd.read_csv(results_csv_path)
        # Strip whitespaces from column names
        df.columns = [c.strip() for c in df.columns]
        
        epochs = df["epoch"]
        
        # 1. training_loss.png
        plt.figure(figsize=(8, 5))
        train_loss_cols = [c for c in ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
        for col in train_loss_cols:
            plt.plot(epochs, df[col], label=col, linewidth=2)
        plt.title("YOLO11 Segmentation Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for out_dir in output_dirs:
            plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
        plt.close()
        
        # 2. validation_loss.png
        plt.figure(figsize=(8, 5))
        val_loss_cols = [c for c in ["val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss"] if c in df.columns]
        for col in val_loss_cols:
            plt.plot(epochs, df[col], label=col, linewidth=2)
        plt.title("YOLO11 Segmentation Validation Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for out_dir in output_dirs:
            plt.savefig(os.path.join(out_dir, "validation_loss.png"), dpi=150)
        plt.close()
        
        # 3. box_map50.png
        plt.figure(figsize=(8, 5))
        if "metrics/mAP50(B)" in df.columns:
            plt.plot(epochs, df["metrics/mAP50(B)"], label="Box mAP50", color="royalblue", linewidth=2)
        if "metrics/mAP50-95(B)" in df.columns:
            plt.plot(epochs, df["metrics/mAP50-95(B)"], label="Box mAP50-95", color="cornflowerblue", linestyle="--", linewidth=1.5)
        plt.title("Bounding Box mAP Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for out_dir in output_dirs:
            plt.savefig(os.path.join(out_dir, "box_map50.png"), dpi=150)
        plt.close()
        
        # 4. mask_map50.png
        plt.figure(figsize=(8, 5))
        if "metrics/mAP50(M)" in df.columns:
            plt.plot(epochs, df["metrics/mAP50(M)"], label="Mask mAP50", color="forestgreen", linewidth=2)
        if "metrics/mAP50-95(M)" in df.columns:
            plt.plot(epochs, df["metrics/mAP50-95(M)"], label="Mask mAP50-95", color="limegreen", linestyle="--", linewidth=1.5)
        plt.title("Mask Segmentation mAP Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for out_dir in output_dirs:
            plt.savefig(os.path.join(out_dir, "mask_map50.png"), dpi=150)
        plt.close()
        
        # 5. learning_rate.png
        plt.figure(figsize=(8, 5))
        lr_cols = [c for c in ["lr/pg0", "lr/pg1", "lr/pg2"] if c in df.columns]
        for col in lr_cols:
            plt.plot(epochs, df[col], label=col, linewidth=2)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for out_dir in output_dirs:
            plt.savefig(os.path.join(out_dir, "learning_rate.png"), dpi=150)
        plt.close()
        
        print(f"Generated learning curve plots in {output_dirs}")
    except Exception as e:
        print(f"Failed to generate learning curve plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="QYRO Acne YOLO11 Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/segmentation_config_yolo11.yaml", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Execute a 1-epoch sanity smoke test")
    parser.add_argument("--deterministic", type=bool, default=True, help="Toggle strict determinism vs performance benchmark")
    parser.add_argument("--resume", action="store_true", help="Automatically resume training from the latest checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Override model weight file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--name", type=str, default=None, help="Override logging/run project name")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_params = config["model"]
    opt_params = config["optimizer"]
    aug_params = config["augmentation"]
    exp_params = config["experiment"]
    
    # Apply command line overrides
    if args.model:
        model_params["weights"] = args.model
    if args.name:
        exp_params["name"] = args.name
        
    epochs = model_params["epochs"]
    if args.epochs is not None:
        epochs = args.epochs
    elif args.smoke_test:
        epochs = 1
        print("SMOKE TEST MODE ENABLED: Enforcing 1 total training epoch limit.")
        
    if args.epochs is not None:
        print(f"Command Line Override: Enforcing {args.epochs} total training epochs.")

    # Setup reproducibility seeding
    seed = model_params.get("seed", 42)
    seed_everything(seed=seed, deterministic=args.deterministic)
    
    # Calculate task type variant layout (e.g. segmentation/yolo11s)
    task_type = f"{config['task']}/{model_params['variant']}"
    
    # Initialize Experiment Logger
    logger = ExperimentLogger(experiment_name=exp_params["name"], task_type=task_type)
    
    # Log resolved config
    resolved_config = config.copy()
    resolved_config["model"]["epochs"] = epochs
    logger.log_run_start(resolved_config)
    
    # Pre-launch checks: dataset paths existence validation
    data_yaml = exp_params["data_yaml"]
    if not os.path.exists(data_yaml):
        logger.log(f"CRITICAL ERROR: Dataset yaml not found at {data_yaml}")
        sys.exit(1)
        
    try:
        with open(data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
        for split_key in ["train", "val", "test"]:
            if split_key in data_dict:
                split_path = os.path.join(yaml_dir, data_dict[split_key].replace("/", os.sep))
                labels_path = split_path.replace(f"images{os.sep}", f"labels{os.sep}").replace("images", "labels")
                if not os.path.exists(split_path):
                    logger.log(f"CRITICAL ERROR: Dataset images path for {split_key} not found at {split_path}")
                    sys.exit(1)
                if not os.path.exists(labels_path):
                    logger.log(f"CRITICAL ERROR: Dataset labels path for {split_key} not found at {labels_path}")
                    sys.exit(1)
        logger.log("Dataset paths and labels successfully verified.")
    except Exception as e:
        logger.log(f"CRITICAL ERROR: Failed to parse or verify dataset yaml: {e}")
        sys.exit(1)

    # GPU / CPU device routing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.log(f"Routing training execution to device: {device}")
    if torch.cuda.is_available():
        logger.log(f"Current VRAM details: {get_vram_usage()}")

    # Setup automatic resume path
    checkpoint_dir = logger.checkpoint_dir
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
    
    model_path = model_params["weights"]
    resume = False
    
    if args.resume:
        if os.path.exists(last_checkpoint_path):
            model_path = last_checkpoint_path
            resume = True
            logger.log(f"Resuming training from current run checkpoint: {last_checkpoint_path}")
        else:
            parent_checkpoint_dir = os.path.dirname(checkpoint_dir)
            if os.path.exists(parent_checkpoint_dir):
                candidate_checkpoints = []
                for subdir in os.listdir(parent_checkpoint_dir):
                    subpath = os.path.join(parent_checkpoint_dir, subdir)
                    if os.path.isdir(subpath):
                        candidate_last = os.path.join(subpath, "last.pt")
                        if os.path.exists(candidate_last):
                            mtime = os.path.getmtime(candidate_last)
                            candidate_checkpoints.append((candidate_last, mtime, subdir))
                if candidate_checkpoints:
                    candidate_checkpoints.sort(key=lambda x: x[1], reverse=True)
                    latest_last_pt, _, run_id = candidate_checkpoints[0]
                    model_path = latest_last_pt
                    resume = True
                    logger.log(f"Auto-detected and resuming from most recent checkpoint: {latest_last_pt}")
                    logger.run_id = run_id
                    # Re-map directories
                    logger.checkpoint_dir = os.path.join(logger.exp_dir, "checkpoints", logger.run_id)
                    logger.run_dir = os.path.join(logger.exp_dir, "runs", logger.run_id)
                    logger.tb_dir = os.path.join(logger.exp_dir, "tensorboard", logger.run_id)
                    checkpoint_dir = logger.checkpoint_dir
                    last_checkpoint_path = latest_last_pt
            if not resume:
                logger.log("Warning: '--resume' was specified, but no previous 'last.pt' checkpoint was found. Starting a fresh training run.")

    # Load model
    logger.log(f"Initializing YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # OOM Guard Training Loop
    batch_size = model_params["batch"]
    success = False
    project_run_dir = os.path.abspath(os.path.join(exp_params["project"], config["task"], model_params["variant"]))

    while not success:
        try:
            logger.log(f"Attempting to launch YOLO training with batch size: {batch_size}")
            
            # Map configuration values to YOLO trainer
            results = model.train(
                data=exp_params["data_yaml"],
                epochs=epochs,
                patience=model_params["patience"] if not args.smoke_test else 1,
                batch=batch_size,
                imgsz=model_params["imgsz"],
                save=model_params["save"],
                cache=model_params["cache"],
                device=device,
                workers=model_params["workers"],
                seed=seed,
                deterministic=args.deterministic,
                project=project_run_dir,
                name=logger.run_id, # Link YOLO run folder with experiment run ID
                optimizer=opt_params["optimizer"],
                lr0=opt_params["lr0"],
                lrf=opt_params["lrf"],
                cos_lr=opt_params.get("cos_lr", False),
                val=True,
                plots=True,
                amp=True, # Mixed precision FP16
                hsv_h=aug_params["hsv_h"],
                hsv_s=aug_params["hsv_s"],
                hsv_v=aug_params["hsv_v"],
                degrees=aug_params["degrees"],
                translate=aug_params["translate"],
                scale=aug_params["scale"],
                shear=aug_params["shear"],
                perspective=aug_params["perspective"],
                flipud=aug_params["flipud"],
                fliplr=aug_params["fliplr"],
                mosaic=aug_params["mosaic"],
                mixup=aug_params["mixup"],
                copy_paste=aug_params["copy_paste"],
                close_mosaic=model_params.get("close_mosaic", 0) if not args.smoke_test else 0,
                save_period=model_params.get("save_period", 25) if not args.smoke_test else 0,
                resume=resume
            )
            success = True
            
            # Manual copy of weights to standardized checkpoints location
            yolo_run_weights = os.path.join(project_run_dir, logger.run_id, "weights")
            if os.path.exists(yolo_run_weights):
                for f in ["best.pt", "last.pt"]:
                    src = os.path.join(yolo_run_weights, f)
                    dest = os.path.join(checkpoint_dir, f)
                    if os.path.exists(src):
                        shutil.copyfile(src, dest)
                        logger.log(f"Copied YOLO weights '{f}' to checkpoint directory: {dest}")
                
                # Copy milestone weights if present
                for f in os.listdir(yolo_run_weights):
                    if f.startswith("epoch") and f.endswith(".pt"):
                        epoch_num = f.replace("epoch", "").replace(".pt", "")
                        dest_name = f"epoch_{epoch_num}.pt"
                        src = os.path.join(yolo_run_weights, f)
                        dest = os.path.join(checkpoint_dir, dest_name)
                        shutil.copyfile(src, dest)
                        logger.log(f"Copied milestone weight '{f}' to checkpoint directory as '{dest_name}': {dest}")

            # Generate individual learning curve plots
            yolo_run_dir = os.path.join(project_run_dir, logger.run_id)
            results_csv_path = os.path.join(yolo_run_dir, "results.csv")
            generate_learning_curves(results_csv_path, [yolo_run_dir, logger.run_dir])
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.log("Warning: CUDA Out-of-Memory detected during training launch!")
                oom_clean()
                new_batch = decay_batch_size(batch_size, task_type)
                if new_batch is None:
                    logger.log("Critical Error: Cannot recover from VRAM exhaustion. Halting pipeline.")
                    raise e
                batch_size = new_batch
                resume = False
            else:
                logger.log(f"Critical Runtime Exception occurred: {e}")
                raise e

    # ----------------------------------------------------
    # POST-TRAINING AUTOMATED EVALUATION ON BEST CHECKPOINT
    # ----------------------------------------------------
    # Free training memory before starting evaluation
    try:
        del model
    except NameError:
        pass
    oom_clean()

    best_weights_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_weights_path):
        logger.log(f"Warning: best.pt weights not found at {best_weights_path}. Evaluation will be skipped.")
        return

    logger.log(f"\n--- Launching Automated Checkpoint Evaluation on best.pt ---")
    eval_model = YOLO(best_weights_path)
    
    # Run validation
    metrics = eval_model.val(
        data=exp_params["data_yaml"],
        split="val",
        iou=0.60,
        conf=0.25,
        save_json=False,
        plots=True,
        device=device
    )
    
    # Extract Bounding Box metrics
    box_p = float(metrics.box.mp)
    box_r = float(metrics.box.mr)
    box_map50 = float(metrics.box.map50)
    box_map50_95 = float(metrics.box.map)
    
    # Extract Mask Segmentation metrics
    mask_p = 0.0
    mask_r = 0.0
    mask_map50 = 0.0
    mask_map50_95 = 0.0
    if hasattr(metrics, 'seg') and metrics.seg is not None:
        mask_p = float(metrics.seg.mp)
        mask_r = float(metrics.seg.mr)
        mask_map50 = float(metrics.seg.map50)
        mask_map50_95 = float(metrics.seg.map)
    elif hasattr(metrics, 'masks') and metrics.masks is not None:
        mask_p = float(metrics.masks.mp)
        mask_r = float(metrics.masks.mr)
        mask_map50 = float(metrics.masks.map50)
        mask_map50_95 = float(metrics.masks.map)
        
    logger.log(f"Box Metrics: P={box_p:.4f}, R={box_r:.4f}, mAP50={box_map50:.4f}")
    logger.log(f"Mask Metrics: P={mask_p:.4f}, R={mask_r:.4f}, mAP50={mask_map50:.4f}")

    # Extract classes before deleting metrics
    num_classes = len(metrics.names) if hasattr(metrics, 'names') else 10
    try:
        del metrics
    except NameError:
        pass
    oom_clean()

    # 1. Save resolved config snapshot to run directory
    resolved_config_path = os.path.join(logger.run_dir, "config_snapshot.yaml")
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
        
    # 2. Write metrics.json
    metrics_payload = {
        "box": {
            "precision": box_p,
            "recall": box_r,
            "mAP50": box_map50,
            "mAP50-95": box_map50_95
        },
        "mask": {
            "precision": mask_p,
            "recall": mask_r,
            "mAP50": mask_map50,
            "mAP50-95": mask_map50_95
        }
    }
    metrics_json_path = os.path.join(logger.run_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
        
    # 3. Write metadata.json
    metadata_payload = {
        "model": model_params["weights"],
        "task": config["task"],
        "dataset": os.path.basename(exp_params["data_yaml"]),
        "classes": num_classes,
        "epochs": epochs,
        "imgsz": model_params["imgsz"],
        "batch": batch_size,
        "seed": seed,
        "ultralytics_version": ultralytics.__version__,
        "git_commit": logger._get_git_hash()
    }
    metadata_json_path = os.path.join(logger.run_dir, "metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)

    # 4. Generate 10 prediction visual overlays
    try:
        del eval_model
    except NameError:
        pass
    oom_clean()

    overlay_model = YOLO(best_weights_path)

    val_dir = os.path.join(yaml_dir, data_dict["val"].replace("/", os.sep))
    out_overlay_dir = os.path.join(logger.run_dir, "sanity_predictions")
    os.makedirs(out_overlay_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()
    
    with torch.inference_mode():
        for img_name in img_files[:10]:
            img_path = os.path.join(val_dir, img_name)
            res = overlay_model(img_path, verbose=False, device="cpu")
            if len(res) > 0:
                annotated_img = res[0].plot()
                cv2.imwrite(os.path.join(out_overlay_dir, img_name), annotated_img)
                del res
                oom_clean()
            
    logger.log(f"Generated validation predictions inside {out_overlay_dir}")

    # 5. Write experiment_summary.md
    summary_md_path = os.path.join(logger.run_dir, "experiment_summary.md")
    
    # Calculate difference from baseline
    baseline_map50 = 0.694
    difference = box_map50 - baseline_map50
    diff_sign = "+" if difference >= 0 else ""
    
    summary_md = f"""# Experiment Summary: {model_params['weights']}

- **Model**: {model_params['weights']} ({model_params['variant']})
- **Task**: YOLO11 Segmentation
- **Dataset**: {os.path.basename(exp_params['data_yaml'])}
- **Epochs Run**: {epochs}
- **Baseline Comparison (YOLOv8s Detection)**:
  - **Baseline mAP50 (Box)**: {baseline_map50:.2%} ({baseline_map50:.4f})
  - **Current mAP50 (Box)**: {box_map50:.2%} ({box_map50:.4f})
  - **Difference (Box)**: {diff_sign}{difference:.2%}

## Validation Metrics (IoU=0.60, Conf=0.25)

| Metric | Box Bounding Targets | Mask Segmentation Targets |
| :--- | :---: | :---: |
| **Precision** | {box_p:.4f} | {mask_p:.4f} |
| **Recall** | {box_r:.4f} | {mask_r:.4f} |
| **mAP50** | {box_map50:.4f} | {mask_map50:.4f} |
| **mAP50-95** | {box_map50_95:.4f} | {mask_map50_95:.4f} |

---
*Summary generated automatically by train_segmentation.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
        
    logger.log(f"Experiment summary written to {summary_md_path}")
    logger.log_run_end({"status": "success", "metrics": metrics_payload})

if __name__ == "__main__":
    main()
