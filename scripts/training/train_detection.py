import os
import sys
import argparse
from datetime import datetime

# Add root folder to import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_dependencies():
    """Verifies that all required packages are installed."""
    required = ["torch", "pandas", "numpy", "yaml", "ultralytics"]
    print("Checking training pipeline dependencies...")
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
from ultralytics import YOLO
from scripts.utils.seed_everything import seed_everything
from scripts.utils.experiment_logger import ExperimentLogger
from scripts.training.utils.config_loader import load_config
from scripts.training.utils.oom_guard import oom_clean, decay_batch_size, get_vram_usage

def main():
    parser = argparse.ArgumentParser(description="QYRO Acne YOLOv8n Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/detection_config.yaml", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Execute a 1-epoch sanity smoke test")
    parser.add_argument("--deterministic", type=bool, default=True, help="Toggle strict determinism vs performance benchmark")
    parser.add_argument("--resume", action="store_true", help="Automatically resume training from the latest checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_params = config["model_params"]
    opt_params = config["optimizer_params"]
    aug_params = config["augmentation_params"]
    log_params = config["logging_params"]
    
    # Override for 1-epoch smoke test
    epochs = 1 if args.smoke_test else model_params["epochs"]
    if args.smoke_test:
        print("SMOKE TEST MODE ENABLED: Enforcing 1 total training epoch limit.")
        
    # Setup reproducibility seeding
    seed = model_params.get("seed", 42)
    seed_everything(seed=seed, deterministic=args.deterministic)
    
    # Initialize Experiment Logger
    logger = ExperimentLogger(experiment_name=log_params["name"], task_type="detection")
    logger.log_run_start(config)
    
    # Pre-launch checks: dataset paths existence validation
    data_yaml = model_params["data_yaml"]
    if not os.path.exists(data_yaml):
        logger.log(f"CRITICAL ERROR: Dataset yaml not found at {data_yaml}")
        sys.exit(1)
        
    try:
        import yaml
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
    
    model_path = model_params["model_type"]
    resume = False
    
    if args.resume:
        if os.path.exists(last_checkpoint_path):
            model_path = last_checkpoint_path
            resume = True
            logger.log(f"Resuming detection training from current run checkpoint: {last_checkpoint_path}")
        else:
            # Search for the latest checkpoint directory under check_point root
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
                    # Re-map dirs
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
    
    while not success:
        try:
            logger.log(f"Attempting to launch YOLOv8n training with batch size: {batch_size}")
            
            # Map configuration values to YOLOv8 trainer
            results = model.train(
                data=model_params["data_yaml"],
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
                project=os.path.abspath(log_params["project"]),
                name=logger.run_id, # Link YOLO run folder with experiment run ID
                optimizer=opt_params["optimizer"],
                lr0=opt_params["lr0"],
                lrf=opt_params["lrf"],
                cos_lr=opt_params.get("cos_lr", False),
                val=True,
                plots=log_params["plots"],
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
                resume=resume
            )
            success = True
            
            # Manual copy of weights to standardized checkpoints location
            yolo_run_weights = os.path.join(os.path.abspath(log_params["project"]), logger.run_id, "weights")
            if os.path.exists(yolo_run_weights):
                for f in ["best.pt", "last.pt"]:
                    src = os.path.join(yolo_run_weights, f)
                    dest = os.path.join(checkpoint_dir, f)
                    if os.path.exists(src):
                        import shutil
                        shutil.copyfile(src, dest)
                        logger.log(f"Copied YOLO weight '{f}' to checkpoint directory: {dest}")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.log("Warning: CUDA Out-of-Memory detected during training launch!")
                oom_clean()
                # Decay batch size
                new_batch = decay_batch_size(batch_size, "detection")
                if new_batch is None:
                    logger.log("Critical Error: Cannot recover from VRAM exhaustion. Halting pipeline.")
                    raise e
                batch_size = new_batch
                # Disable resume flag if we have to re-allocate
                resume = False
            else:
                logger.log(f"Critical Runtime Exception occurred: {e}")
                raise e
                
    # Log metrics to final metrics file
    final_metrics = {"status": "success", "epochs_run": epochs}
    logger.log_run_end(final_metrics)

if __name__ == "__main__":
    main()
