import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image

# Add root folder to import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_dependencies():
    """Verifies that all required packages are installed."""
    required = ["torch", "torchvision", "pandas", "numpy", "yaml", "timm", "tensorboard", "matplotlib"]
    print("Checking severity grading dependencies...")
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from scripts.utils.seed_everything import seed_everything
from scripts.utils.experiment_logger import ExperimentLogger
from scripts.training.utils.config_loader import load_config
from scripts.training.utils.checkpoint_manager import save_checkpoint, load_checkpoint
from scripts.training.utils.early_stopping import EarlyStopping
from scripts.training.utils.tensorboard_logger import TensorBoardLogger
from scripts.training.utils.metrics import calculate_ordinal_metrics, convert_ordinal_logits_to_labels
from scripts.training.utils.oom_guard import oom_clean, decay_batch_size, get_vram_usage

# Fixed mapping of severity stages
SEVERITY_MAP = {
    "stage_1": 0,
    "stage_2": 1,
    "stage_3": 2,
    "stage_4": 3
}

class AcneSeverityDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.workspace_root = r"c:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI"
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        abs_img_path = os.path.join(self.workspace_root, row['image_path'].replace("/", os.sep))
        image = Image.open(abs_img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = SEVERITY_MAP[row['severity_label']]
        return image, label

def get_dataloaders(df_manifest, batch_size, num_workers, pin_memory, prefetch_factor):
    """Splits dataset into train/valid/test loaders."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df_train = df_manifest[df_manifest['split'] == 'train'].reset_index(drop=True)
    df_val = df_manifest[df_manifest['split'] == 'valid'].reset_index(drop=True)
    df_test = df_manifest[df_manifest['split'] == 'test'].reset_index(drop=True)
    
    train_ds = AcneSeverityDataset(df_train, train_transform)
    val_ds = AcneSeverityDataset(df_val, val_transform)
    test_ds = AcneSeverityDataset(df_test, val_transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader

def ordinal_loss(logits, targets):
    """
    Computes Ordinal Regression loss using BCEWithLogitsLoss.
    logits: shape (batch, K-1)
    targets: shape (batch) containing labels 0, 1, 2, 3
    """
    device = targets.device
    num_classes = 4
    # Create ordinal targets: t_i = 1 if target > i else 0
    # E.g. target = 2 -> threshold [0, 1, 2] -> target > threshold -> [True, True, False] -> [1.0, 1.0, 0.0]
    thresholds = torch.tensor([0, 1, 2], dtype=torch.float32).to(device)
    ordinal_targets = (targets.unsqueeze(1) > thresholds).float()
    
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, ordinal_targets)

def main():
    parser = argparse.ArgumentParser(description="QYRO Acne EfficientNet-B0 Severity Grading Pipeline")
    parser.add_argument("--config", type=str, default="configs/severity_config.yaml", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Execute a 1-epoch sanity smoke test")
    parser.add_argument("--deterministic", type=bool, default=True, help="Toggle strict determinism vs performance benchmark")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_params = config["model_params"]
    ds_params = config["dataset_params"]
    train_params = config["training_params"]
    log_params = config["logging_params"]
    
    # 1-epoch override
    epochs = 1 if args.smoke_test else train_params["epochs"]
    if args.smoke_test:
        print("SMOKE TEST MODE ENABLED: Enforcing 1 total training epoch limit.")
        
    # Seeding reproducibility
    seed = model_params.get("seed", 42)
    seed_everything(seed=seed, deterministic=args.deterministic)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name=log_params["name"], task_type="severity")
    logger.log_run_start(config)
    tb_logger = TensorBoardLogger(logger.tb_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(f"Routing training execution to device: {device}")
    
    # Load manifest
    df_manifest = pd.read_csv(ds_params["manifest_path"])
    logger.log(f"Loaded severity manifest with {len(df_manifest)} entries.")
    
    # Launch training loop with OOM safety guard
    batch_size = ds_params["batch_size"]
    success = False
    
    while not success:
        try:
            logger.log(f"Attempting dataloader allocation with batch size: {batch_size}")
            train_loader, val_loader, test_loader = get_dataloaders(
                df_manifest, batch_size, ds_params["num_workers"],
                ds_params["pin_memory"], ds_params["prefetch_factor"]
            )
            
            # Setup Model: EfficientNet-B0 with custom output head (size K-1 = 3)
            logger.log(f"Initializing timm model backbone: {model_params['backbone']}")
            model = timm.create_model(
                model_params['backbone'], pretrained=model_params['pretrained'],
                num_classes=3 # K-1 classification nodes for ordinal logits
            )
            model = model.to(device)
            
            # Phase 1: Train head only
            logger.log("Phase 1: Training classification head only...")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_classifier().parameters():
                param.requires_grad = True
                
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=train_params["head_lr"], weight_decay=train_params["weight_decay"]
            )
            
            scaler = torch.cuda.amp.GradScaler() # AMP FP16 scaler
            early_stopping = EarlyStopping(patience=train_params["early_stopping_patience"], mode="min")
            
            checkpoint_dir = logger.checkpoint_dir
            last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
            
            start_epoch = 0
            best_val_loss = float("inf")
            
            # Automatic Resume check
            if os.path.exists(last_checkpoint_path):
                start_epoch, loaded_loss = load_checkpoint(last_checkpoint_path, model, optimizer, device=device)
                if loaded_loss is not None:
                    best_val_loss = loaded_loss
                logger.log(f"Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
                
            # Training loop
            for epoch in range(start_epoch, epochs):
                # Phase transition: check if we should unfreeze backbone
                if not args.smoke_test and epoch == train_params["head_only_epochs"]:
                    logger.log("Phase 2: Unfreezing backbone for full fine-tuning...")
                    for param in model.parameters():
                        param.requires_grad = True
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=train_params["fine_tune_lr"],
                        weight_decay=train_params["weight_decay"]
                    )
                
                # --- Train Epoch ---
                model.train()
                train_loss = 0.0
                for step, (images, targets) in enumerate(train_loader):
                    images, targets = images.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = ordinal_loss(outputs, targets)
                        
                    # NaN Loss Protection
                    if torch.isnan(loss):
                        logger.log("CRITICAL ERROR: Training Loss became NaN! Halting pipeline execution immediately.")
                        raise ValueError("NaN loss detected.")
                        
                    scaler.scale(loss).backward()
                    
                    # Gradient Clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item() * images.size(0)
                    
                train_loss /= len(train_loader.dataset)
                tb_logger.log_scalar("train/loss", train_loss, epoch)
                
                # --- Validation Epoch ---
                model.eval()
                val_loss = 0.0
                all_probs = []
                all_targets = []
                
                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = images.to(device), targets.to(device)
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = ordinal_loss(outputs, targets)
                            
                        val_loss += loss.item() * images.size(0)
                        
                        # Apply sigmoid to convert to probabilities
                        probs = torch.sigmoid(outputs)
                        all_probs.extend(probs.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        
                val_loss /= len(val_loader.dataset)
                
                # Convert ordinal predictions to class labels (0 to 3)
                all_probs = np.array(all_probs)
                all_preds = convert_ordinal_logits_to_labels(all_probs)
                
                metrics = calculate_ordinal_metrics(all_targets, all_preds, num_classes=4)
                
                # Log to TensorBoard
                tb_logger.log_scalar("val/loss", val_loss, epoch)
                tb_logger.log_scalar("val/accuracy", metrics["accuracy"], epoch)
                tb_logger.log_scalar("val/macro_f1", metrics["macro_f1"], epoch)
                tb_logger.log_scalar("val/mae", metrics["mae"], epoch)
                tb_logger.log_scalar("val/qwk", metrics["qwk"], epoch)
                
                # VRAM Monitoring
                if torch.cuda.is_available():
                    vram_usage = get_vram_usage()
                    tb_logger.log_scalar("system/vram_allocated_mb", vram_usage["allocated_mb"], epoch)
                    tb_logger.log_scalar("system/vram_reserved_mb", vram_usage["reserved_mb"], epoch)
                    
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    
                # Save checkpoints
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_loss
                }
                save_checkpoint(checkpoint_state, is_best, checkpoint_dir, epoch=epoch, save_periodic=True, period=10)
                
                logger.log_epoch_metrics(epoch, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "val_mae": metrics["mae"],
                    "val_qwk": metrics["qwk"]
                })
                
                # Check Early Stopping
                if not args.smoke_test and early_stopping(val_loss):
                    logger.log("Early stopping triggered.")
                    break
                    
            success = True
            logger.log("Severity Grading ordinal training pipeline runs successfully completed!")
            tb_logger.close()
            
            final_metrics = {"status": "success", "epochs_run": epochs, "best_val_loss": best_val_loss}
            logger.log_run_end(final_metrics)
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.log("Warning: CUDA Out-of-Memory detected during Severity training launch!")
                oom_clean()
                new_batch = decay_batch_size(batch_size, "severity")
                if new_batch is None:
                    logger.log("Critical Error: Cannot recover from VRAM exhaustion. Halting pipeline.")
                    raise e
                batch_size = new_batch
            else:
                logger.log(f"Critical Runtime Exception occurred: {e}")
                raise e

if __name__ == "__main__":
    main()
