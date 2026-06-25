import os
import json
import yaml
import time
import subprocess
from datetime import datetime
from scripts.utils.system_info import get_system_specs

class ExperimentLogger:
    """
    Standardized logger for tracking QYRO model training metadata,
    hardware specs, git hashes, config parameters, and metric histories.
    """
    def __init__(self, experiment_name, task_type, base_dir="experiments"):
        """
        Args:
            experiment_name: Name of the active run (e.g. 'yolov8n_baseline')
            task_type: 'detection', 'subtype', or 'severity'
            base_dir: Root directory for experiments
        """
        self.experiment_name = experiment_name
        self.task_type = task_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        
        # Setup standard folder structures
        self.exp_dir = os.path.join(base_dir, task_type)
        self.run_dir = os.path.join(self.exp_dir, "runs", self.run_id)
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints", self.run_id)
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.tb_dir = os.path.join(self.exp_dir, "tensorboard", self.run_id)
        
        for d in [self.run_dir, self.checkpoint_dir, self.log_dir, self.tb_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.log_file = os.path.join(self.log_dir, f"{self.run_id}.log")
        self.metrics_file = os.path.join(self.run_dir, "metrics.json")
        self.log_csv = os.path.join(self.run_dir, "training_log.csv")
        
    def _get_git_hash(self):
        """Attempts to retrieve the current git hash."""
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return "unknown_or_no_git"
            
    def _get_dataset_hash(self, workspace_root=r"c:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI"):
        """Gets the SHA256 checksum of the master registry CSV in the final dataset."""
        registry_path = os.path.join(workspace_root, "datasets", "skin", "acne", "final", "registry", "master_acne_registry.csv")
        if not os.path.exists(registry_path):
            return "unknown_dataset_no_registry"
        try:
            import hashlib
            sha = hashlib.sha256()
            with open(registry_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception:
            return "error_calculating_hash"

    def log_run_start(self, config_dict):
        """Initializes the experiment log with config snapshots and hardware info."""
        # Probes hardware specifications
        hw_specs = get_system_specs()
        git_hash = self._get_git_hash()
        dataset_hash = self._get_dataset_hash()
        
        # Build comprehensive metadata dictionary
        run_metadata = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "git_hash": git_hash,
            "dataset_hash": dataset_hash,
            "hardware_profile": hw_specs,
            "config": config_dict
        }
        
        # Save snapshot
        snapshot_yaml = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(snapshot_yaml, "w") as f:
            yaml.dump(run_metadata, f, default_flow_style=False, sort_keys=False)
            
        self.log(f"Experiment Run Started: {self.run_id}")
        self.log(f"Git Hash: {git_hash} | Dataset Hash: {dataset_hash}")
        self.log(f"Training GPU: {hw_specs.get('gpu_model')} | VRAM: {hw_specs.get('gpu_vram_gb')} GB")
        self.log(f"Config Snapshot saved to {snapshot_yaml}")
        
    def log(self, message):
        """Appends a timestamped string message to the run logs."""
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{t_str}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_line)
        print(message)
        
    def log_epoch_metrics(self, epoch, epoch_metrics):
        """
        Logs metrics at the end of each epoch to a CSV log.
        epoch_metrics should be a dict: {'loss': 0.23, 'val_loss': 0.28, ...}
        """
        import csv
        header = ["epoch", "timestamp"] + list(epoch_metrics.keys())
        row = [epoch, time.time()] + list(epoch_metrics.values())
        
        file_exists = os.path.exists(self.log_csv)
        with open(self.log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
            
        self.log(f"Epoch {epoch} Metrics: " + ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in epoch_metrics.items()]))
        
    def log_run_end(self, final_metrics):
        """Saves final metrics summary to JSON."""
        with open(self.metrics_file, "w") as f:
            json.dump(final_metrics, f, indent=4)
        self.log(f"Experiment completed. Final Metrics saved to {self.metrics_file}")

if __name__ == "__main__":
    # Test script run
    logger = ExperimentLogger(experiment_name="test_logger", task_type="detection")
    test_config = {"lr": 0.01, "batch_size": 16, "epochs": 10}
    logger.log_run_start(test_config)
    logger.log_epoch_metrics(1, {"loss": 0.5, "val_loss": 0.6})
    logger.log_run_end({"best_loss": 0.5})
