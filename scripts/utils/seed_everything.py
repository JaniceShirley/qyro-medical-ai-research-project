import os
import random
import numpy as np
import torch

def seed_everything(seed=42, deterministic=True):
    """
    Sets seeds for reproducibility across random, numpy, and torch.
    Supports a configurable choice between strict determinism or max performance.
    """
    print(f"Setting global seed: {seed}")
    
    # 1. Standard python & numpy seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch CPU & GPU seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 3. Determinism vs Performance Mode
    if deterministic:
        print("Determinism mode: ENABLED. Strict reproducibility (slower execution).")
        # Ensure CuDNN always returns the same algorithm
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Configure CUBLAS workspace for deterministic operations (avoids runtime warning in torch)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"Warning: strict torch determinism setup error: {e}")
            print("Falling back to soft determinism (deterministic CuDNN only).")
            torch.use_deterministic_algorithms(False)
    else:
        print("Performance mode: ENABLED. Optimizing for CUDA execution speed.")
        # Allow CuDNN to dynamically search for the fastest convolution algorithm
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

if __name__ == "__main__":
    # Test script run
    seed_everything(seed=42, deterministic=True)
