import gc
import torch

def get_vram_usage():
    """
    Retrieves current CUDA VRAM allocations in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        
    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    reserved = torch.cuda.memory_reserved(0) / (1024**2)
    
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2)
    }

def oom_clean():
    """
    Forces garbage collection and clears PyTorch's VRAM caching allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("OOM Guard: Cleaned CUDA memory cache and forced garbage collection.")

def decay_batch_size(current_batch_size, task_type):
    """
    Decays the batch size according to task-specific constraints.
    - Detection: 16 -> 12 -> 8 -> 4 -> None (unrecoverable)
    - Classification / Severity: 32 -> 24 -> 16 -> 8 -> None (unrecoverable)
    """
    if task_type == "detection":
        decay_map = {16: 12, 12: 8, 8: 4, 4: None}
    elif task_type in ["classification", "severity"]:
        decay_map = {32: 24, 24: 16, 16: 8, 8: None}
    else:
        # Generic fallback
        decay_map = {current_batch_size: max(1, current_batch_size // 2)}
        if current_batch_size <= 1:
            decay_map[current_batch_size] = None
            
    new_batch_size = decay_map.get(current_batch_size, None)
    if new_batch_size is None:
        print(f"OOM Guard: Cannot decay batch size below {current_batch_size}. Training must halt.")
        return None
        
    print(f"OOM Guard: Decaying batch size from {current_batch_size} to {new_batch_size}.")
    return new_batch_size
