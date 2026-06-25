import os
import shutil
import torch

def save_checkpoint(state, is_best, checkpoint_dir, epoch=None, save_periodic=False, period=10):
    """
    Saves the training state to checkpoints directory.
    Saves last.pt, best.pt (if is_best), and optionally periodic epoch checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save last checkpoint
    last_path = os.path.join(checkpoint_dir, "last.pt")
    torch.save(state, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        shutil.copyfile(last_path, best_path)
        
    # Save periodic checkpoint
    if save_periodic and epoch is not None and epoch % period == 0:
        periodic_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        shutil.copyfile(last_path, periodic_path)
        print(f"Saved periodic checkpoint at epoch {epoch}: {periodic_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads checkpoint state dictionary into model, optimizer, and scheduler.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
        
    epoch = state.get('epoch', 0)
    best_metric = state.get('best_metric', None)
    
    print(f"Loaded checkpoint at epoch {epoch} (best metric: {best_metric})")
    return epoch, best_metric
