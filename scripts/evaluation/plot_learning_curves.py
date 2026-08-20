import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(results_csv_path):
    if not os.path.exists(results_csv_path):
        print(f"[ERROR] results.csv not found at: {results_csv_path}")
        return False

    # Read CSV
    df = pd.read_csv(results_csv_path)
    # Clean whitespace from column headers
    df.columns = [c.strip() for c in df.columns]
    
    output_dir = os.path.dirname(results_csv_path)
    print(f"Loading results from {results_csv_path}")
    print(f"Saving plots inside {output_dir}")
    
    epochs = df['epoch']
    
    # 1. Training Loss Curves
    plt.figure(figsize=(10, 6))
    for col in ['train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss']:
        if col in df.columns:
            plt.plot(epochs, df[col], label=col.split('/')[-1], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    train_loss_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(train_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training loss plot: {train_loss_path}")
    
    # 2. Validation Loss Curves (Filter out inf values)
    plt.figure(figsize=(10, 6))
    for col in ['val/box_loss', 'val/seg_loss', 'val/cls_loss', 'val/dfl_loss']:
        if col in df.columns:
            # Replace 'inf' string or np.inf with NaN and plot
            series = pd.to_numeric(df[col], errors='coerce')
            # Filter epochs where series is not null/inf to plot cleanly
            valid_idx = series.notnull() & np.isfinite(series)
            if valid_idx.any():
                plt.plot(epochs[valid_idx], series[valid_idx], label=col.split('/')[-1], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    val_loss_path = os.path.join(output_dir, 'validation_loss.png')
    plt.savefig(val_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved validation loss plot: {val_loss_path}")
    
    # 3. Box mAP50 Curve
    plt.figure(figsize=(10, 6))
    box_map_col = 'metrics/mAP50(B)'
    if box_map_col in df.columns:
        plt.plot(epochs, df[box_map_col], label='Box mAP50', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.title('Bounding Box mAP50')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    box_map_path = os.path.join(output_dir, 'box_map50.png')
    plt.savefig(box_map_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Box mAP50 plot: {box_map_path}")
    
    # 4. Mask mAP50 Curve
    plt.figure(figsize=(10, 6))
    mask_map_col = 'metrics/mAP50(M)'
    if mask_map_col in df.columns:
        plt.plot(epochs, df[mask_map_col], label='Mask mAP50', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.title('Mask Segmentation mAP50')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    mask_map_path = os.path.join(output_dir, 'mask_map50.png')
    plt.savefig(mask_map_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Mask mAP50 plot: {mask_map_path}")
    
    # 5. Learning Rate Curves
    plt.figure(figsize=(10, 6))
    for col in ['lr/pg0', 'lr/pg1', 'lr/pg2']:
        if col in df.columns:
            plt.plot(epochs, df[col], label=col, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate progression')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    lr_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(lr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved learning rate plot: {lr_path}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate learning curves from results.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    args = parser.parse_args()
    plot_curves(args.csv)
