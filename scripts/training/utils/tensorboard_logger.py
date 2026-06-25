import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

class TensorBoardLogger:
    """
    Structured Tensorboard Logger that organizes metrics into clean sub-groups
    (e.g., train/loss, val/accuracy) and logs confusion matrix visualisations.
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard SummaryWriter initialized at: {log_dir}")

    def log_scalar(self, tag, value, step):
        """Logs scalar metric (e.g. tag='train/loss')"""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Logs multiple scalars under a shared group tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_confusion_matrix(self, cm, class_names, step, tag="val/confusion_matrix"):
        """
        Plots confusion matrix and logs it as an image to TensorBoard.
        cm: numpy array representing confusion matrix
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title="Confusion Matrix",
               ylabel="True label",
               xlabel="Predicted label")
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        
        # Log figure to TensorBoard
        self.writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    def close(self):
        self.writer.close()
