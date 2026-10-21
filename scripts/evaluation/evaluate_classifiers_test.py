"""
Evaluate a trained subtype or severity checkpoint on the frozen TEST split.

The training scripts only report validation metrics (the split used to pick best.pt),
so this script gives the unbiased number to quote.

Usage:
    python scripts/evaluation/evaluate_classifiers_test.py --task subtype
    python scripts/evaluation/evaluate_classifiers_test.py --task severity --checkpoint path/to/best.pt
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd
import timm

from scripts.training.utils.config_loader import load_config
from scripts.training.utils.metrics import (
    calculate_classification_metrics,
    calculate_ordinal_metrics,
    convert_ordinal_logits_to_labels,
)


def latest_best_checkpoint(task):
    candidates = glob.glob(os.path.join("experiments", task, "checkpoints", "*", "best.pt"))
    if not candidates:
        sys.exit(f"No best.pt found under experiments/{task}/checkpoints/. Train the model first.")
    return max(candidates, key=os.path.getmtime)


def per_class_report(y_true, y_pred, class_names):
    rows = {}
    for c, name in enumerate(class_names):
        support = int(np.sum(y_true == c))
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows[name] = {"support": support, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate QYRO subtype/severity classifiers on the test split")
    parser.add_argument("--task", choices=["subtype", "severity"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Defaults to the newest best.pt for the task")
    args = parser.parse_args()

    if args.task == "subtype":
        from scripts.training.train_subtype import SUBTYPE_MAP as LABEL_MAP, get_dataloaders
        num_outputs = len(LABEL_MAP)
    else:
        from scripts.training.train_severity import SEVERITY_MAP as LABEL_MAP, get_dataloaders
        num_outputs = len(LABEL_MAP) - 1  # ordinal head has K-1 nodes

    config = load_config(f"configs/{args.task}_config.yaml")
    ds_params = config["dataset_params"]
    checkpoint_path = args.checkpoint or latest_best_checkpoint(args.task)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    df_manifest = pd.read_csv(ds_params["manifest_path"])
    _, _, test_loader = get_dataloaders(df_manifest, ds_params["batch_size"], 0, False, None)

    model = timm.create_model(config["model_params"]["backbone"], pretrained=False, num_classes=num_outputs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    outputs_all, targets_all = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            outputs_all.append(model(images.to(device)).cpu())
            targets_all.append(targets)
    outputs = torch.cat(outputs_all)
    y_true = torch.cat(targets_all).numpy().astype(int)

    class_names = sorted(LABEL_MAP, key=LABEL_MAP.get)
    if args.task == "subtype":
        y_pred = torch.argmax(outputs, dim=1).numpy().astype(int)
        metrics = calculate_classification_metrics(y_true, y_pred, num_classes=len(class_names))
        per_class = per_class_report(y_true, y_pred, class_names)
        present = [name for name in class_names if per_class[name]["support"] > 0]
        metrics["macro_f1_present_classes"] = float(np.mean([per_class[n]["f1"] for n in present]))
        metrics["present_classes"] = present
    else:
        y_pred = convert_ordinal_logits_to_labels(torch.sigmoid(outputs).numpy())
        metrics = calculate_ordinal_metrics(y_true, y_pred, num_classes=len(class_names))
        per_class = per_class_report(y_true, y_pred, class_names)
        metrics["within_one_stage"] = float(np.mean(np.abs(y_true - y_pred) <= 1))
        metrics["confusion_matrix"] = calculate_classification_metrics(y_true, y_pred, len(class_names))["confusion_matrix"]

    metrics["confusion_matrix"] = np.asarray(metrics["confusion_matrix"]).tolist()
    result = {
        "task": args.task,
        "split": "test",
        "checkpoint": checkpoint_path,
        "num_test_images": int(len(y_true)),
        "class_names": class_names,
        "metrics": metrics,
        "per_class": per_class,
    }

    out_path = os.path.join(os.path.dirname(checkpoint_path), "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved test metrics to {out_path}")


if __name__ == "__main__":
    main()
