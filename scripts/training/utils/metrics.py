import numpy as np

def cohen_kappa_score_quadratic_numpy(y_true, y_pred, num_classes=4):
    """
    Computes Quadratic Weighted Kappa (QWK) using pure NumPy.
    Works independently of scikit-learn.
    """
    # 1. Compute confusion matrix O
    O = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            O[int(t), int(p)] += 1
            
    # 2. Compute weights matrix W
    W = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
            
    # 3. Compute expected matrix E
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    total = np.sum(O)
    if total == 0:
        return 0.0
        
    E = np.outer(hist_true, hist_pred) / total
    
    # 4. Compute QWK
    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0:
        return 1.0 # Perfect agreement if no variances exist
    return float(1.0 - (num / den))

def calculate_classification_metrics(y_true, y_pred, num_classes):
    """
    Calculates Macro F1, Recall, Precision, and Overall Accuracy using NumPy.
    Provides robust fallback if scikit-learn is not present.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Overall Accuracy
    accuracy = float(np.sum(y_true == y_pred) / len(y_true)) if len(y_true) > 0 else 0.0
    
    # Calculate per-class Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    
    # Build Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
            
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }

def calculate_ordinal_metrics(y_true, y_pred, num_classes=4):
    """
    Calculates metrics specific to ordinal regression.
    Includes Mean Absolute Error (MAE) and Quadratic Weighted Kappa.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0
    qwk = cohen_kappa_score_quadratic_numpy(y_true, y_pred, num_classes)
    
    # Use standard classification metrics helper for macro F1
    clf_metrics = calculate_classification_metrics(y_true, y_pred, num_classes)
    
    metrics = {
        "mae": mae,
        "qwk": qwk,
        "accuracy": clf_metrics["accuracy"],
        "macro_f1": clf_metrics["macro_f1"]
    }
    return metrics

def convert_ordinal_logits_to_labels(sigmoid_outputs, threshold=0.5):
    """
    Converts sigmoid ordinal task predictions to integer labels.
    Inputs: sigmoid_outputs: shape (batch_size, num_classes - 1)
    Output: integer class labels (0 to num_classes - 1)
    """
    # For each sample, we sum up the indicators: class = sum(pred > threshold)
    # E.g., if pred = [0.9, 0.8, 0.1] -> sum = 2 -> class 2 (Stage 3, 0-indexed)
    # Output classes will be 0, 1, 2, 3
    preds = (sigmoid_outputs > threshold).astype(int)
    return np.sum(preds, axis=1)
