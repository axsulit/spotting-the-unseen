import torch
import numpy as np
from networks.freqnet import freqnet
from sklearn.metrics import average_precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from options.test_options import TestOptions
from data import create_dataloader

def validate(model, opt):
    """
    Run validation on a dataset using a given model and options.

    Evaluates binary classification performance by computing accuracy,
    average precision, recall, F1 score, and confusion matrix statistics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        opt (argparse.Namespace): Parsed test options including dataroot.

    Returns:
        Tuple containing:
            - acc (float): Overall accuracy.
            - ap (float): Average precision.
            - r_acc (float or None): Accuracy on real (label 0) samples.
            - f_acc (float or None): Accuracy on fake (label 1) samples.
            - recall (float): Recall score.
            - f1 (float): F1 score.
            - tp (int): True positives.
            - tn (int): True negatives.
            - fp (int): False positives.
            - fn (int): False negatives.
            - y_true (np.ndarray): Ground truth labels.
            - y_pred (np.ndarray): Raw sigmoid predictions.
    """
    
    data_loader = create_dataloader(opt)

    # Get the device where model is located
    device = next(model.parameters()).device

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            img = img.to(device)
            label = label.to(device)

            outputs = model(img).sigmoid().flatten()
            y_pred.extend(outputs.tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred_bin = (y_pred > 0.5).astype(int)

    # Compute accuracy metrics
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) if np.any(y_true == 0) else None
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) if np.any(y_true == 1) else None
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average='binary', zero_division=0
    )

    return acc, ap, r_acc, f_acc, recall, f1, tp, tn, fp, fn, y_true, y_pred
