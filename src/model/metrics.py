"""Evaluation metrics required by the project.

FAR (False Acceptance Ratio) = class-0 samples predicted as 1 / number of class-0 samples
FRR (False Rejection Ratio)  = class-1 samples predicted as 0 / number of class-1 samples
"""
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg, pos = y_true == 0, y_true == 1
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "FAR": float((y_pred[neg] == 1).mean()) if neg.any() else float("nan"),
        "FRR": float((y_pred[pos] == 0).mean()) if pos.any() else float("nan"),
        "accuracy": float((y_true == y_pred).mean()),
    }
