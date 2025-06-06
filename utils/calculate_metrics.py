"""
Calculate classification metrics such as accuracy, precision, recall, and F1 score.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (np.array(y_pred) > threshold).astype(int)
    y_true = np.array(y_true)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "f1_score": f1_score(y_true, y_pred_binary),
    }

    return metrics
