import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef, precision_score, recall_score, roc_auc_score

def classification_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int); p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int); k = max(1, math.ceil(0.10 * len(y)))
    top = y[np.argsort(-p, kind="stable")[:k]]; positives = int(y.sum()); prevalence = float(y.mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)), "mcc": float(matthews_corrcoef(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "top_10_percent_k": k, "precision_at_top_10_percent": float(top.mean()),
        "recall_at_top_10_percent": float(top.sum() / positives) if positives else 0.0,
        "lift_at_top_10_percent": float(top.mean() / prevalence) if prevalence else 0.0,
    }
