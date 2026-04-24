from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from utils import PAPER_CLASS_ORDER


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def class_wise_detection(
    multiclass_labels: pd.Series,
    y_pred_binary: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = pd.Series(multiclass_labels).fillna("UNKNOWN")
    for cls, idx in labels.groupby(labels).groups.items():
        mask = labels.index.isin(idx)
        support = int(mask.sum())
        if cls == "BENIGN":
            rate = float((1 - y_pred_binary[mask]).mean()) if support else 0.0
            metric = "Correct Benign Rate"
        else:
            rate = float(y_pred_binary[mask].mean()) if support else 0.0
            metric = "Detection Rate"
        rows.append({"Class": cls, "Support": support, "Metric": metric, "Rate": rate})
    out = pd.DataFrame(rows)
    order_map = {name: idx for idx, name in enumerate(PAPER_CLASS_ORDER)}
    out["_order"] = out["Class"].map(lambda x: order_map.get(x, 9999))
    out = out.sort_values(["_order", "Class"]).drop(columns=["_order"]).reset_index(drop=True)
    return out
