#!/usr/bin/env python3
"""
compare_anomaly_baselines_valcal.py
===================================

Validation-calibrated comparison of common anomaly-detection baselines against
the Hybrid-Cascade on the CIC-IDS2017 split used by the main project.

Key change vs. the earlier script
---------------------------------
Thresholds are selected on the validation split and then frozen for the
held-out test split:

  1. val_f1_calibrated
     Threshold maximises F1 on D_val, then is applied unchanged to D_test.

  2. val_far_calibrated
     Threshold is chosen on D_val so benign FPR ~= target FAR, then is applied
     unchanged to D_test.

Optional benchmark-only rows can still be emitted:
  3. test_f1_optimal      (optimistic upper bound)
  4. test_far_matched     (benchmark-only matched-FAR on test)

Outputs
-------
  <out_dir>/overall_metrics_baselines.csv
  <out_dir>/baselines_table_fragment.tex
  <out_dir>/run_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM

from config import RFConfig
from features import build_tabular_preprocessor, select_feature_columns
from load_data import load_and_prepare_detection_data
from lstm_autoencoder_baseline import lstm_autoencoder_scores


DEFAULT_EXCLUDE_COLUMNS: tuple[str, ...] = RFConfig.exclude_columns


def log(msg: str) -> None:
    print(f"[baselines-valcal] {msg}", flush=True)


# -------------------------------------------------------------------------
# metric helpers
# -------------------------------------------------------------------------

def _far_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    return float(fp) / float(denom) if denom else 0.0


def _metric_row(
    method: str,
    operating_point: str,
    threshold: float,
    threshold_source: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    fit_seconds: float,
    score_seconds: float,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr_auc = float("nan")

    return {
        "method": method,
        "operating_point": operating_point,
        "threshold_source": threshold_source,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": float(_far_from_confusion(y_true, y_pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fit_seconds": float(fit_seconds),
        "score_seconds": float(score_seconds),
    }


def _threshold_for_target_far(
    y_true: np.ndarray, y_score: np.ndarray, target_far: float
) -> float:
    """Smallest threshold t such that FPR(y_score >= t) <= target_far."""
    benign_scores = y_score[y_true == 0]
    if benign_scores.size == 0:
        return float("inf")
    n_benign = benign_scores.size
    allowed_fp = int(np.floor(n_benign * target_far))
    sorted_desc = np.sort(benign_scores)[::-1]
    if allowed_fp >= n_benign:
        return float(sorted_desc[-1]) - 1.0
    return float(sorted_desc[allowed_fp]) + 1e-12


def _threshold_f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    f1s = f1s[:-1]
    if f1s.size == 0 or thresholds.size == 0:
        return float(np.median(y_score))
    best_idx = int(np.nanargmax(f1s))
    return float(thresholds[best_idx])


def _threshold_accuracy_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return a threshold t that maximises accuracy of (y_score >= t).

    Exact O(n log n) search. Only rank cuts between DISTINCT scores (plus
    the two endpoints) are candidates, because a threshold cannot split
    two rows that share the same score. This matters when the score has
    large ties -- e.g., a cascade where many rows get score = 0 because
    they weren't escalated. Tie-break among equally-accurate achievable
    thresholds prefers the tighter one (lower FAR).
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    n = y_score_np.size
    if n == 0:
        return 0.0
    # Sort by score descending -> rank k means "top-k rows predicted positive".
    order = np.argsort(-y_score_np, kind="stable")
    y_sorted = y_true_np[order]
    s_sorted = y_score_np[order]
    n_pos = int(y_sorted.sum())
    n_neg = n - n_pos
    # cum_tp[k] = # positives among top-k (ranks 0..k-1 predicted positive).
    cum_tp = np.concatenate(([0], np.cumsum(y_sorted)))
    ks = np.arange(n + 1, dtype=np.float64)
    # Accuracy at rank-k cut: (TP + TN) / n = (2*cum_tp + n_neg - k) / n.
    acc = (2.0 * cum_tp + n_neg - ks) / float(n)
    # Only keep cuts that correspond to an ACHIEVABLE threshold: either the
    # two endpoints, or boundaries between two distinct scores.
    valid = np.zeros(n + 1, dtype=bool)
    valid[0] = True
    valid[n] = True
    if n > 1:
        valid[1:n] = s_sorted[:-1] > s_sorted[1:]
    acc_valid = np.where(valid, acc, -np.inf)
    best_acc = float(acc_valid.max())
    candidates = np.flatnonzero(acc_valid >= best_acc - 1e-15)
    best_k = int(candidates.min())  # tie-break: tightest cut (lowest FAR)
    if best_k == 0:
        return float(s_sorted[0]) + 1e-12
    if best_k == n:
        return float(s_sorted[-1]) - 1e-12
    hi = float(s_sorted[best_k - 1])
    lo = float(s_sorted[best_k])
    # valid[best_k]=True implies hi > lo.
    return 0.5 * (hi + lo)


def _threshold_balanced_accuracy_optimal(
    y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Return a threshold t maximising BALANCED accuracy = (TPR + TNR)/2.

    Prior-invariant, so the val-optimal threshold generalises better than
    plain accuracy under temporal class-prior shift. Same achievable-cut
    constraint as the accuracy variant.
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    n = y_score_np.size
    if n == 0:
        return 0.0
    order = np.argsort(-y_score_np, kind="stable")
    y_sorted = y_true_np[order]
    s_sorted = y_score_np[order]
    n_pos = int(y_sorted.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return _threshold_accuracy_optimal(y_true_np, y_score_np)
    cum_tp = np.concatenate(([0], np.cumsum(y_sorted)))
    ks = np.arange(n + 1, dtype=np.float64)
    cum_fp = ks - cum_tp
    tpr = cum_tp / float(n_pos)
    tnr = (float(n_neg) - cum_fp) / float(n_neg)
    bal_acc = 0.5 * (tpr + tnr)
    valid = np.zeros(n + 1, dtype=bool)
    valid[0] = True
    valid[n] = True
    if n > 1:
        valid[1:n] = s_sorted[:-1] > s_sorted[1:]
    ba_valid = np.where(valid, bal_acc, -np.inf)
    best_ba = float(ba_valid.max())
    candidates = np.flatnonzero(ba_valid >= best_ba - 1e-15)
    best_k = int(candidates.min())
    if best_k == 0:
        return float(s_sorted[0]) + 1e-12
    if best_k == n:
        return float(s_sorted[-1]) - 1e-12
    hi = float(s_sorted[best_k - 1])
    lo = float(s_sorted[best_k])
    return 0.5 * (hi + lo)


def _isotonic_calibrate_scores(
    y_val: np.ndarray,
    s_val: np.ndarray,
    s_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression on (s_val, y_val), apply to s_val and s_test.

    Monotone-preserving; corrects distributional drift under temporal
    shift for methods whose raw scores are only ordinally correct.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(
        np.asarray(s_val, dtype=np.float64),
        np.asarray(y_val, dtype=np.float64),
    )
    s_val_cal = iso.predict(np.asarray(s_val, dtype=np.float64))
    s_test_cal = iso.predict(np.asarray(s_test, dtype=np.float64))
    return s_val_cal.astype(np.float64), s_test_cal.astype(np.float64)


# -------------------------------------------------------------------------
# model fit / score helpers
# -------------------------------------------------------------------------

def fit_isolation_forest(
    X_train_benign: np.ndarray,
    n_estimators: int,
    seed: int,
) -> tuple[IsolationForest, float]:
    log(
        f"training IsolationForest(n_estimators={n_estimators}) on "
        f"{X_train_benign.shape[0]:,} benign flows x {X_train_benign.shape[1]} features"
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples="auto",
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(X_train_benign)
    fit_s = time.perf_counter() - t0
    return model, fit_s


def score_isolation_forest(model: IsolationForest, X_eval: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    scores = -model.decision_function(X_eval)
    score_s = time.perf_counter() - t0
    return scores.astype(np.float64), score_s


def fit_one_class_svm(
    X_train_benign: np.ndarray,
    train_size: int,
    nu: float,
    gamma: str | float,
    seed: int,
) -> tuple[OneClassSVM, float, int]:
    rng = np.random.default_rng(seed)
    n = X_train_benign.shape[0]
    if n > train_size:
        idx = rng.choice(n, size=train_size, replace=False)
        X_tr = X_train_benign[idx]
        log(f"subsampling OC-SVM training set: {n:,} -> {train_size:,} benign flows")
    else:
        X_tr = X_train_benign
    log(
        f"training OneClassSVM(kernel='rbf', nu={nu}, gamma={gamma!r}) on "
        f"{X_tr.shape[0]:,} flows x {X_tr.shape[1]} features"
    )
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma, cache_size=1000)
    t0 = time.perf_counter()
    model.fit(X_tr)
    fit_s = time.perf_counter() - t0
    return model, fit_s, int(X_tr.shape[0])


def score_one_class_svm(model: OneClassSVM, X_eval: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    scores = -model.decision_function(X_eval)
    score_s = time.perf_counter() - t0
    return scores.astype(np.float64), score_s


def fit_and_score_lstm_autoencoder(
    X_train_benign: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    seq_len: int,
    hidden_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    train_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fit one LSTM autoencoder and score both validation and test sets.

    Returns
    -------
    val_scores   : per-row anomaly scores on validation set
    test_scores  : per-row anomaly scores on test set
    fit_seconds  : training time
    score_seconds: test-set scoring time (as returned by helper)
    """
    result = lstm_autoencoder_scores(
        X_train_benign=X_train_benign,
        X_test=X_test,
        X_val=X_val,
        seq_len=seq_len,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_size=train_size,
        device=device,
        seed=seed,
    )

    # Expected project helper format:
    #   (val_scores, test_scores, fit_s, score_s)
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError(
            f"Unexpected return format from lstm_autoencoder_scores(...): "
            f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
        )

    val_scores, test_scores, fit_s, score_s = result

    if val_scores is None:
        raise ValueError(
            "lstm_autoencoder_scores(...) returned val_scores=None even though X_val was provided."
        )

    return (
        np.asarray(val_scores, dtype=np.float64),
        np.asarray(test_scores, dtype=np.float64),
        float(fit_s),
        float(score_s),
    )
# -------------------------------------------------------------------------
# evaluation
# -------------------------------------------------------------------------

def evaluate_baseline_valcal(
    method_name: str,
    y_val: np.ndarray,
    y_score_val: np.ndarray,
    y_test: np.ndarray,
    y_score_test: np.ndarray,
    target_far: float,
    fit_seconds: float,
    score_seconds: float,
    include_test_optimistic: bool,
    include_balanced_accuracy: bool = True,
    calibrate_isotonic: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _emit_accuracy_pair(
        s_val: np.ndarray,
        s_test: np.ndarray,
        suffix: str,
    ) -> None:
        # accuracy-optimal
        t_acc = _threshold_accuracy_optimal(y_val, s_val)
        yhat = (s_test >= t_acc).astype(np.int32)
        op = f"val_accuracy_calibrated{suffix}"
        r = _metric_row(
            method=method_name,
            operating_point=op,
            threshold=t_acc,
            threshold_source="validation",
            y_true=y_test,
            y_score=s_test,
            y_pred=yhat,
            fit_seconds=fit_seconds,
            score_seconds=score_seconds,
        )
        rows.append(r)
        log(
            f"{method_name} @ {op}: acc={r['accuracy']:.4f} "
            f"prec={r['precision']:.4f} rec={r['recall']:.4f} "
            f"f1={r['f1']:.4f} far={r['far']:.2e} (tau={t_acc:.6g})"
        )
        # balanced-accuracy-optimal (prior-invariant, often more stable under shift)
        if include_balanced_accuracy:
            t_ba = _threshold_balanced_accuracy_optimal(y_val, s_val)
            yhat_ba = (s_test >= t_ba).astype(np.int32)
            op_ba = f"val_balanced_accuracy_calibrated{suffix}"
            r_ba = _metric_row(
                method=method_name,
                operating_point=op_ba,
                threshold=t_ba,
                threshold_source="validation",
                y_true=y_test,
                y_score=s_test,
                y_pred=yhat_ba,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
            rows.append(r_ba)
            log(
                f"{method_name} @ {op_ba}: acc={r_ba['accuracy']:.4f} "
                f"prec={r_ba['precision']:.4f} rec={r_ba['recall']:.4f} "
                f"f1={r_ba['f1']:.4f} far={r_ba['far']:.2e} (tau={t_ba:.6g})"
            )

    # 0) MAIN ROWS: validation-calibrated operating points on the raw scores.
    _emit_accuracy_pair(y_score_val, y_score_test, suffix="")

    # 0b) Optional isotonic-calibrated variants (ablation).
    if calibrate_isotonic:
        s_val_iso, s_test_iso = _isotonic_calibrate_scores(
            y_val, y_score_val, y_score_test
        )
        _emit_accuracy_pair(s_val_iso, s_test_iso, suffix="_isotonic")

    # 1) validation F1-calibrated threshold -> evaluate on test
    t_val_f1 = _threshold_f1_optimal(y_val, y_score_val)
    yhat_test_from_val_f1 = (y_score_test >= t_val_f1).astype(np.int32)
    row_val_f1 = _metric_row(
        method=method_name,
        operating_point="val_f1_calibrated",
        threshold=t_val_f1,
        threshold_source="validation",
        y_true=y_test,
        y_score=y_score_test,
        y_pred=yhat_test_from_val_f1,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
    )
    rows.append(row_val_f1)

    # 2) validation FAR-calibrated threshold -> evaluate on test
    t_val_far = _threshold_for_target_far(y_val, y_score_val, target_far)
    yhat_test_from_val_far = (y_score_test >= t_val_far).astype(np.int32)
    row_val_far = _metric_row(
        method=method_name,
        operating_point=f"val_far_calibrated_{target_far:.2e}",
        threshold=t_val_far,
        threshold_source="validation",
        y_true=y_test,
        y_score=y_score_test,
        y_pred=yhat_test_from_val_far,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
    )
    rows.append(row_val_far)

    log(
        f"{method_name} @ val-F1-calibrated: "
        f"acc={row_val_f1['accuracy']:.4f} prec={row_val_f1['precision']:.4f} "
        f"rec={row_val_f1['recall']:.4f} f1={row_val_f1['f1']:.4f} far={row_val_f1['far']:.2e}"
    )
    log(
        f"{method_name} @ val-FAR={target_far:.1e}: "
        f"acc={row_val_far['accuracy']:.4f} prec={row_val_far['precision']:.4f} "
        f"rec={row_val_far['recall']:.4f} f1={row_val_far['f1']:.4f} far={row_val_far['far']:.2e}"
    )

    if include_test_optimistic:
        # benchmark-only optimistic rows
        t_test_f1 = _threshold_f1_optimal(y_test, y_score_test)
        yhat_test_f1 = (y_score_test >= t_test_f1).astype(np.int32)
        rows.append(
            _metric_row(
                method=method_name,
                operating_point="test_f1_optimal",
                threshold=t_test_f1,
                threshold_source="test",
                y_true=y_test,
                y_score=y_score_test,
                y_pred=yhat_test_f1,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )

        t_test_far = _threshold_for_target_far(y_test, y_score_test, target_far)
        yhat_test_far = (y_score_test >= t_test_far).astype(np.int32)
        rows.append(
            _metric_row(
                method=method_name,
                operating_point=f"test_far_matched_{target_far:.2e}",
                threshold=t_test_far,
                threshold_source="test",
                y_true=y_test,
                y_score=y_score_test,
                y_pred=yhat_test_far,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )

    return rows


# -------------------------------------------------------------------------
# LaTeX emitter
# -------------------------------------------------------------------------

def _fmt_sci(x: float) -> str:
    if x == 0.0:
        return "$0$"
    exp = int(np.floor(np.log10(x)))
    base = x / (10 ** exp)
    return f"${base:.1f}{{\\times}}10^{{{exp}}}$"


def emit_latex_fragment(
    rows: list[dict[str, Any]],
    out_path: Path,
    headline_operating_point: str = "val_accuracy_calibrated",
) -> None:
    """Emit one row per method at the chosen headline operating point,
    sorted by accuracy (desc).

    The operating point must be declared in advance (CLI flag); the
    emitter never picks by test metrics. If a method lacks the chosen
    row (e.g., isotonic was off), it falls back in this order:
      val_balanced_accuracy_calibrated_isotonic
      val_accuracy_calibrated_isotonic
      val_balanced_accuracy_calibrated
      val_accuracy_calibrated
      val_far_calibrated_*

    Columns: Accuracy | Precision | Recall | F1 | FAR | ROC-AUC | PR-AUC.
    """
    fallback_order = [
        headline_operating_point,
        "val_balanced_accuracy_calibrated_isotonic",
        "val_accuracy_calibrated_isotonic",
        "val_balanced_accuracy_calibrated",
        "val_accuracy_calibrated",
    ]
    # de-duplicate while preserving order
    seen: set[str] = set()
    fallback_order = [op for op in fallback_order if not (op in seen or seen.add(op))]

    lines = [
        "% Auto-generated by compare_anomaly_baselines_valcal.py.",
        f"% Headline operating point: {headline_operating_point}.",
        "% Threshold picked on D_val, applied unchanged to D_test.",
        "% Columns: Method | Acc | Prec | Rec | F1 | FAR | ROC-AUC | PR-AUC.",
        "% Rows sorted by accuracy (descending) per locked protocol.",
        "",
    ]

    by_method: dict[str, dict[str, Any]] = {}
    # precompute: method -> {operating_point -> row}
    per_method_ops: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        per_method_ops.setdefault(r["method"], {})[str(r["operating_point"])] = r
    for method, ops in per_method_ops.items():
        chosen: dict[str, Any] | None = None
        for op in fallback_order:
            if op in ops:
                chosen = ops[op]
                break
        if chosen is None:
            # last-resort FAR row
            far_ops = [op for op in ops if op.startswith("val_far_calibrated_")]
            if far_ops:
                chosen = ops[far_ops[0]]
        if chosen is not None:
            by_method[method] = chosen

    selected = sorted(
        by_method.values(),
        key=lambda r: float(r["accuracy"]),
        reverse=True,
    )
    for r in selected:
        pr_auc = r.get("pr_auc", float("nan"))
        lines.append(
            f"{r['method']:<30s} & {r['accuracy']:.4f} & {r['precision']:.4f} & "
            f"{r['recall']:.4f} & {r['f1']:.4f} & {_fmt_sci(float(r['far']))} & "
            f"{r['roc_auc']:.3f} & {float(pr_auc):.3f} \\\\"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote LaTeX fragment: {out_path}")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="CIC-IDS2017 CSV folder")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "temporal", "temporal_by_file"],
        # Locked environment: temporal_by_file (day-level) split.
        default="temporal_by_file",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size-from-train", type=float, default=0.20)
    parser.add_argument(
        "--target-far",
        type=float,
        default=8.1e-4,
        help="Validation FAR target used to calibrate threshold before test evaluation.",
    )
    parser.add_argument("--iforest-n-estimators", type=int, default=200)
    parser.add_argument("--ocsvm-train-size", type=int, default=20000)
    parser.add_argument("--ocsvm-nu", type=float, default=0.01)
    parser.add_argument("--ocsvm-gamma", default="scale")
    parser.add_argument("--skip-ocsvm", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--lstm-seq-len", type=int, default=10)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-latent", type=int, default=32)
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--lstm-batch", type=int, default=256)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-train-size", type=int, default=200_000)
    parser.add_argument("--lstm-device", default="cpu")
    parser.add_argument(
        "--include-test-optimistic",
        action="store_true",
        help="Also emit benchmark-only rows using thresholds selected on the test set.",
    )
    parser.add_argument(
        "--calibrate-isotonic",
        action="store_true",
        help="Fit isotonic regression on D_val scores before threshold selection. "
             "Emits additional '*_isotonic' rows so the effect is ablatable.",
    )
    parser.add_argument(
        "--skip-val-balanced-accuracy",
        dest="include_val_balanced_accuracy",
        action="store_false",
        help="Skip the val-balanced-accuracy-calibrated row (emitted by default).",
    )
    parser.set_defaults(include_val_balanced_accuracy=True)
    parser.add_argument(
        "--headline-operating-point",
        default="val_accuracy_calibrated",
        choices=[
            "val_accuracy_calibrated",
            "val_balanced_accuracy_calibrated",
            "val_accuracy_calibrated_isotonic",
            "val_balanced_accuracy_calibrated_isotonic",
        ],
        help="Which operating point the LaTeX fragment headlines per method.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading data from {args.data_dir} | split_strategy={args.split_strategy}")
    cleaned, splits = load_and_prepare_detection_data(
        data_dir=args.data_dir,
        test_size=args.test_size,
        val_size_from_train=args.val_size_from_train,
        random_state=args.seed,
        split_strategy=args.split_strategy,
    )
    log(
        f"splits | train={len(splits.train_all):,} val={len(splits.val_all):,} "
        f"test={len(splits.test_all):,} | train_benign={len(splits.train_benign):,}"
    )

    feature_cols = select_feature_columns(cleaned, exclude_columns=DEFAULT_EXCLUDE_COLUMNS)
    log(f"feature columns: {len(feature_cols)} (shared with RF baseline)")
    pre = build_tabular_preprocessor(cleaned, feature_cols, scale_numeric=True)
    pre.fit(splits.train_benign[feature_cols])

    X_train_benign = pre.transform(splits.train_benign[feature_cols])
    X_val = pre.transform(splits.val_all[feature_cols])
    X_test = pre.transform(splits.test_all[feature_cols])

    y_val = splits.val_all["binary_label"].to_numpy().astype(np.int32)
    y_test = splits.test_all["binary_label"].to_numpy().astype(np.int32)

    transformed_dim = X_train_benign.shape[1]
    if transformed_dim > 500:
        raise RuntimeError(
            f"Preprocessor produced {transformed_dim} output dims; likely a high-cardinality identifier leaked through."
        )

    if hasattr(X_train_benign, "toarray"):
        X_train_benign = X_train_benign.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)
    else:
        X_train_benign = np.asarray(X_train_benign, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

    log(
        f"feature matrix | X_train_benign={X_train_benign.shape} "
        f"X_val={X_val.shape} X_test={X_test.shape}"
    )

    all_rows: list[dict[str, Any]] = []

    # Isolation Forest
    if_model, if_fit_s = fit_isolation_forest(
        X_train_benign=X_train_benign,
        n_estimators=args.iforest_n_estimators,
        seed=args.seed,
    )
    if_val_scores, if_val_score_s = score_isolation_forest(if_model, X_val)
    if_test_scores, if_test_score_s = score_isolation_forest(if_model, X_test)
    all_rows.extend(
        evaluate_baseline_valcal(
            method_name="Isolation Forest",
            y_val=y_val,
            y_score_val=if_val_scores,
            y_test=y_test,
            y_score_test=if_test_scores,
            target_far=args.target_far,
            fit_seconds=if_fit_s,
            score_seconds=if_val_score_s + if_test_score_s,
            include_test_optimistic=args.include_test_optimistic,
            include_balanced_accuracy=args.include_val_balanced_accuracy,
            calibrate_isotonic=args.calibrate_isotonic,
        )
    )

    # One-Class SVM
    if not args.skip_ocsvm:
        oc_model, oc_fit_s, oc_train_n = fit_one_class_svm(
            X_train_benign=X_train_benign,
            train_size=args.ocsvm_train_size,
            nu=args.ocsvm_nu,
            gamma=args.ocsvm_gamma,
            seed=args.seed,
        )
        oc_val_scores, oc_val_score_s = score_one_class_svm(oc_model, X_val)
        oc_test_scores, oc_test_score_s = score_one_class_svm(oc_model, X_test)
        all_rows.extend(
            evaluate_baseline_valcal(
                method_name="One-Class SVM",
                y_val=y_val,
                y_score_val=oc_val_scores,
                y_test=y_test,
                y_score_test=oc_test_scores,
                target_far=args.target_far,
                fit_seconds=oc_fit_s,
                score_seconds=oc_val_score_s + oc_test_score_s,
                include_test_optimistic=args.include_test_optimistic,
                include_balanced_accuracy=args.include_val_balanced_accuracy,
                calibrate_isotonic=args.calibrate_isotonic,
            )
        )
    else:
        oc_train_n = 0
        log("skipping One-Class SVM (per --skip-ocsvm)")

        # LSTM Autoencoder
    if not args.skip_lstm:
        def _time_order(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ts = None
            for col in ("timestamp", "Timestamp"):
                if col in df.columns:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().any():
                        ts = parsed
                        break
            if ts is not None:
                order = np.argsort(ts.fillna(pd.Timestamp.min).to_numpy(), kind="stable")
            else:
                order = np.arange(X.shape[0])
            return X[order], y[order]

        X_val_sorted, y_val_sorted = _time_order(splits.val_all, X_val, y_val)
        X_test_sorted, y_test_sorted = _time_order(splits.test_all, X_test, y_test)

        log("training/scoring LSTM autoencoder on validation + test splits")
        lstm_val_scores, lstm_test_scores, lstm_fit_s, lstm_score_s = fit_and_score_lstm_autoencoder(
            X_train_benign=X_train_benign,
            X_val=X_val_sorted,
            X_test=X_test_sorted,
            seq_len=args.lstm_seq_len,
            hidden_size=args.lstm_hidden,
            latent_dim=args.lstm_latent,
            epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            lr=args.lstm_lr,
            train_size=args.lstm_train_size,
            device=args.lstm_device,
            seed=args.seed,
        )

        all_rows.extend(
            evaluate_baseline_valcal(
                method_name="LSTM Autoencoder",
                y_val=y_val_sorted,
                y_score_val=lstm_val_scores,
                y_test=y_test_sorted,
                y_score_test=lstm_test_scores,
                target_far=args.target_far,
                fit_seconds=lstm_fit_s,
                score_seconds=lstm_score_s,
                include_test_optimistic=args.include_test_optimistic,
                include_balanced_accuracy=args.include_val_balanced_accuracy,
                calibrate_isotonic=args.calibrate_isotonic,
            )
        )
    else:
        log("skipping LSTM autoencoder (per --skip-lstm)")

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "overall_metrics_baselines.csv"
    df.to_csv(csv_path, index=False)
    log(f"wrote metrics: {csv_path}")

    latex_path = out_dir / "baselines_table_fragment.tex"
    emit_latex_fragment(
        all_rows,
        latex_path,
        headline_operating_point=args.headline_operating_point,
    )

    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data_dir": args.data_dir,
                "out_dir": str(out_dir),
                "split_strategy": args.split_strategy,
                "seed": args.seed,
                "test_size": args.test_size,
                "val_size_from_train": args.val_size_from_train,
                "target_far": args.target_far,
                "iforest_n_estimators": args.iforest_n_estimators,
                "ocsvm_train_size": args.ocsvm_train_size,
                "ocsvm_nu": args.ocsvm_nu,
                "ocsvm_gamma": args.ocsvm_gamma,
                "skip_ocsvm": args.skip_ocsvm,
                "skip_lstm": args.skip_lstm,
                "lstm_seq_len": args.lstm_seq_len,
                "lstm_hidden": args.lstm_hidden,
                "lstm_latent": args.lstm_latent,
                "lstm_epochs": args.lstm_epochs,
                "lstm_batch": args.lstm_batch,
                "lstm_lr": args.lstm_lr,
                "lstm_train_size": args.lstm_train_size,
                "lstm_device": args.lstm_device,
                "include_test_optimistic": args.include_test_optimistic,
                "calibrate_isotonic": bool(args.calibrate_isotonic),
                "include_val_balanced_accuracy": bool(
                    args.include_val_balanced_accuracy
                ),
                "headline_operating_point": args.headline_operating_point,
                "n_features": int(X_test.shape[1]),
                "n_train_benign": int(X_train_benign.shape[0]),
                "n_val": int(X_val.shape[0]),
                "n_test": int(X_test.shape[0]),
                "ocsvm_actual_train_n": int(oc_train_n),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote run config: {cfg_path}")
    log("done.")


if __name__ == "__main__":
    main()
