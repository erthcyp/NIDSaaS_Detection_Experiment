#!/usr/bin/env python3
"""
compare_anomaly_baselines.py
============================

Fair head-to-head comparison of two common unsupervised anomaly-detection
baselines against the Hybrid-Cascade on the CIC-IDS2017 temporal split.

Baselines
---------
  1. Isolation Forest  (sklearn.ensemble.IsolationForest)
       Tree-based, benign-only training, score = -decision_function.
  2. One-Class SVM     (sklearn.svm.OneClassSVM, RBF kernel)
       Kernel-based, benign-only training (subsampled for tractability),
       score = -decision_function.

Both baselines follow the standard one-class anomaly-detection convention:
    fit on D_train benigns only, score on D_test, threshold the score.

Each baseline is evaluated at FOUR operating points per run:
    (a) test_f1_optimal          -- F1-optimal threshold picked on D_test
                                    itself (upper-bound, test-set peek).
    (b) test_far_matched_<FAR>   -- FAR-matching threshold picked on D_test
                                    benigns (still test-set peek).
    (c) val_f1_optimal           -- F1-optimal threshold picked on D_val,
                                    applied unchanged to D_test (fair, no
                                    test leakage).
    (d) val_far_matched_<FAR>    -- FAR-matching threshold picked on D_val
                                    benigns to hit `target_far`, applied
                                    unchanged to D_test. This is the
                                    HEADLINE row used in the main table:
                                    symmetric with the Hybrid-Cascade's
                                    default gate threshold (which is also
                                    chosen off the test set).

Outputs
-------
  <out_dir>/overall_metrics_baselines.csv
      one row per {method, operating-point} combination with accuracy,
      precision, recall, F1, FAR, ROC-AUC, threshold, TP/FP/TN/FN.

  <out_dir>/baselines_table_fragment.tex
      drop-in LaTeX rows for the main comparison table.

Usage
-----
  python compare_anomaly_baselines.py \
      --data-dir /path/to/MachineLearningCVE \
      --out-dir outputs_baselines_temporal \
      --split-strategy temporal \
      --seed 42 \
      --ocsvm-train-size 20000 \
      --target-far 8.1e-4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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


# Identifier / high-cardinality columns that must not reach OneHotEncoder.
# Mirrors RFConfig.exclude_columns in config.py so the baselines see the
# exact same feature space as the RF anomaly scorer.
DEFAULT_EXCLUDE_COLUMNS: tuple[str, ...] = RFConfig.exclude_columns


def log(msg: str) -> None:
    print(f"[baselines] {msg}", flush=True)


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
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    fit_seconds: float,
    score_seconds: float,
) -> dict:
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
        "threshold": threshold,
        # Locked-environment metric list (accuracy reported first).
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "far": _far_from_confusion(y_true, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
    }


def _threshold_for_target_far(
    y_true: np.ndarray, y_score: np.ndarray, target_far: float
) -> float:
    """Smallest threshold t such that FPR(y_score >= t) <= target_far."""
    benign_scores = y_score[y_true == 0]
    if benign_scores.size == 0:
        return float("inf")
    # Want benign FPR <= target_far -> allow at most floor(n * target_far) FPs.
    n_benign = benign_scores.size
    allowed_fp = int(np.floor(n_benign * target_far))
    # Sorted descending; the (allowed_fp + 1)-th largest benign score is the
    # tightest threshold that keeps FP <= allowed_fp when using "score >= t".
    sorted_desc = np.sort(benign_scores)[::-1]
    if allowed_fp >= n_benign:
        return float(sorted_desc[-1]) - 1.0  # accept every benign (no-op)
    return float(sorted_desc[allowed_fp]) + 1e-12  # smallest t above the (k+1)-th


def _threshold_f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of length n-1 relative to p/r.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    # Last entry of precisions/recalls has no corresponding threshold; drop it.
    f1s = f1s[:-1]
    if f1s.size == 0 or thresholds.size == 0:
        return float(np.median(y_score))
    best_idx = int(np.nanargmax(f1s))
    return float(thresholds[best_idx])


def _threshold_accuracy_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold t maximising accuracy of (y_score >= t). O(n log n).

    Only achievable thresholds are considered: rank cuts interior to a tied
    score plateau are skipped because no threshold can split tied rows.
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
    cum_tp = np.concatenate(([0], np.cumsum(y_sorted)))
    ks = np.arange(n + 1, dtype=np.float64)
    acc = (2.0 * cum_tp + n_neg - ks) / float(n)
    valid = np.zeros(n + 1, dtype=bool)
    valid[0] = True
    valid[n] = True
    if n > 1:
        valid[1:n] = s_sorted[:-1] > s_sorted[1:]
    acc_valid = np.where(valid, acc, -np.inf)
    best_acc = float(acc_valid.max())
    candidates = np.flatnonzero(acc_valid >= best_acc - 1e-15)
    best_k = int(candidates.min())  # tie-break: tightest (fewest positives)
    if best_k == 0:
        return float(s_sorted[0]) + 1e-12
    if best_k == n:
        return float(s_sorted[-1]) - 1e-12
    hi = float(s_sorted[best_k - 1])
    lo = float(s_sorted[best_k])
    return 0.5 * (hi + lo)


# -------------------------------------------------------------------------
# baselines
# -------------------------------------------------------------------------

def run_isolation_forest(
    X_train_benign: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (val_scores, test_scores, fit_seconds, score_seconds_test)."""
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

    log(f"scoring {X_val.shape[0]:,} val flows with IsolationForest")
    val_scores = -model.decision_function(X_val)

    log(f"scoring {X_test.shape[0]:,} test flows with IsolationForest")
    t1 = time.perf_counter()
    test_scores = -model.decision_function(X_test)
    score_s = time.perf_counter() - t1
    return val_scores.astype(np.float64), test_scores.astype(np.float64), fit_s, score_s


def run_one_class_svm(
    X_train_benign: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_size: int,
    nu: float,
    gamma: str | float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (val_scores, test_scores, fit_seconds, score_seconds_test)."""
    rng = np.random.default_rng(seed)
    n = X_train_benign.shape[0]
    if n > train_size:
        idx = rng.choice(n, size=train_size, replace=False)
        X_tr = X_train_benign[idx]
        log(
            f"subsampling OC-SVM training set: {n:,} -> {train_size:,} benign flows"
        )
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

    log(f"scoring {X_val.shape[0]:,} val flows with OneClassSVM")
    val_scores = -model.decision_function(X_val)

    log(f"scoring {X_test.shape[0]:,} test flows with OneClassSVM")
    t1 = time.perf_counter()
    test_scores = -model.decision_function(X_test)
    score_s = time.perf_counter() - t1
    return val_scores.astype(np.float64), test_scores.astype(np.float64), fit_s, score_s


# -------------------------------------------------------------------------
# evaluation driver
# -------------------------------------------------------------------------

def evaluate_baseline(
    method_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_far: float,
    fit_seconds: float,
    score_seconds: float,
    y_val: np.ndarray | None = None,
    y_val_score: np.ndarray | None = None,
) -> list[dict]:
    """Score the baseline at five operating points and return metric rows.

    Operating points
    ----------------
      (0) val_accuracy_optimal -- Threshold maximising ACCURACY on D_val,
                                  applied to D_test. No test peek.
                                  HEADLINE row per the locked environment
                                  (main reported operating point).
      (a) test_f1_optimal      -- F1-maxing threshold picked ON TEST
                                  (upper-bound reference / appendix).
      (b) test_far_matched     -- FAR-matching threshold picked ON TEST.
      (c) val_f1_optimal       -- F1-maxing threshold picked on D_val,
                                  applied to D_test. No test peek.
      (d) val_far_matched      -- FAR-matching threshold picked on D_val
                                  benigns to target `target_far`, applied
                                  to D_test. No test peek.
    """
    rows: list[dict] = []

    # ---- (0) val accuracy-optimal (MAIN operating point, locked env) ----
    if y_val is not None and y_val_score is not None:
        t_acc_va = _threshold_accuracy_optimal(y_val, y_val_score)
        rows.append(
            _metric_row(
                method=method_name,
                operating_point="val_accuracy_optimal",
                threshold=t_acc_va,
                y_true=y_true,
                y_score=y_score,
                y_pred=(y_score >= t_acc_va).astype(np.int32),
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )

    # ---- (a) test F1-optimal (test-set peek, upper-bound reference) ----
    t_f1_te = _threshold_f1_optimal(y_true, y_score)
    rows.append(
        _metric_row(
            method=method_name,
            operating_point="test_f1_optimal",
            threshold=t_f1_te,
            y_true=y_true,
            y_score=y_score,
            y_pred=(y_score >= t_f1_te).astype(np.int32),
            fit_seconds=fit_seconds,
            score_seconds=score_seconds,
        )
    )

    # ---- (b) test FAR-matched (test-set peek, upper-bound reference) ----
    t_far_te = _threshold_for_target_far(y_true, y_score, target_far)
    rows.append(
        _metric_row(
            method=method_name,
            operating_point=f"test_far_matched_{target_far:.2e}",
            threshold=t_far_te,
            y_true=y_true,
            y_score=y_score,
            y_pred=(y_score >= t_far_te).astype(np.int32),
            fit_seconds=fit_seconds,
            score_seconds=score_seconds,
        )
    )

    # ---- (c) + (d) val-calibrated (no test leakage) ----
    if y_val is not None and y_val_score is not None:
        # F1-optimal chosen on validation, applied to test
        t_f1_va = _threshold_f1_optimal(y_val, y_val_score)
        rows.append(
            _metric_row(
                method=method_name,
                operating_point="val_f1_optimal",
                threshold=t_f1_va,
                y_true=y_true,
                y_score=y_score,
                y_pred=(y_score >= t_f1_va).astype(np.int32),
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )

        # FAR target chosen on validation benigns, applied to test
        t_far_va = _threshold_for_target_far(y_val, y_val_score, target_far)
        rows.append(
            _metric_row(
                method=method_name,
                operating_point=f"val_far_matched_{target_far:.2e}",
                threshold=t_far_va,
                y_true=y_true,
                y_score=y_score,
                y_pred=(y_score >= t_far_va).astype(np.int32),
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
            )
        )
    else:
        log(
            f"{method_name}: no validation scores provided, "
            "skipping val-calibrated operating points"
        )

    for r in rows:
        log(
            f"{method_name} @ {r['operating_point']:<28s}  "
            f"acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
            f"rec={r['recall']:.4f} f1={r['f1']:.4f} "
            f"far={r['far']:.2e} (thr={r['threshold']:.4g})"
        )
    return rows


# -------------------------------------------------------------------------
# LaTeX emitter
# -------------------------------------------------------------------------

def _fmt_sci(x: float) -> str:
    if x == 0.0:
        return "$0$"
    exp = int(np.floor(np.log10(x)))
    base = x / (10**exp)
    return f"${base:.1f}{{\\times}}10^{{{exp}}}$"


def emit_latex_fragment(rows: list[dict], out_path: Path) -> None:
    """Emit the fair, val-calibrated FAR-matched row per baseline.

    We prefer ``val_far_matched_*`` (threshold chosen on D_val, applied to
    D_test, no test-set peek). If a baseline only has test-calibrated rows
    (e.g. evaluate_baseline was called without y_val), fall back to the
    ``test_far_matched_*`` row so the fragment is never empty.
    """
    lines = [
        "% Auto-generated by compare_anomaly_baselines.py.",
        "% Rows use the val-calibrated FAR-matched operating point",
        "% (threshold picked on D_val benigns, applied unchanged to D_test).",
        "% Drop into the main results table after the RF baseline.",
        "",
    ]
    # Group rows by method and pick the best available operating point.
    # Preference order: val_accuracy_optimal (HEADLINE per locked env)
    #                 > val_far_matched > test_far_matched (fallback).
    by_method: dict[str, dict] = {}
    for r in rows:
        op = str(r["operating_point"])
        if op == "val_accuracy_optimal":
            by_method[r["method"]] = r  # always wins
        elif op.startswith("val_far_matched") and (
            r["method"] not in by_method
            or str(by_method[r["method"]]["operating_point"]) != "val_accuracy_optimal"
        ):
            by_method[r["method"]] = r
        elif op.startswith("test_far_matched") and r["method"] not in by_method:
            by_method[r["method"]] = r  # last-resort fallback

    # Sort by accuracy descending (locked-environment headline metric).
    selected = sorted(
        by_method.values(),
        key=lambda r: float(r.get("accuracy", 0.0)),
        reverse=True,
    )
    for r in selected:
        method = r["method"]
        acc = r["accuracy"]
        prec = r["precision"]
        rec = r["recall"]
        f1 = r["f1"]
        far = r["far"]
        auc = r["roc_auc"]
        pr_auc = r.get("pr_auc", float("nan"))
        lines.append(
            f"{method:<30s} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} "
            f"& {_fmt_sci(far)} & {auc:.3f} & {float(pr_auc):.3f} \\\\"
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
        help="FAR to match for the head-to-head row (default = Hybrid-Cascade FAR).",
    )
    parser.add_argument("--iforest-n-estimators", type=int, default=200)
    parser.add_argument(
        "--ocsvm-train-size",
        type=int,
        default=20000,
        help="Benign flows used to fit OC-SVM (O(n^2), so subsampled).",
    )
    parser.add_argument("--ocsvm-nu", type=float, default=0.01)
    parser.add_argument("--ocsvm-gamma", default="scale")
    parser.add_argument(
        "--skip-ocsvm",
        action="store_true",
        help="Skip the One-Class SVM baseline (useful for quick iForest-only runs).",
    )
    # ---- LSTM-autoencoder hyperparameters ----
    parser.add_argument("--skip-lstm", action="store_true", help="Skip the LSTM baseline.")
    parser.add_argument("--lstm-seq-len", type=int, default=10)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-latent", type=int, default=32)
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--lstm-batch", type=int, default=256)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument(
        "--lstm-train-size",
        type=int,
        default=200_000,
        help="Consecutive benign rows used to build LSTM training sequences.",
    )
    parser.add_argument("--lstm-device", default="cpu", help="Torch device, e.g. 'cpu' or 'cuda'.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load + split (same splits as the main experiment) ----
    log(
        f"loading data from {args.data_dir} | "
        f"split_strategy={args.split_strategy}"
    )
    cleaned, splits = load_and_prepare_detection_data(
        data_dir=args.data_dir,
        test_size=args.test_size,
        val_size_from_train=args.val_size_from_train,
        random_state=args.seed,
        split_strategy=args.split_strategy,
    )
    log(
        f"splits | train={len(splits.train_all):,} "
        f"val={len(splits.val_all):,} test={len(splits.test_all):,} "
        f"| train_benign={len(splits.train_benign):,}"
    )

    # ---- preprocess ----
    # Exclude identifier / high-cardinality columns (Flow ID, IPs, Timestamp,
    # SimillarHTTP) so they don't get one-hot-encoded into millions of dummies.
    # Same exclusion list as the RF anomaly scorer -> apples-to-apples features.
    feature_cols = select_feature_columns(
        cleaned, exclude_columns=DEFAULT_EXCLUDE_COLUMNS
    )
    log(
        f"feature columns: {len(feature_cols)} "
        f"(excluded {len(DEFAULT_EXCLUDE_COLUMNS)} identifier cols)"
    )
    pre = build_tabular_preprocessor(cleaned, feature_cols, scale_numeric=True)
    pre.fit(splits.train_benign[feature_cols])

    X_train_benign = pre.transform(splits.train_benign[feature_cols])
    X_val = pre.transform(splits.val_all[feature_cols])
    X_test = pre.transform(splits.test_all[feature_cols])
    y_val = splits.val_all["binary_label"].to_numpy().astype(np.int32)
    y_test = splits.test_all["binary_label"].to_numpy().astype(np.int32)

    # Fail fast on runaway one-hot expansion (e.g., if an identifier column
    # slips through). Anything > ~500 output dims is almost certainly a bug
    # for CIC-IDS2017's ~80 numeric flow features.
    transformed_dim = X_train_benign.shape[1]
    if transformed_dim > 500:
        raise RuntimeError(
            f"Preprocessor produced {transformed_dim} output dims, which is "
            f"almost certainly a high-cardinality categorical leaking through. "
            f"Check DEFAULT_EXCLUDE_COLUMNS and the dtype of your feature columns."
        )

    if hasattr(X_train_benign, "toarray"):  # sparse -> dense for OC-SVM RBF
        X_train_benign = X_train_benign.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)
    else:
        X_train_benign = np.asarray(X_train_benign, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

    log(
        f"feature matrix | X_train_benign={X_train_benign.shape} "
        f"X_test={X_test.shape}"
    )
    log(
        f"test-set composition | benign={(y_test == 0).sum():,} "
        f"attack={(y_test == 1).sum():,}"
    )

    all_rows: list[dict] = []

    # ---- baseline 1: Isolation Forest ----
    if_val_scores, if_test_scores, if_fit_s, if_score_s = run_isolation_forest(
        X_train_benign=X_train_benign,
        X_val=X_val,
        X_test=X_test,
        n_estimators=args.iforest_n_estimators,
        seed=args.seed,
    )
    all_rows.extend(
        evaluate_baseline(
            method_name="Isolation Forest",
            y_true=y_test,
            y_score=if_test_scores,
            target_far=args.target_far,
            fit_seconds=if_fit_s,
            score_seconds=if_score_s,
            y_val=y_val,
            y_val_score=if_val_scores,
        )
    )

    # ---- baseline 2: One-Class SVM ----
    if not args.skip_ocsvm:
        oc_val_scores, oc_test_scores, oc_fit_s, oc_score_s = run_one_class_svm(
            X_train_benign=X_train_benign,
            X_val=X_val,
            X_test=X_test,
            train_size=args.ocsvm_train_size,
            nu=args.ocsvm_nu,
            gamma=args.ocsvm_gamma,
            seed=args.seed,
        )
        all_rows.extend(
            evaluate_baseline(
                method_name="One-Class SVM",
                y_true=y_test,
                y_score=oc_test_scores,
                target_far=args.target_far,
                fit_seconds=oc_fit_s,
                score_seconds=oc_score_s,
                y_val=y_val,
                y_val_score=oc_val_scores,
            )
        )
    else:
        log("skipping One-Class SVM (per --skip-ocsvm)")

    # ---- baseline 3: LSTM autoencoder ----
    # LSTM needs a time-ordered sequence. We therefore:
    #   (1) sort val by timestamp  -> build val sequences        -> val scores
    #   (2) sort test by timestamp -> build test sequences       -> test scores
    # Labels are permuted identically so metrics remain aligned.
    if not args.skip_lstm:

        def _sort_by_ts(df: pd.DataFrame, tag: str, n_rows: int) -> np.ndarray:
            """Return a row permutation sorting df by its timestamp column."""
            ts = None
            for col in ("timestamp", "Timestamp"):
                if col in df.columns:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().any():
                        ts = parsed
                        break
            if ts is not None:
                order = np.argsort(
                    ts.fillna(pd.Timestamp.min).to_numpy(), kind="stable"
                )
                log(f"sorted {tag} set by timestamp for LSTM sequence construction")
            else:
                order = np.arange(n_rows)
                log(f"no usable timestamp column; LSTM uses row order as time proxy ({tag})")
            return order

        val_order = _sort_by_ts(splits.val_all, tag="val", n_rows=X_val.shape[0])
        test_order = _sort_by_ts(splits.test_all, tag="test", n_rows=X_test.shape[0])

        X_val_sorted = X_val[val_order]
        y_val_sorted = y_val[val_order]
        X_test_sorted = X_test[test_order]
        y_test_sorted = y_test[test_order]

        lstm_val_scores, lstm_test_scores, lstm_fit_s, lstm_score_s = (
            lstm_autoencoder_scores(
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
        )
        all_rows.extend(
            evaluate_baseline(
                method_name="LSTM Autoencoder",
                y_true=y_test_sorted,
                y_score=lstm_test_scores,
                target_far=args.target_far,
                fit_seconds=lstm_fit_s,
                score_seconds=lstm_score_s,
                y_val=y_val_sorted,
                y_val_score=lstm_val_scores,
            )
        )
    else:
        log("skipping LSTM autoencoder (per --skip-lstm)")

    # ---- persist outputs ----
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "overall_metrics_baselines.csv"
    df.to_csv(csv_path, index=False)
    log(f"wrote metrics: {csv_path}")

    latex_path = out_dir / "baselines_table_fragment.tex"
    emit_latex_fragment(all_rows, latex_path)

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
                "n_features": int(X_test.shape[1]),
                "n_train_benign": int(X_train_benign.shape[0]),
                "n_test": int(X_test.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote run config: {cfg_path}")

    log("done.")


if __name__ == "__main__":
    main()
