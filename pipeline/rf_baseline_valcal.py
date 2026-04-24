#!/usr/bin/env python3
"""
rf_baseline_valcal.py
=====================

Emit the "Random Forest" and "RF + Conformal" rows of the main comparison
table *without retraining the RF*. We reuse the RF scores already saved by
the hybrid-cascade run under the locked split (``temporal_by_file``,
seed 42, 64/16/20), and apply the new main operating point:

    validation-calibrated target ACCURACY
    (threshold maximising accuracy on D_val, applied unchanged to D_test).

Inputs
------
Two CSVs produced by ``hybrid_cascade_splitcal_fastsnort.py``:

    --val-csv  outputs_hybrid_cascade_splitcal_fastsnort_temporal/val_cascade_predictions.csv
    --test-csv outputs_hybrid_cascade_splitcal_fastsnort_temporal/test_cascade_predictions.csv

Each must contain the columns:

    binary_label, rf_score, rf_pvalue

Outputs
-------
<out_dir>/overall_metrics_rf_valcal.csv
<out_dir>/rf_table_fragment.tex
<out_dir>/run_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def log(msg: str) -> None:
    print(f"[rf-valcal] {msg}", flush=True)


# -------------------------------------------------------------------------
# shared helpers (mirror the other valcal scripts exactly)
# -------------------------------------------------------------------------

def _far_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    return float(fp) / float(denom) if denom else 0.0


def threshold_accuracy_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold t maximising accuracy of (y_score >= t).

    Exact O(n log n) search. Only achievable cuts are considered: rank
    cuts interior to a tied-score plateau are skipped because no threshold
    can split rows sharing a score. Tie-break prefers the tightest cut
    (lowest FAR).
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
    best_k = int(candidates.min())
    if best_k == 0:
        return float(s_sorted[0]) + 1e-12
    if best_k == n:
        return float(s_sorted[-1]) - 1e-12
    hi = float(s_sorted[best_k - 1])
    lo = float(s_sorted[best_k])
    return 0.5 * (hi + lo)


def threshold_balanced_accuracy_optimal(
    y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Threshold t maximising BALANCED accuracy = (TPR + TNR)/2.

    Prior-invariant, so the val-optimal threshold generalises better than
    plain accuracy under temporal class-prior shift. Same achievable-cut
    constraint as ``threshold_accuracy_optimal``.
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
        return threshold_accuracy_optimal(y_true_np, y_score_np)
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


def isotonic_calibrate_scores(
    y_val: np.ndarray,
    s_val: np.ndarray,
    s_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression on (s_val, y_val), apply to s_val and s_test."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(
        np.asarray(s_val, dtype=np.float64),
        np.asarray(y_val, dtype=np.float64),
    )
    s_val_cal = iso.predict(np.asarray(s_val, dtype=np.float64))
    s_test_cal = iso.predict(np.asarray(s_test, dtype=np.float64))
    return s_val_cal.astype(np.float64), s_test_cal.astype(np.float64)


def metric_row(
    method: str,
    operating_point: str,
    threshold_source: str,
    threshold: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
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
        "far": _far_from_confusion(y_true, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# -------------------------------------------------------------------------
# LaTeX fragment
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
    """One row per method at the pre-declared headline operating point,
    sorted by accuracy (desc). Falls back to other val-calibrated rows
    if the headline one is absent for a method.
    """
    fallback_order = [
        headline_operating_point,
        "val_balanced_accuracy_calibrated_isotonic",
        "val_accuracy_calibrated_isotonic",
        "val_balanced_accuracy_calibrated",
        "val_accuracy_calibrated",
    ]
    seen: set[str] = set()
    fallback_order = [op for op in fallback_order if not (op in seen or seen.add(op))]

    lines = [
        "% Auto-generated by rf_baseline_valcal.py.",
        f"% Headline operating point: {headline_operating_point}.",
        "% Threshold picked on D_val, applied unchanged to D_test.",
        "% Columns: Method | Acc | Prec | Rec | F1 | FAR | ROC-AUC | PR-AUC.",
        "% Rows sorted by accuracy (descending) per locked protocol.",
        "",
    ]
    per_method_ops: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        per_method_ops.setdefault(r["method"], {})[str(r["operating_point"])] = r
    selected_rows: list[dict[str, Any]] = []
    for method, ops in per_method_ops.items():
        chosen: dict[str, Any] | None = None
        for op in fallback_order:
            if op in ops:
                chosen = ops[op]
                break
        if chosen is not None:
            selected_rows.append(chosen)
    selected_rows.sort(key=lambda r: float(r["accuracy"]), reverse=True)
    for r in selected_rows:
        lines.append(
            f"{r['method']:<30s} & {r['accuracy']:.4f} & {r['precision']:.4f} & "
            f"{r['recall']:.4f} & {r['f1']:.4f} & {_fmt_sci(float(r['far']))} & "
            f"{r['roc_auc']:.3f} & {float(r['pr_auc']):.3f} \\\\"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote LaTeX fragment: {out_path}")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def _score_and_report(
    method: str,
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    include_balanced_accuracy: bool = True,
    calibrate_isotonic: bool = False,
) -> list[dict[str, Any]]:
    """Emit the val-calibrated rows for one method:

      - val_accuracy_calibrated           (headline)
      - val_balanced_accuracy_calibrated  (prior-invariant alternative)
      - *_isotonic variants when --calibrate-isotonic is set

    Returns the list of rows (in emission order).
    """
    out_rows: list[dict[str, Any]] = []

    def _emit_pair(
        s_val_in: np.ndarray,
        s_test_in: np.ndarray,
        suffix: str,
    ) -> None:
        tau_acc = threshold_accuracy_optimal(y_val, s_val_in)
        yhat_acc = (s_test_in >= tau_acc).astype(np.int32)
        op_acc = f"val_accuracy_calibrated{suffix}"
        row_acc = metric_row(
            method=method,
            operating_point=op_acc,
            threshold_source="validation",
            threshold=tau_acc,
            y_true=y_test,
            y_score=s_test_in,
            y_pred=yhat_acc,
        )
        out_rows.append(row_acc)
        log(
            f"{method} @ {op_acc} | tau={tau_acc:.6g} acc={row_acc['accuracy']:.4f} "
            f"prec={row_acc['precision']:.4f} rec={row_acc['recall']:.4f} "
            f"f1={row_acc['f1']:.4f} far={row_acc['far']:.2e} "
            f"roc={row_acc['roc_auc']:.3f} pr={row_acc['pr_auc']:.3f}"
        )
        if include_balanced_accuracy:
            tau_ba = threshold_balanced_accuracy_optimal(y_val, s_val_in)
            yhat_ba = (s_test_in >= tau_ba).astype(np.int32)
            op_ba = f"val_balanced_accuracy_calibrated{suffix}"
            row_ba = metric_row(
                method=method,
                operating_point=op_ba,
                threshold_source="validation",
                threshold=tau_ba,
                y_true=y_test,
                y_score=s_test_in,
                y_pred=yhat_ba,
            )
            out_rows.append(row_ba)
            log(
                f"{method} @ {op_ba} | tau={tau_ba:.6g} acc={row_ba['accuracy']:.4f} "
                f"prec={row_ba['precision']:.4f} rec={row_ba['recall']:.4f} "
                f"f1={row_ba['f1']:.4f} far={row_ba['far']:.2e} "
                f"roc={row_ba['roc_auc']:.3f} pr={row_ba['pr_auc']:.3f}"
            )

    # raw-score rows
    _emit_pair(s_val, s_test, suffix="")

    # isotonic-calibrated rows
    if calibrate_isotonic:
        s_val_iso, s_test_iso = isotonic_calibrate_scores(y_val, s_val, s_test)
        _emit_pair(s_val_iso, s_test_iso, suffix="_isotonic")

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-csv",
        default="outputs_hybrid_cascade_splitcal_fastsnort_temporal/val_cascade_predictions.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="outputs_hybrid_cascade_splitcal_fastsnort_temporal/test_cascade_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_rf_baseline_valcal",
    )
    parser.add_argument("--label-col", default="binary_label")
    parser.add_argument("--rf-score-col", default="rf_score")
    parser.add_argument("--rf-pvalue-col", default="rf_pvalue")
    parser.add_argument(
        "--skip-conformal",
        action="store_true",
        help="Skip the RF + Conformal row (useful if rf_pvalue column is absent).",
    )
    parser.add_argument(
        "--calibrate-isotonic",
        action="store_true",
        help="Fit isotonic regression on D_val scores before threshold selection. "
             "Emits '*_isotonic' rows alongside the raw-score rows.",
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
        help="Operating point the LaTeX fragment headlines per method.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading val CSV:  {args.val_csv}")
    val_df = pd.read_csv(args.val_csv, usecols=None)
    log(f"loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv, usecols=None)

    for df, name in ((val_df, "val"), (test_df, "test")):
        missing = [c for c in (args.label_col, args.rf_score_col) if c not in df.columns]
        if not args.skip_conformal and args.rf_pvalue_col not in df.columns:
            missing.append(args.rf_pvalue_col)
        if missing:
            raise ValueError(f"{name} CSV is missing required columns: {missing}")

    y_val = pd.to_numeric(val_df[args.label_col], errors="raise").to_numpy().astype(np.int32)
    y_test = pd.to_numeric(test_df[args.label_col], errors="raise").to_numpy().astype(np.int32)

    log(
        f"splits | val={len(y_val):,} (benign {(y_val == 0).sum():,}) "
        f"test={len(y_test):,} (benign {(y_test == 0).sum():,})"
    )

    rows: list[dict[str, Any]] = []

    # --- Random Forest row ---
    s_val_rf = pd.to_numeric(val_df[args.rf_score_col], errors="raise").to_numpy().astype(np.float64)
    s_test_rf = pd.to_numeric(test_df[args.rf_score_col], errors="raise").to_numpy().astype(np.float64)
    rows.extend(
        _score_and_report(
            method="Random Forest",
            y_val=y_val,
            s_val=s_val_rf,
            y_test=y_test,
            s_test=s_test_rf,
            include_balanced_accuracy=args.include_val_balanced_accuracy,
            calibrate_isotonic=args.calibrate_isotonic,
        )
    )

    # --- RF + Conformal row ---
    # p-value is "how benign-looking" (low p = more anomalous), so invert.
    if not args.skip_conformal:
        p_val = pd.to_numeric(val_df[args.rf_pvalue_col], errors="raise").to_numpy().astype(np.float64)
        p_test = pd.to_numeric(test_df[args.rf_pvalue_col], errors="raise").to_numpy().astype(np.float64)
        s_val_conf = 1.0 - p_val
        s_test_conf = 1.0 - p_test
        rows.extend(
            _score_and_report(
                method="RF + Conformal",
                y_val=y_val,
                s_val=s_val_conf,
                y_test=y_test,
                s_test=s_test_conf,
                include_balanced_accuracy=args.include_val_balanced_accuracy,
                calibrate_isotonic=args.calibrate_isotonic,
            )
        )
    else:
        log("skipping RF + Conformal row (per --skip-conformal)")

    # --- persist outputs ---
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "overall_metrics_rf_valcal.csv"
    df_out.to_csv(csv_path, index=False)
    log(f"wrote metrics: {csv_path}")

    tex_path = out_dir / "rf_table_fragment.tex"
    emit_latex_fragment(
        rows,
        tex_path,
        headline_operating_point=args.headline_operating_point,
    )

    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "val_csv": args.val_csv,
                "test_csv": args.test_csv,
                "label_col": args.label_col,
                "rf_score_col": args.rf_score_col,
                "rf_pvalue_col": args.rf_pvalue_col,
                "skip_conformal": args.skip_conformal,
                "calibrate_isotonic": bool(args.calibrate_isotonic),
                "include_val_balanced_accuracy": bool(
                    args.include_val_balanced_accuracy
                ),
                "headline_operating_point": args.headline_operating_point,
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "n_val_benign": int((y_val == 0).sum()),
                "n_test_benign": int((y_test == 0).sum()),
                "protocol": (
                    "Headline operating point: validation-calibrated target Accuracy. "
                    "Balanced-accuracy row (prior-invariant) and isotonic-calibrated "
                    "variants are emitted alongside for ablation. "
                    "RF model reused from the hybrid-cascade run "
                    "(split_strategy=temporal_by_file, seed=42); no retraining."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote run config: {cfg_path}")
    log("done.")


if __name__ == "__main__":
    main()
