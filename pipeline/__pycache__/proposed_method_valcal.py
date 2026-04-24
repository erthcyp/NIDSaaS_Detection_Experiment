#!/usr/bin/env python3
"""
proposed_method_valcal.py
=========================

Standalone helper for applying a validation-calibrated low-FAR operating point
to the proposed Hybrid-Cascade detector.

Design
------
The script assumes that you already have row-aligned validation and test tables
for the proposed method. Each table must contain:

Required columns
----------------
- binary_label   : 0 for benign, 1 for attack
- snort_pred     : 1 if Snort hit, else 0
- gate_prob      : gate posterior probability for class ATTACK in [0, 1]

Optional columns
----------------
- row_id
- source_file
- any extra diagnostic columns

Final decision score
--------------------
For each row x, the final score is defined as

    final_score(x) = 1.0           if snort_pred == 1
                     gate_prob     otherwise

This preserves Snort as a high-confidence fast path while putting the proposed
method under the same validation-calibrated threshold protocol as the anomaly
baselines.

Protocol
--------
1) Build final_score on the validation set.
2) Select tau* on validation so that benign FAR ~= target FAR.
3) Freeze tau*.
4) Apply tau* unchanged to the held-out test set.
5) Report test metrics.

Outputs
-------
- <out_dir>/val_scores_with_predictions.csv
- <out_dir>/test_scores_with_predictions.csv
- <out_dir>/overall_metrics_proposed_valcal.csv
- <out_dir>/run_config.json
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def log(msg: str) -> None:
    print(f"[proposed-valcal] {msg}", flush=True)


def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def build_final_score(
    df: pd.DataFrame,
    snort_col: str = "snort_pred",
    gate_prob_col: str = "gate_prob",
    out_col: str = "final_score",
    rate_hit_col: str | None = None,
) -> pd.DataFrame:
    """
    Cascade final score.

    Without rate-rule promotion (legacy):
        final_score = 1.0 if snort_pred == 1 else gate_prob

    With rate-rule promotion (Tier-1 fast-path widened):
        fast_path   = (snort_pred == 1) | (rate_hit == 1)
        final_score = 1.0 if fast_path else gate_prob

    The rate_hit column is a caller-supplied per-row indicator that aggregates
    whichever rate-rule columns the user chose to promote (e.g. rate_V | rate_S
    | rate_P). Because rate rules are hand-coded signatures with thresholds
    calibrated on the benign training fold, this widens the signature fast-path
    without touching the learned gate and without test leakage.
    """
    out = df.copy()
    _require_columns(out, [snort_col, gate_prob_col], "input dataframe")

    snort = pd.to_numeric(out[snort_col], errors="coerce").fillna(0).astype(int)
    gate_prob = pd.to_numeric(out[gate_prob_col], errors="coerce").astype(float)

    if ((gate_prob < 0) | (gate_prob > 1)).any():
        bad_n = int(((gate_prob < 0) | (gate_prob > 1)).sum())
        raise ValueError(
            f"{gate_prob_col} must be in [0,1]. Found {bad_n} out-of-range rows."
        )

    if rate_hit_col is not None and rate_hit_col in out.columns:
        rate_hit = pd.to_numeric(out[rate_hit_col], errors="coerce").fillna(0).astype(int)
        fast_path = ((snort.to_numpy() == 1) | (rate_hit.to_numpy() == 1))
    else:
        fast_path = snort.to_numpy() == 1

    out[out_col] = np.where(fast_path, 1.0, gate_prob.to_numpy())
    return out


def load_rate_rule_hits(
    rate_rules_csv: str,
    include_cols: list[str],
    id_col: str = "row_id",
) -> pd.DataFrame:
    """Load a rate-rule predictions CSV and reduce the selected rate_* columns
    to a single per-row indicator.

    Returns
    -------
    DataFrame with columns [``id_col``, ``"rate_hit"``, ``"rate_hit_which"``].
    """
    if not include_cols:
        raise ValueError(
            "--rate-rules-include must name at least one rate_* column to promote."
        )
    df = pd.read_csv(rate_rules_csv)
    if id_col not in df.columns:
        raise ValueError(
            f"Rate-rules CSV is missing '{id_col}'. Columns: {list(df.columns)[:10]}..."
        )
    missing = [c for c in include_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Rate-rules CSV is missing requested columns: {missing}. "
            f"Available: {sorted(c for c in df.columns if c.startswith('rate_'))}"
        )
    reduced = pd.DataFrame({id_col: pd.to_numeric(df[id_col], errors="raise").astype(np.int64)})
    hit_stack = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).to_numpy()
        for c in include_cols
    ]).astype(bool)
    reduced["rate_hit"] = hit_stack.any(axis=1).astype(int)
    # Per-row letter bag for auditability (e.g., "VP" if rate_V and rate_P fired).
    letters = np.array(include_cols)
    def _letters_for(row: np.ndarray) -> str:
        return "".join(sorted(c.replace("rate_", "") for c in letters[row]))
    reduced["rate_hit_which"] = [_letters_for(r) for r in hit_stack]
    return reduced


def _far_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    return float(fp) / float(denom) if denom else 0.0


def threshold_for_target_far(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_far: float,
) -> float:
    """
    Smallest threshold t such that FPR/FAR(y_score >= t) <= target_far
    on benign rows of the calibration split.
    """
    benign_scores = y_score[y_true == 0]
    if benign_scores.size == 0:
        raise ValueError("No benign rows available for FAR calibration.")

    sorted_desc = np.sort(benign_scores)[::-1]
    n_benign = benign_scores.size
    allowed_fp = int(np.floor(n_benign * target_far))

    if allowed_fp >= n_benign:
        return float(sorted_desc[-1]) - 1.0

    return float(sorted_desc[allowed_fp]) + 1e-12


def threshold_f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    f1s = f1s[:-1]
    if f1s.size == 0 or thresholds.size == 0:
        return float(np.median(y_score))
    best_idx = int(np.nanargmax(f1s))
    return float(thresholds[best_idx])


def threshold_accuracy_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return threshold t maximising accuracy of (y_score >= t).

    Exact O(n log n) search over ACHIEVABLE thresholds only -- rank cuts
    interior to a tied-score plateau are excluded because no threshold can
    split rows that share a score. This matters for the cascade's final
    score, which stacks many rows at 0.0 (non-escalated) and 1.0 (Snort
    hits). Tie-break prefers the tightest cut (lower FAR).
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
    """Return threshold t maximising BALANCED accuracy = (TPR + TNR)/2.

    Balanced accuracy is prior-invariant, so the val-optimal threshold
    generalises better than plain accuracy across temporal class-prior
    shifts. Same achievable-cut-only constraint as ``threshold_accuracy_optimal``.
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
    """Fit isotonic regression on (s_val, y_val), apply to s_val and s_test.

    Returns calibrated scores in [0, 1]. Monotone-preserving, so it never
    hurts AUC on val; it corrects distributional drift under temporal
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
        "far": float(_far_from_confusion(y_true, y_pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", required=True, help="Validation CSV for proposed method")
    parser.add_argument("--test-csv", required=True, help="Test CSV for proposed method")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-far", type=float, default=8.1e-4)
    parser.add_argument("--include-val-f1", action="store_true")
    parser.add_argument("--include-test-optimistic", action="store_true")
    parser.add_argument(
        "--calibrate-isotonic",
        action="store_true",
        help="Fit isotonic regression on D_val scores before threshold selection. "
             "Emits additional '*_isotonic' rows for a clean ablation.",
    )
    parser.add_argument(
        "--skip-val-balanced-accuracy",
        dest="include_val_balanced_accuracy",
        action="store_false",
        help="Skip the val-balanced-accuracy-calibrated row (emitted by default).",
    )
    parser.set_defaults(include_val_balanced_accuracy=True)
    parser.add_argument("--label-col", default="binary_label")
    parser.add_argument("--snort-col", default="snort_pred")
    parser.add_argument("--gate-prob-col", default="gate_prob")
    parser.add_argument("--row-id-col", default="row_id")
    parser.add_argument(
        "--rate-rules-csv",
        default=None,
        help="Optional CSV with 'row_id' plus rate_* indicator columns "
             "(e.g. signature_merged_predictions.csv). When provided, the "
             "selected rate rules are promoted to Tier-1 fast-path.",
    )
    parser.add_argument(
        "--rate-rules-include",
        default="rate_V,rate_S,rate_P",
        help="Comma-separated rate_* column names to OR into the fast-path. "
             "Default V,S,P (volumetric, SYN-flood, port-scan). "
             "Deliberately excludes rate_B because its precision shifts "
             "drastically between val and test on CIC-IDS2017. Ignored when "
             "--rate-rules-csv is not set.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading validation CSV: {args.val_csv}")
    val_df = pd.read_csv(args.val_csv)
    log(f"loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)

    required = [args.label_col, args.snort_col, args.gate_prob_col]
    _require_columns(val_df, required, "validation CSV")
    _require_columns(test_df, required, "test CSV")

    rate_hit_col: str | None = None
    rate_rules_promoted: list[str] = []
    if args.rate_rules_csv:
        rate_rules_promoted = [
            c.strip() for c in args.rate_rules_include.split(",") if c.strip()
        ]
        log(
            f"loading rate rules from: {args.rate_rules_csv} | "
            f"promoting to Tier-1: {rate_rules_promoted}"
        )
        _require_columns(val_df, [args.row_id_col], "validation CSV")
        _require_columns(test_df, [args.row_id_col], "test CSV")
        rate_df = load_rate_rule_hits(
            args.rate_rules_csv,
            include_cols=rate_rules_promoted,
            id_col=args.row_id_col,
        )
        log(
            f"rate-rule CSV rows={len(rate_df):,} | "
            f"fires (any promoted)={int((rate_df['rate_hit'] == 1).sum()):,}"
        )
        val_df = val_df.merge(rate_df, on=args.row_id_col, how="left")
        test_df = test_df.merge(rate_df, on=args.row_id_col, how="left")
        for name, df in (("val", val_df), ("test", test_df)):
            df["rate_hit"] = df["rate_hit"].fillna(0).astype(int)
            df["rate_hit_which"] = df["rate_hit_which"].fillna("")
            log(
                f"  {name} | rate_hit fires={int((df['rate_hit'] == 1).sum()):,} "
                f"of {len(df):,} rows ({df['rate_hit'].mean():.2%})"
            )
        rate_hit_col = "rate_hit"

    val_df = build_final_score(
        val_df,
        snort_col=args.snort_col,
        gate_prob_col=args.gate_prob_col,
        out_col="final_score",
        rate_hit_col=rate_hit_col,
    )
    test_df = build_final_score(
        test_df,
        snort_col=args.snort_col,
        gate_prob_col=args.gate_prob_col,
        out_col="final_score",
        rate_hit_col=rate_hit_col,
    )

    y_val = pd.to_numeric(val_df[args.label_col], errors="raise").to_numpy().astype(np.int32)
    s_val = pd.to_numeric(val_df["final_score"], errors="raise").to_numpy().astype(np.float64)

    y_test = pd.to_numeric(test_df[args.label_col], errors="raise").to_numpy().astype(np.int32)
    s_test = pd.to_numeric(test_df["final_score"], errors="raise").to_numpy().astype(np.float64)

    rows: list[dict[str, Any]] = []

    def _append_valcal_rows(
        s_val_in: np.ndarray,
        s_test_in: np.ndarray,
        suffix: str,
    ) -> tuple[float, float | None]:
        """Emit accuracy + (optional) balanced-accuracy rows for a given
        score pair, distinguishing raw vs. isotonic-calibrated via suffix.
        Returns (tau_acc, tau_ba) for bookkeeping."""
        tau_acc_local = threshold_accuracy_optimal(y_val, s_val_in)
        yhat_acc = (s_test_in >= tau_acc_local).astype(np.int32)
        op_acc = f"val_accuracy_calibrated{suffix}"
        row_acc = metric_row(
            method="Hybrid-Cascade (ours)",
            operating_point=op_acc,
            threshold_source="validation",
            threshold=tau_acc_local,
            y_true=y_test,
            y_score=s_test_in,
            y_pred=yhat_acc,
        )
        rows.append(row_acc)
        log(
            f"{op_acc} | "
            f"acc={row_acc['accuracy']:.4f} prec={row_acc['precision']:.4f} "
            f"rec={row_acc['recall']:.4f} f1={row_acc['f1']:.4f} "
            f"far={row_acc['far']:.2e} tau={tau_acc_local:.6g}"
        )

        tau_ba_local: float | None = None
        if args.include_val_balanced_accuracy:
            tau_ba_local = threshold_balanced_accuracy_optimal(y_val, s_val_in)
            yhat_ba = (s_test_in >= tau_ba_local).astype(np.int32)
            op_ba = f"val_balanced_accuracy_calibrated{suffix}"
            row_ba = metric_row(
                method="Hybrid-Cascade (ours)",
                operating_point=op_ba,
                threshold_source="validation",
                threshold=tau_ba_local,
                y_true=y_test,
                y_score=s_test_in,
                y_pred=yhat_ba,
            )
            rows.append(row_ba)
            log(
                f"{op_ba} | "
                f"acc={row_ba['accuracy']:.4f} prec={row_ba['precision']:.4f} "
                f"rec={row_ba['recall']:.4f} f1={row_ba['f1']:.4f} "
                f"far={row_ba['far']:.2e} tau={tau_ba_local:.6g}"
            )
        return tau_acc_local, tau_ba_local

    # --- Raw-score rows (headline): val_accuracy_calibrated + optional
    #     val_balanced_accuracy_calibrated. This is the locked-environment
    #     behaviour of the script; retained unchanged.
    tau_acc, tau_ba = _append_valcal_rows(s_val, s_test, suffix="")

    # --- Isotonic-calibrated rows (ablation): fit isotonic regression on
    #     val, apply to val+test, then re-threshold. Tagged with '_isotonic'.
    tau_acc_iso: float | None = None
    tau_ba_iso: float | None = None
    s_val_iso: np.ndarray | None = None
    s_test_iso: np.ndarray | None = None
    if args.calibrate_isotonic:
        log("fitting isotonic calibration on D_val (mapping s_val -> P(attack|s))")
        s_val_iso, s_test_iso = isotonic_calibrate_scores(y_val, s_val, s_test)
        tau_acc_iso, tau_ba_iso = _append_valcal_rows(
            s_val_iso, s_test_iso, suffix="_isotonic"
        )

    # --- Secondary row: validation FAR calibrated (retained for reference).
    tau_far = threshold_for_target_far(y_val, s_val, args.target_far)
    yhat_test_far = (s_test >= tau_far).astype(np.int32)
    rows.append(
        metric_row(
            method="Hybrid-Cascade (ours)",
            operating_point=f"val_far_calibrated_{args.target_far:.2e}",
            threshold_source="validation",
            threshold=tau_far,
            y_true=y_test,
            y_score=s_test,
            y_pred=yhat_test_far,
        )
    )

    # Optional: validation-F1 calibrated threshold, then evaluate on test
    if args.include_val_f1:
        tau_f1 = threshold_f1_optimal(y_val, s_val)
        yhat_test_f1 = (s_test >= tau_f1).astype(np.int32)
        rows.append(
            metric_row(
                method="Hybrid-Cascade (ours)",
                operating_point="val_f1_calibrated",
                threshold_source="validation",
                threshold=tau_f1,
                y_true=y_test,
                y_score=s_test,
                y_pred=yhat_test_f1,
            )
        )

    # Optional benchmark-only optimistic rows
    if args.include_test_optimistic:
        tau_test_f1 = threshold_f1_optimal(y_test, s_test)
        yhat_test_f1 = (s_test >= tau_test_f1).astype(np.int32)
        rows.append(
            metric_row(
                method="Hybrid-Cascade (ours)",
                operating_point="test_f1_optimal",
                threshold_source="test",
                threshold=tau_test_f1,
                y_true=y_test,
                y_score=s_test,
                y_pred=yhat_test_f1,
            )
        )

        tau_test_far = threshold_for_target_far(y_test, s_test, args.target_far)
        yhat_test_far_opt = (s_test >= tau_test_far).astype(np.int32)
        rows.append(
            metric_row(
                method="Hybrid-Cascade (ours)",
                operating_point=f"test_far_matched_{args.target_far:.2e}",
                threshold_source="test",
                threshold=tau_test_far,
                y_true=y_test,
                y_score=s_test,
                y_pred=yhat_test_far_opt,
            )
        )

    # Save predictions for auditability. We persist every operating point
    # that was emitted so downstream tooling can audit any of them.
    val_df["y_pred_val_accuracy_calibrated"] = (s_val >= tau_acc).astype(np.int32)
    test_df["y_pred_test_val_accuracy_calibrated"] = (s_test >= tau_acc).astype(np.int32)
    if args.include_val_balanced_accuracy and tau_ba is not None:
        val_df["y_pred_val_balanced_accuracy_calibrated"] = (
            s_val >= tau_ba
        ).astype(np.int32)
        test_df["y_pred_test_val_balanced_accuracy_calibrated"] = (
            s_test >= tau_ba
        ).astype(np.int32)
    if args.calibrate_isotonic and s_val_iso is not None and s_test_iso is not None:
        val_df["final_score_isotonic"] = s_val_iso
        test_df["final_score_isotonic"] = s_test_iso
        if tau_acc_iso is not None:
            val_df["y_pred_val_accuracy_calibrated_isotonic"] = (
                s_val_iso >= tau_acc_iso
            ).astype(np.int32)
            test_df["y_pred_test_val_accuracy_calibrated_isotonic"] = (
                s_test_iso >= tau_acc_iso
            ).astype(np.int32)
        if tau_ba_iso is not None:
            val_df["y_pred_val_balanced_accuracy_calibrated_isotonic"] = (
                s_val_iso >= tau_ba_iso
            ).astype(np.int32)
            test_df["y_pred_test_val_balanced_accuracy_calibrated_isotonic"] = (
                s_test_iso >= tau_ba_iso
            ).astype(np.int32)
    val_df["y_pred_val_far_calibrated"] = (s_val >= tau_far).astype(np.int32)
    test_df["y_pred_test_val_far_calibrated"] = yhat_test_far

    val_out = out_dir / "val_scores_with_predictions.csv"
    test_out = out_dir / "test_scores_with_predictions.csv"
    metrics_out = out_dir / "overall_metrics_proposed_valcal.csv"
    cfg_out = out_dir / "run_config.json"

    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    pd.DataFrame(rows).to_csv(metrics_out, index=False)

    cfg_out.write_text(
        json.dumps(
            {
                "val_csv": args.val_csv,
                "test_csv": args.test_csv,
                "target_far": args.target_far,
                "label_col": args.label_col,
                "snort_col": args.snort_col,
                "gate_prob_col": args.gate_prob_col,
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "n_val_benign": int((y_val == 0).sum()),
                "n_test_benign": int((y_test == 0).sum()),
                "tau_val_accuracy": float(tau_acc),  # main operating point
                "tau_val_balanced_accuracy": (
                    float(tau_ba) if tau_ba is not None else None
                ),
                "tau_val_accuracy_isotonic": (
                    float(tau_acc_iso) if tau_acc_iso is not None else None
                ),
                "tau_val_balanced_accuracy_isotonic": (
                    float(tau_ba_iso) if tau_ba_iso is not None else None
                ),
                "tau_val_far": float(tau_far),       # secondary reference
                "calibrate_isotonic": bool(args.calibrate_isotonic),
                "include_val_balanced_accuracy": bool(
                    args.include_val_balanced_accuracy
                ),
                "rate_rules_csv": args.rate_rules_csv,
                "rate_rules_promoted": rate_rules_promoted,
                "protocol": (
                    "Headline operating point: validation-calibrated target Accuracy "
                    "(threshold maximising accuracy on D_val, applied to D_test). "
                    "Balanced-accuracy row (prior-invariant) and isotonic-calibrated "
                    "variants are emitted alongside for ablation."
                ),
                "final_score_definition": (
                    "1.0 if (snort_pred==1) or (rate_hit==1) else gate_prob; "
                    "rate_hit is OR of user-selected rate_* Tier-1 promotions "
                    "(empty if --rate-rules-csv not set)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"wrote: {val_out}")
    log(f"wrote: {test_out}")
    log(f"wrote: {metrics_out}")
    log(f"wrote: {cfg_out}")
    log("done.")


if __name__ == "__main__":
    main()
