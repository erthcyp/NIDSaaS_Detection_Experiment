#!/usr/bin/env python3
"""
rate_rules_baseline_valcal.py
=============================

Rate-rule-only ablation for the comparison table.

Emits the following methods, all under the locked split (temporal_by_file,
seed 42, 64/16/20) and the headline val-calibrated accuracy operating point:

    Rate Rules (V|S|P)            tier-1 set actually promoted in the cascade
    Rate Rules (all 6, OR)        full union of all six rules
    Rate Rules (count)            sum of fires (0..6); val-cal picks the cut
    Snort + Rate Rules            full signature stack, no ML (rate OR snort)

The point of this script is to answer the reviewer question:

    "What if you ran rate rules alone, with no ML gate and no Snort?"

It reuses the prediction CSVs already emitted by the cascade run for labels
and the Snort fast-path column, and the per-row rate_* fires from the
signature_merged_predictions.csv that the cascade consumes. No retraining
is required, the rate rules are deterministic functions of the cleaned
flow features.

Inputs
------
--val-csv      outputs_hybrid_cascade_splitcal_fastsnort_temporal/val_cascade_predictions.csv
--test-csv     outputs_hybrid_cascade_splitcal_fastsnort_temporal/test_cascade_predictions.csv
--rate-csv     signature_merged_predictions.csv        # has rate_V..rate_B and rate_X (= snort)

Outputs
-------
<out_dir>/overall_metrics_rate_rules_valcal.csv
<out_dir>/rate_rules_table_fragment.tex
<out_dir>/run_config.json

The CSV/LaTeX schema matches rf_baseline_valcal.py exactly so the rows can
be concatenated straight into the master comparison table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    print(f"[rate-rules-valcal] {msg}", flush=True)


# -------------------------------------------------------------------------
# shared helpers (mirror rf_baseline_valcal.py exactly)
# -------------------------------------------------------------------------

def _far_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    return float(fp) / float(denom) if denom else 0.0


def threshold_accuracy_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold t maximising accuracy of (y_score >= t).

    Same achievable-cut search as rf_baseline_valcal.threshold_accuracy_optimal.
    Tie-break prefers the tightest cut.
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
    """Threshold t maximising BALANCED accuracy = (TPR + TNR)/2."""
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
# LaTeX fragment (same shape as rf_baseline_valcal.emit_latex_fragment)
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
    fallback_order = [
        headline_operating_point,
        "val_balanced_accuracy_calibrated",
        "val_accuracy_calibrated",
    ]
    seen: set[str] = set()
    fallback_order = [op for op in fallback_order if not (op in seen or seen.add(op))]
    lines = [
        "% Auto-generated by rate_rules_baseline_valcal.py.",
        f"% Headline operating point: {headline_operating_point}.",
        "% Threshold picked on D_val, applied unchanged to D_test.",
        "% Columns: Method | Acc | Prec | Rec | F1 | FAR | ROC-AUC | PR-AUC.",
        "% Rows sorted by accuracy (descending).",
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
# scoring + reporting
# -------------------------------------------------------------------------

RATE_LETTERS = ("V", "L", "S", "R", "P", "B")
RATE_COLS = tuple(f"rate_{c}" for c in RATE_LETTERS)


def _score_and_report(
    method: str,
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    include_balanced_accuracy: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    tau_acc = threshold_accuracy_optimal(y_val, s_val)
    yhat_acc = (s_test >= tau_acc).astype(np.int32)
    row_acc = metric_row(
        method=method,
        operating_point="val_accuracy_calibrated",
        threshold_source="validation",
        threshold=tau_acc,
        y_true=y_test,
        y_score=s_test,
        y_pred=yhat_acc,
    )
    rows.append(row_acc)
    log(
        f"{method} @ val_accuracy_calibrated | tau={tau_acc:.6g} "
        f"acc={row_acc['accuracy']:.4f} prec={row_acc['precision']:.4f} "
        f"rec={row_acc['recall']:.4f} f1={row_acc['f1']:.4f} "
        f"far={row_acc['far']:.2e} roc={row_acc['roc_auc']:.3f} "
        f"pr={row_acc['pr_auc']:.3f}"
    )

    if include_balanced_accuracy:
        tau_ba = threshold_balanced_accuracy_optimal(y_val, s_val)
        yhat_ba = (s_test >= tau_ba).astype(np.int32)
        row_ba = metric_row(
            method=method,
            operating_point="val_balanced_accuracy_calibrated",
            threshold_source="validation",
            threshold=tau_ba,
            y_true=y_test,
            y_score=s_test,
            y_pred=yhat_ba,
        )
        rows.append(row_ba)
        log(
            f"{method} @ val_balanced_accuracy_calibrated | tau={tau_ba:.6g} "
            f"acc={row_ba['accuracy']:.4f} prec={row_ba['precision']:.4f} "
            f"rec={row_ba['recall']:.4f} f1={row_ba['f1']:.4f} "
            f"far={row_ba['far']:.2e} roc={row_ba['roc_auc']:.3f} "
            f"pr={row_ba['pr_auc']:.3f}"
        )

    return rows


def per_class_report(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    if "multiclass_label" not in test_df.columns:
        return pd.DataFrame()
    out = []
    for cls, sub in test_df.groupby("multiclass_label"):
        idx = sub.index.to_numpy()
        support = len(idx)
        fires = int(y_pred[idx].sum())
        metric = "Correct Benign Rate" if cls == "BENIGN" else "Detection Rate"
        rate = 1.0 - fires / support if cls == "BENIGN" else fires / support
        out.append(
            {
                "class": cls,
                "support": support,
                "metric": metric,
                "rate": round(rate, 4),
                "fires": fires,
            }
        )
    return (
        pd.DataFrame(out)
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def _load_rate_columns(rate_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(rate_csv)
    needed = ["row_id", *RATE_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"rate-csv missing columns {missing}. Expected at least: {needed}"
        )
    keep = ["row_id", *RATE_COLS]
    if "rate_X" in df.columns:
        keep.append("rate_X")  # snort fires, optional
    out = df[keep].copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(np.int64)
    for c in RATE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.int8)
    if "rate_X" in out.columns:
        out["rate_X"] = pd.to_numeric(out["rate_X"], errors="coerce").fillna(0).astype(np.int8)
    return out


def _attach_rates(
    cascade_df: pd.DataFrame, rate_df: pd.DataFrame, name: str
) -> pd.DataFrame:
    merged = cascade_df.merge(rate_df, on="row_id", how="left")
    n_missing = int(merged[RATE_COLS[0]].isna().sum())
    if n_missing:
        log(
            f"{name}: {n_missing:,} rows had no rate-rule match "
            f"(treating as no fire)"
        )
    for c in RATE_COLS:
        merged[c] = merged[c].fillna(0).astype(np.int8)
    if "rate_X" in merged.columns:
        merged["rate_X"] = merged["rate_X"].fillna(0).astype(np.int8)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-csv",
        default="outputs_hybrid_cascade_splitcal_fastsnort_temporal/val_cascade_predictions.csv",
        help="Cascade val CSV (gives row_id, binary_label, snort_pred).",
    )
    parser.add_argument(
        "--test-csv",
        default="outputs_hybrid_cascade_splitcal_fastsnort_temporal/test_cascade_predictions.csv",
    )
    parser.add_argument(
        "--rate-csv",
        default="signature_merged_predictions.csv",
        help="Rate-rule predictions CSV (row_id, rate_V..rate_B, optional rate_X).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_rate_rules_baseline_valcal",
    )
    parser.add_argument("--label-col", default="binary_label")
    parser.add_argument(
        "--snort-pred-col",
        default="snort_pred",
        help="Column in val/test CSVs holding the Snort fast-path prediction. "
        "Used only for the 'Snort + Rate Rules' row.",
    )
    parser.add_argument(
        "--skip-snort-row",
        action="store_true",
        help="Do not emit the 'Snort + Rate Rules (signature only)' row.",
    )
    parser.add_argument(
        "--skip-val-balanced-accuracy",
        dest="include_val_balanced_accuracy",
        action="store_false",
    )
    parser.set_defaults(include_val_balanced_accuracy=True)
    parser.add_argument(
        "--headline-operating-point",
        default="val_accuracy_calibrated",
        choices=[
            "val_accuracy_calibrated",
            "val_balanced_accuracy_calibrated",
        ],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading val CSV : {args.val_csv}")
    val_df = pd.read_csv(args.val_csv, low_memory=False)
    log(f"loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    log(f"loading rate-rule CSV: {args.rate_csv}")
    rate_df = _load_rate_columns(Path(args.rate_csv))

    for df, name in ((val_df, "val"), (test_df, "test")):
        if "row_id" not in df.columns:
            raise ValueError(f"{name} CSV missing row_id column")
        if args.label_col not in df.columns:
            raise ValueError(f"{name} CSV missing {args.label_col} column")
        df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(np.int64)

    val_df = _attach_rates(val_df, rate_df, "val")
    test_df = _attach_rates(test_df, rate_df, "test")

    y_val = pd.to_numeric(val_df[args.label_col], errors="raise").to_numpy().astype(np.int32)
    y_test = pd.to_numeric(test_df[args.label_col], errors="raise").to_numpy().astype(np.int32)

    log(
        f"splits | val={len(y_val):,} (benign {(y_val == 0).sum():,}) "
        f"test={len(y_test):,} (benign {(y_test == 0).sum():,})"
    )

    rows: list[dict[str, Any]] = []
    per_class_dump: dict[str, pd.DataFrame] = {}

    # ---- Method 1: tier-1 promotion set only (V|S|P) ------------------
    s_val_vsp = (
        (val_df["rate_V"].to_numpy() == 1)
        | (val_df["rate_S"].to_numpy() == 1)
        | (val_df["rate_P"].to_numpy() == 1)
    ).astype(np.float64)
    s_test_vsp = (
        (test_df["rate_V"].to_numpy() == 1)
        | (test_df["rate_S"].to_numpy() == 1)
        | (test_df["rate_P"].to_numpy() == 1)
    ).astype(np.float64)
    rows.extend(
        _score_and_report(
            method="Rate Rules (V|S|P)",
            y_val=y_val,
            s_val=s_val_vsp,
            y_test=y_test,
            s_test=s_test_vsp,
            include_balanced_accuracy=args.include_val_balanced_accuracy,
        )
    )
    tau_vsp = threshold_accuracy_optimal(y_val, s_val_vsp)
    per_class_dump["rate_VSP"] = per_class_report(
        test_df, (s_test_vsp >= tau_vsp).astype(np.int32)
    )

    # ---- Method 2: full union of all 6 rules --------------------------
    rate_vec_val = val_df[list(RATE_COLS)].to_numpy(dtype=np.int8)
    rate_vec_test = test_df[list(RATE_COLS)].to_numpy(dtype=np.int8)
    s_val_or = (rate_vec_val.sum(axis=1) > 0).astype(np.float64)
    s_test_or = (rate_vec_test.sum(axis=1) > 0).astype(np.float64)
    rows.extend(
        _score_and_report(
            method="Rate Rules (all 6, OR)",
            y_val=y_val,
            s_val=s_val_or,
            y_test=y_test,
            s_test=s_test_or,
            include_balanced_accuracy=args.include_val_balanced_accuracy,
        )
    )
    tau_or = threshold_accuracy_optimal(y_val, s_val_or)
    per_class_dump["rate_OR"] = per_class_report(
        test_df, (s_test_or >= tau_or).astype(np.int32)
    )

    # ---- Method 3: count of fires (0..6), val-cal picks the cut --------
    s_val_cnt = rate_vec_val.sum(axis=1).astype(np.float64)
    s_test_cnt = rate_vec_test.sum(axis=1).astype(np.float64)
    rows.extend(
        _score_and_report(
            method="Rate Rules (count)",
            y_val=y_val,
            s_val=s_val_cnt,
            y_test=y_test,
            s_test=s_test_cnt,
            include_balanced_accuracy=args.include_val_balanced_accuracy,
        )
    )
    tau_cnt = threshold_accuracy_optimal(y_val, s_val_cnt)
    per_class_dump["rate_count"] = per_class_report(
        test_df, (s_test_cnt >= tau_cnt).astype(np.int32)
    )

    # ---- Method 4: Snort + Rate Rules (no ML) -------------------------
    if not args.skip_snort_row:
        if args.snort_pred_col not in val_df.columns or args.snort_pred_col not in test_df.columns:
            log(
                f"snort_pred col '{args.snort_pred_col}' missing -- skipping "
                f"'Snort + Rate Rules' row"
            )
        else:
            snort_val = pd.to_numeric(
                val_df[args.snort_pred_col], errors="coerce"
            ).fillna(0).to_numpy().astype(np.int8)
            snort_test = pd.to_numeric(
                test_df[args.snort_pred_col], errors="coerce"
            ).fillna(0).to_numpy().astype(np.int8)
            s_val_sigall = (
                (rate_vec_val.sum(axis=1) > 0) | (snort_val == 1)
            ).astype(np.float64)
            s_test_sigall = (
                (rate_vec_test.sum(axis=1) > 0) | (snort_test == 1)
            ).astype(np.float64)
            rows.extend(
                _score_and_report(
                    method="Snort + Rate Rules",
                    y_val=y_val,
                    s_val=s_val_sigall,
                    y_test=y_test,
                    s_test=s_test_sigall,
                    include_balanced_accuracy=args.include_val_balanced_accuracy,
                )
            )
            tau_sig = threshold_accuracy_optimal(y_val, s_val_sigall)
            per_class_dump["snort_plus_rates"] = per_class_report(
                test_df, (s_test_sigall >= tau_sig).astype(np.int32)
            )

    # --- persist outputs ---
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "overall_metrics_rate_rules_valcal.csv"
    df_out.to_csv(csv_path, index=False)
    log(f"wrote metrics: {csv_path}")

    tex_path = out_dir / "rate_rules_table_fragment.tex"
    emit_latex_fragment(
        rows,
        tex_path,
        headline_operating_point=args.headline_operating_point,
    )

    # per-class dumps (one CSV per method) for the failure-mode discussion
    for tag, frame in per_class_dump.items():
        if frame.empty:
            continue
        path = out_dir / f"per_class_{tag}.csv"
        frame.to_csv(path, index=False)
        log(f"wrote per-class report: {path}")

    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "val_csv": args.val_csv,
                "test_csv": args.test_csv,
                "rate_csv": args.rate_csv,
                "label_col": args.label_col,
                "snort_pred_col": args.snort_pred_col,
                "skip_snort_row": bool(args.skip_snort_row),
                "include_val_balanced_accuracy": bool(
                    args.include_val_balanced_accuracy
                ),
                "headline_operating_point": args.headline_operating_point,
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "n_val_benign": int((y_val == 0).sum()),
                "n_test_benign": int((y_test == 0).sum()),
                "rate_letters": list(RATE_LETTERS),
                "protocol": (
                    "Rate-rule-only ablation. Three scoring variants of the "
                    "deterministic rate-rule engine plus a 'signature-only' "
                    "row (rate OR snort). Threshold picked on D_val to "
                    "maximise accuracy, applied unchanged to D_test. Rate "
                    "rules are deterministic functions of the cleaned flow "
                    "features; no model retraining."
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
