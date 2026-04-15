from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def log(msg: str) -> None:
    print(f"[hybrid_assisted_from_snort_rf] {msg}", flush=True)


def resolve_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def to_binary_pred(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0.5).astype(int)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
                "attack": 1,
                "benign": 0,
                "malicious": 1,
                "normal": 0,
                "yes": 1,
                "no": 0,
            }
        )
    )
    if mapped.notna().all():
        return mapped.astype(int)
    raise ValueError("Could not convert prediction column to binary 0/1.")


def to_score(series: Optional[pd.Series], pred_series: pd.Series) -> np.ndarray:
    if series is None:
        return pred_series.astype(float).to_numpy()
    return pd.to_numeric(series, errors="coerce").fillna(pred_series.astype(float)).to_numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    far = float(fp / max(1, (fp + tn)))

    out = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "far": far,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if scores is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    return out


def load_and_prepare_rf_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log(f"rf file loaded | rows={len(df):,} | path={path}")

    row_id_col = resolve_first_existing(df, ["row_id"])
    label_col = resolve_first_existing(df, ["binary_label"])
    pred_col = resolve_first_existing(
        df,
        ["rf_pred", "prediction", "pred", "y_pred", "rf_prediction", "anomaly_pred", "final_pred"],
    )
    if pred_col is None:
        raise ValueError("Could not find RF prediction column.")

    score_col = resolve_first_existing(
        df,
        ["rf_score", "score", "pred_score", "anomaly_score", "y_score", "prob_attack", "probability"],
    )
    if score_col is None:
        raise ValueError("RF predictions file must contain an RF score column for assisted hybrid.")

    out = df.copy()
    out["_rf_pred"] = to_binary_pred(out[pred_col])
    out["_rf_score"] = to_score(out[score_col], out["_rf_pred"])

    if row_id_col is not None:
        out["_row_id"] = pd.to_numeric(out[row_id_col], errors="coerce")
    if label_col is not None:
        out["_binary_label"] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    return out


def load_and_prepare_snort_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log(f"snort file loaded | rows={len(df):,} | path={path}")

    row_id_col = resolve_first_existing(df, ["row_id"])
    label_col = resolve_first_existing(df, ["binary_label"])
    pred_col = resolve_first_existing(df, ["signature_pred", "snort_pred", "prediction", "pred"])
    if pred_col is None:
        raise ValueError("Could not find Snort prediction column.")

    score_col = resolve_first_existing(df, ["signature_score", "snort_score", "score", "pred_score"])

    out = df.copy()
    out["_snort_pred"] = to_binary_pred(out[pred_col])
    out["_snort_score"] = to_score(out[score_col] if score_col else None, out["_snort_pred"])

    if row_id_col is not None:
        out["_row_id"] = pd.to_numeric(out[row_id_col], errors="coerce")
    if label_col is not None:
        out["_binary_label"] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    return out


def merge_predictions(rf_df: pd.DataFrame, snort_df: pd.DataFrame) -> pd.DataFrame:
    if "_row_id" in rf_df.columns and "_row_id" in snort_df.columns:
        merged = rf_df.merge(
            snort_df[["_row_id", "_snort_pred", "_snort_score"]],
            on="_row_id",
            how="inner",
        )
        method = "row_id"
    elif len(rf_df) == len(snort_df):
        merged = pd.concat(
            [rf_df.reset_index(drop=True), snort_df[["_snort_pred", "_snort_score"]].reset_index(drop=True)],
            axis=1,
        )
        method = "positional_index"
    else:
        raise ValueError("Could not merge RF and Snort predictions. Need shared row_id in both files or identical row counts.")

    log(f"merged predictions | rows={len(merged):,} | method={method}")
    return merged


def assisted_predict(
    rf_score: np.ndarray,
    snort_pred: np.ndarray,
    base_threshold: float,
    assist_threshold: float,
) -> np.ndarray:
    return (
        (rf_score > base_threshold)
        | ((snort_pred == 1) & (rf_score > assist_threshold))
    ).astype(int)


def grid_search_assisted(
    y_true: np.ndarray,
    rf_score: np.ndarray,
    rf_pred: np.ndarray,
    snort_pred: np.ndarray,
    far_ceiling: Optional[float],
    base_values: np.ndarray,
    assist_values: np.ndarray,
) -> Dict[str, float]:
    rf_metrics = compute_metrics(y_true, rf_pred, rf_score)

    best = None
    for base_thr in base_values:
        for assist_thr in assist_values:
            if assist_thr > base_thr:
                continue

            y_pred = assisted_predict(rf_score, snort_pred, float(base_thr), float(assist_thr))
            metrics = compute_metrics(y_true, y_pred, rf_score)

            if far_ceiling is not None and metrics["far"] > far_ceiling:
                continue

            candidate = {"base_threshold": float(base_thr), "assist_threshold": float(assist_thr), **metrics}

            if best is None:
                best = candidate
                continue

            key_best = (best["f1"], best["recall"], -best["far"], best["precision"])
            key_new = (candidate["f1"], candidate["recall"], -candidate["far"], candidate["precision"])
            if key_new > key_best:
                best = candidate

    if best is None:
        best = {"base_threshold": 0.5, "assist_threshold": 0.5, **rf_metrics}

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine RF predictions with Snort signature predictions using assisted hybrid gating.")
    parser.add_argument("--rf-predictions", required=True)
    parser.add_argument("--snort-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-model-name", default="Hybrid-Assisted-Snort+RF")
    parser.add_argument("--base-threshold", type=float, default=None)
    parser.add_argument("--assist-threshold", type=float, default=None)
    parser.add_argument("--search", action="store_true", default=False)
    parser.add_argument("--far-ceiling", type=float, default=None)
    parser.add_argument("--base-grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--assist-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_df = load_and_prepare_rf_predictions(args.rf_predictions)
    snort_df = load_and_prepare_snort_predictions(args.snort_predictions)
    merged = merge_predictions(rf_df, snort_df)

    if "_binary_label" not in merged.columns:
        raise ValueError("Merged dataframe does not contain binary_label. Need labels to compute metrics.")

    y_true = merged["_binary_label"].astype(int).to_numpy()
    rf_pred = merged["_rf_pred"].astype(int).to_numpy()
    rf_score = merged["_rf_score"].astype(float).to_numpy()
    snort_pred = merged["_snort_pred"].astype(int).to_numpy()
    snort_score = merged["_snort_score"].astype(float).to_numpy()

    rf_metrics = compute_metrics(y_true, rf_pred, rf_score)
    snort_metrics = compute_metrics(y_true, snort_pred, snort_score)

    if args.search:
        base_values = np.array([float(x) for x in args.base_grid.split(",") if x.strip()], dtype=float)
        assist_values = np.array([float(x) for x in args.assist_grid.split(",") if x.strip()], dtype=float)
        best = grid_search_assisted(
            y_true=y_true,
            rf_score=rf_score,
            rf_pred=rf_pred,
            snort_pred=snort_pred,
            far_ceiling=args.far_ceiling,
            base_values=base_values,
            assist_values=assist_values,
        )
        base_thr = float(best["base_threshold"])
        assist_thr = float(best["assist_threshold"])
        log(
            "best thresholds from search | "
            f"base_threshold={base_thr:.4f}, assist_threshold={assist_thr:.4f}, "
            f"f1={best['f1']:.6f}, recall={best['recall']:.6f}, far={best['far']:.6f}"
        )
    else:
        base_thr = 0.5 if args.base_threshold is None else float(args.base_threshold)
        assist_thr = base_thr if args.assist_threshold is None else float(args.assist_threshold)
        if assist_thr > base_thr:
            raise ValueError("assist_threshold must be <= base_threshold")

    merged["assisted_pred"] = assisted_predict(rf_score, snort_pred, base_thr, assist_thr)
    merged["assisted_score"] = rf_score

    assisted_metrics = compute_metrics(
        y_true,
        merged["assisted_pred"].astype(int).to_numpy(),
        merged["assisted_score"].astype(float).to_numpy(),
    )

    comparison = pd.DataFrame(
        [
            {"paper_model": "RF", "model": "rf", **rf_metrics},
            {"paper_model": "Signature-Snort", "model": "snort_signature", **snort_metrics},
            {
                "paper_model": args.paper_model_name,
                "model": "hybrid_assisted",
                **assisted_metrics,
                "base_threshold": base_thr,
                "assist_threshold": assist_thr,
            },
        ]
    )

    comparison_path = out_dir / "assisted_hybrid_metrics_comparison.csv"
    pred_path = out_dir / "assisted_hybrid_predictions.csv"

    comparison.to_csv(comparison_path, index=False)
    merged.to_csv(pred_path, index=False)

    log(f"saved metrics: {comparison_path}")
    log(f"saved predictions: {pred_path}")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
