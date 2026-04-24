"""Hybrid Cascade split-calibration with FastSnort naming.

Signature layer is two-tiered:
  - Tier 1 (fast-path, short-circuit to alert): signature_pred column, which
    is the OR of the high-precision rate rules {V, S, R, B} and Snort.
  - Tier 2 (gate meta-features): rate_L and rate_P columns, forwarded to the
    escalation gate alongside rf_score and rf_pvalue. These rules carry
    signal but are too loose to short-circuit on their own (especially P,
    which fires on ~4% of benigns due to DNS/load-balancer traffic).

The escalation gate therefore sees:
    raw flow features, rf_score, rf_pvalue, rate_L, rate_P

Only rows with rf_pvalue <= alpha_escalate are passed to the gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import RFConfig
from conformal_wrapper import ConformalAnomalyWrapper, ConformalConfig
from escalation_gate_fastsnort import EscalationGateFastSnort, EscalationGateFastSnortConfig
from load_data import load_and_prepare_detection_data
from metrics import binary_metrics
from rf_anomaly import SelfSupervisedRFAnomaly
from utils import set_random_seed, write_json


def log(msg: str) -> None:
    print(f"[hybrid_cascade_splitcal_fastsnort] {msg}", flush=True)


def _resolve_first(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lut = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lut:
            return lut[c.lower()]
    return None


TIER2_META_COLUMNS = ("rate_V", "rate_L", "rate_S", "rate_R", "rate_P", "rate_B")


def load_signature_table(path: str) -> pd.DataFrame:
    """Load the merged signature predictions CSV. Always returns the fast-path
    column (signature_pred / signature_score) and, when present, the tier-2
    rate-rule indicator columns (rate_V, rate_L, rate_S, rate_R, rate_P,
    rate_B) that are later forwarded to the escalation gate as meta-features.
    """
    df = pd.read_csv(path)
    row_id_col = _resolve_first(df, ["row_id"])
    pred_col = _resolve_first(df, ["signature_pred", "snort_pred", "prediction", "pred"])
    score_col = _resolve_first(df, ["signature_score", "snort_score", "score", "pred_score"])

    if row_id_col is None:
        raise ValueError("Signature predictions CSV must contain row_id.")
    if pred_col is None:
        raise ValueError("Signature predictions CSV must contain signature/snort prediction column.")

    out = pd.DataFrame()
    out["row_id"] = pd.to_numeric(df[row_id_col], errors="coerce")
    out["snort_pred"] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)
    if score_col is not None:
        out["snort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)
    else:
        out["snort_score"] = out["snort_pred"].astype(float)

    # Optional tier-2 meta columns. When the CSV was produced by the old
    # pre-tier schema these will simply be missing; downstream merges fill
    # with zeros so the gate has the same feature count but receives a
    # constant signal.
    for c in TIER2_META_COLUMNS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=["row_id"]).copy()
    out["row_id"] = out["row_id"].astype(np.int64)
    tier2_present = [c for c in TIER2_META_COLUMNS if c in out.columns]
    log(
        f"signature table loaded | rows={len(out):,} | "
        f"fast-path fires={(out['snort_pred'] == 1).sum():,} | "
        f"tier-2 columns present: {tier2_present}"
    )
    return out


# Back-compat alias so anything importing the old name keeps working.
load_snort_table = load_signature_table


def merge_signature(base_df: pd.DataFrame, signature_table: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, dict]:
    if "row_id" not in base_df.columns:
        raise ValueError("Base dataframe must contain row_id for signature merge.")

    merged = base_df.merge(signature_table, on="row_id", how="left", indicator=True)
    matched = int((merged["_merge"] == "both").sum())
    coverage = float(matched / max(1, len(merged)))

    merged["snort_pred"] = merged["snort_pred"].fillna(0).astype(int)
    merged["snort_score"] = merged["snort_score"].fillna(0.0).astype(float)
    for c in TIER2_META_COLUMNS:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype(int)
    merged = merged.drop(columns=["_merge"])

    log(f"signature coverage [{tag}] | matched={matched:,}/{len(merged):,} ({coverage:.1%})")
    return merged, {"matched_rows": matched, "coverage_frac": coverage}


# Back-compat alias
merge_snort = merge_signature


def split_val_for_conformal(
    val_all: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "binary_label" not in val_all.columns:
        raise ValueError("val_all must contain binary_label.")
    if "row_id" not in val_all.columns:
        raise ValueError("val_all must contain row_id.")

    val_benign = val_all.loc[val_all["binary_label"] == 0].copy()
    if len(val_benign) < 100:
        raise ValueError("Validation benign pool is too small for split calibration.")

    frac = float(calibration_fraction)
    if not (0.05 <= frac <= 0.95):
        raise ValueError("calibration_fraction must be between 0.05 and 0.95.")

    n_cal = int(round(len(val_benign) * frac))
    n_cal = max(50, min(n_cal, len(val_benign) - 50))

    cal_benign = val_benign.sample(n=n_cal, random_state=seed).reset_index(drop=True)
    cal_ids = set(cal_benign["row_id"].astype(np.int64).tolist())
    gate_val = val_all.loc[~val_all["row_id"].isin(cal_ids)].copy().reset_index(drop=True)

    log(
        "validation split for conformal | "
        f"val_all={len(val_all):,}, "
        f"cal_benign={len(cal_benign):,}, "
        f"gate_val={len(gate_val):,}, "
        f"gate_val_benign={(gate_val['binary_label'] == 0).sum():,}, "
        f"gate_val_attack={(gate_val['binary_label'] == 1).sum():,}"
    )
    return cal_benign, gate_val


def cascade_predict(
    rf_score: np.ndarray,
    rf_pvalue: np.ndarray,
    snort_pred: np.ndarray,
    gate_prob: np.ndarray,
    alpha_escalate: float,
    gate_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    escalated = rf_pvalue <= alpha_escalate
    gate_attack = (gate_prob >= gate_threshold) & escalated
    final = ((snort_pred == 1) | gate_attack).astype(int)

    score = rf_score.astype(float).copy()
    score = np.where(escalated, np.maximum(score, gate_prob), score)
    score = np.where(snort_pred == 1, np.maximum(score, 1.0), score)
    return final, score

def _to_numpy_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D after ravel, got shape={arr.shape}")
    return arr


def _build_prediction_export(
    base_df: pd.DataFrame,
    split_name: str,
    rf_score,
    rf_pvalue,
    snort_pred,
    snort_score,
    gate_prob,
    escalated,
    cascade_pred,
    cascade_score,
) -> pd.DataFrame:
    out = base_df.copy()

    n = len(out)
    cols = {
        "rf_score": _to_numpy_1d(rf_score, "rf_score"),
        "rf_pvalue": _to_numpy_1d(rf_pvalue, "rf_pvalue"),
        "snort_pred": _to_numpy_1d(snort_pred, "snort_pred"),
        "snort_score": _to_numpy_1d(snort_score, "snort_score"),
        "gate_prob": _to_numpy_1d(gate_prob, "gate_prob"),
        "escalated": _to_numpy_1d(escalated, "escalated"),
        "cascade_pred": _to_numpy_1d(cascade_pred, "cascade_pred"),
        "cascade_score": _to_numpy_1d(cascade_score, "cascade_score"),
    }

    for k, v in cols.items():
        if len(v) != n:
            raise ValueError(
                f"{split_name}: column {k!r} has length {len(v):,}, expected {n:,}"
            )
        out[k] = v

    out["split"] = split_name
    return out


def export_cascade_split_predictions(
    out_dir,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_rf_score,
    val_rf_pvalue,
    val_snort_pred,
    val_snort_score,
    val_gate_prob,
    val_escalated,
    val_cascade_pred,
    val_cascade_score,
    test_rf_score,
    test_rf_pvalue,
    test_snort_pred,
    test_snort_score,
    test_gate_prob,
    test_escalated,
    test_cascade_pred,
    test_cascade_score,
):
    """
    Export validation and test prediction tables for downstream
    validation-calibrated thresholding of the proposed method.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_out = _build_prediction_export(
        base_df=val_df,
        split_name="validation",
        rf_score=val_rf_score,
        rf_pvalue=val_rf_pvalue,
        snort_pred=val_snort_pred,
        snort_score=val_snort_score,
        gate_prob=val_gate_prob,
        escalated=val_escalated,
        cascade_pred=val_cascade_pred,
        cascade_score=val_cascade_score,
    )

    test_out = _build_prediction_export(
        base_df=test_df,
        split_name="test",
        rf_score=test_rf_score,
        rf_pvalue=test_rf_pvalue,
        snort_pred=test_snort_pred,
        snort_score=test_snort_score,
        gate_prob=test_gate_prob,
        escalated=test_escalated,
        cascade_pred=test_cascade_pred,
        cascade_score=test_cascade_score,
    )

    val_path = out_dir / "val_cascade_predictions.csv"
    test_path = out_dir / "test_cascade_predictions.csv"

    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"[cascade-export] wrote: {val_path}", flush=True)
    print(f"[cascade-export] wrote: {test_path}", flush=True)
    return val_path, test_path

def run_cascade(
    data_dir: str,
    snort_predictions_path: str,
    output_dir: str,
    rf_model_path: Optional[str] = None,
    alpha_conformal: float = 0.05,
    alpha_escalate: float = 0.20,
    gate_threshold: float = 0.5,
    split_strategy: str = "random",
    seed: int = 42,
    gate_max_iter: int = 300,
    calibration_fraction: float = 0.50,
    paper_model_name: str = "Hybrid-Cascade-SplitCal-FastSnort",
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)

    log(f"loading data (strategy={split_strategy}) from {data_dir}")
    cleaned, splits = load_and_prepare_detection_data(
        data_dir,
        random_state=seed,
        split_strategy=split_strategy,
    )

    cal_benign_df, gate_val_df = split_val_for_conformal(
        splits.val_all, calibration_fraction=calibration_fraction, seed=seed
    )

    if rf_model_path and Path(rf_model_path).exists():
        log(f"loading RF model from {rf_model_path}")
        rf_model = SelfSupervisedRFAnomaly.load(rf_model_path)
    else:
        log("training fresh RF model (rf-model not provided or missing)")
        rf_model = SelfSupervisedRFAnomaly(RFConfig()).fit(
            splits.train_benign,
            splits.val_benign,
            random_state=seed,
        )
        rf_model.save(out / "rf_anomaly.joblib")

    log("scoring calibration benign subset ...")
    cal_scores = rf_model.score_samples(cal_benign_df)
    log("scoring gate-train validation pool ...")
    gate_val_scores = rf_model.score_samples(gate_val_df)
    log("scoring test set ...")
    test_scores = rf_model.score_samples(splits.test_all)

    conformal = ConformalAnomalyWrapper(
        ConformalConfig(alpha=alpha_conformal, seed=seed)
    ).fit(cal_scores)
    conformal.save(out / "conformal_wrapper.joblib")

    gate_val_pvals = conformal.pvalue(gate_val_scores)
    test_pvals = conformal.pvalue(test_scores)

    signature_table = load_signature_table(snort_predictions_path)
    gate_val_merged, gate_val_sig_cov = merge_signature(gate_val_df, signature_table, tag="gate_val")
    test_merged, test_snort_cov = merge_signature(splits.test_all, signature_table, tag="test")
    test_snort_pred = test_merged["snort_pred"].to_numpy()
    test_snort_score = test_merged["snort_score"].to_numpy()

    # Tier-2 rate-rule columns that were carried through the merge, used as
    # extra meta-features for the escalation gate when present.
    gate_meta_cols_extra = [c for c in TIER2_META_COLUMNS if c in gate_val_merged.columns]
    test_meta_cols_extra = [c for c in TIER2_META_COLUMNS if c in test_merged.columns]
    log(f"gate-val tier-2 meta columns: {gate_meta_cols_extra}")
    log(f"test     tier-2 meta columns: {test_meta_cols_extra}")

    gate_val_meta_dict = {
        "rf_score": gate_val_scores,
        "rf_pvalue": gate_val_pvals,
    }
    for c in gate_meta_cols_extra:
        gate_val_meta_dict[c] = gate_val_merged[c].to_numpy()
    gate_val_meta = pd.DataFrame(gate_val_meta_dict)
    escalation_mask_val = gate_val_pvals <= alpha_escalate
    n_esc = int(escalation_mask_val.sum())
    n_esc_pos = int(gate_val_df.loc[escalation_mask_val, "binary_label"].sum())

    log(
        f"gate-train escalation pool | size={n_esc:,} ({n_esc / len(gate_val_pvals):.1%}) "
        f"| attacks={n_esc_pos:,} | alpha_escalate={alpha_escalate}"
    )

    if n_esc < 200 or n_esc_pos < 50 or (n_esc - n_esc_pos) < 50:
        raise RuntimeError(
            "Escalation pool too small or single-class after split calibration; "
            "increase alpha_escalate or lower calibration_fraction."
        )

    gate = EscalationGateFastSnort(
        EscalationGateFastSnortConfig(
            max_iter=gate_max_iter,
            threshold=gate_threshold,
            random_state=seed,
        )
    ).fit(
        df=gate_val_df.loc[escalation_mask_val].reset_index(drop=True),
        meta=gate_val_meta.loc[escalation_mask_val].reset_index(drop=True),
        y=gate_val_df.loc[escalation_mask_val, "binary_label"].to_numpy(),
        feature_columns=rf_model.feature_columns,
        preprocessor=rf_model.preprocessor,
    )
    gate.save(out / "escalation_gate_fastsnort.joblib")

    test_meta_dict = {
        "rf_score": test_scores,
        "rf_pvalue": test_pvals,
    }
    for c in test_meta_cols_extra:
        test_meta_dict[c] = test_merged[c].to_numpy()
    test_meta = pd.DataFrame(test_meta_dict)
    escalation_mask_test = test_pvals <= alpha_escalate
    n_test_esc = int(escalation_mask_test.sum())
    log(f"test escalation pool | size={n_test_esc:,} ({n_test_esc / len(test_pvals):.1%})")

    gate_probs_test = np.zeros(len(test_pvals), dtype=float)
    if n_test_esc > 0:
        sub_df = splits.test_all.loc[escalation_mask_test].reset_index(drop=True)
        sub_meta = test_meta.loc[escalation_mask_test].reset_index(drop=True)
        gate_probs_test[escalation_mask_test] = gate.predict_proba(sub_df, sub_meta)

    final_pred, cascade_score = cascade_predict(
        rf_score=test_scores,
        rf_pvalue=test_pvals,
        snort_pred=test_snort_pred,
        gate_prob=gate_probs_test,
        alpha_escalate=alpha_escalate,
        gate_threshold=gate_threshold,
    )
    # ------------------------------------------------------------
    # Export split-specific prediction tables for proposed val-cal
    # ------------------------------------------------------------
    val_snort_pred = gate_val_merged["snort_pred"].to_numpy()
    val_snort_score = gate_val_merged["snort_score"].to_numpy()

    gate_probs_val = np.zeros(len(gate_val_pvals), dtype=float)
    if n_esc > 0:
        sub_val_df = gate_val_df.loc[escalation_mask_val].reset_index(drop=True)
        sub_val_meta = gate_val_meta.loc[escalation_mask_val].reset_index(drop=True)
        gate_probs_val[escalation_mask_val] = gate.predict_proba(sub_val_df, sub_val_meta)

    val_final_pred, val_cascade_score = cascade_predict(
        rf_score=gate_val_scores,
        rf_pvalue=gate_val_pvals,
        snort_pred=val_snort_pred,
        gate_prob=gate_probs_val,
        alpha_escalate=alpha_escalate,
        gate_threshold=gate_threshold,
    )

    val_rf_pred = (gate_val_scores > rf_model.threshold).astype(int)
    val_conformal_pred = (gate_val_pvals <= alpha_conformal).astype(int)

    val_csv_path, test_csv_path = export_cascade_split_predictions(
        out_dir=out,
        val_df=gate_val_df,
        test_df=splits.test_all,
        val_rf_score=gate_val_scores,
        val_rf_pvalue=gate_val_pvals,
        val_snort_pred=val_snort_pred,
        val_snort_score=val_snort_score,
        val_gate_prob=gate_probs_val,
        val_escalated=escalation_mask_val.astype(int),
        val_cascade_pred=val_final_pred,
        val_cascade_score=val_cascade_score,
        test_rf_score=test_scores,
        test_rf_pvalue=test_pvals,
        test_snort_pred=test_snort_pred,
        test_snort_score=test_snort_score,
        test_gate_prob=gate_probs_test,
        test_escalated=escalation_mask_test.astype(int),
        test_cascade_pred=final_pred,
        test_cascade_score=cascade_score,
    )
    y_test = splits.test_all["binary_label"].to_numpy()
    rf_pred = (test_scores > rf_model.threshold).astype(int)
    conformal_pred = (test_pvals <= alpha_conformal).astype(int)

    rows = [
        {
            "paper_model": "RF",
            "model": "rf",
            **binary_metrics(y_test, rf_pred, scores=test_scores),
            "threshold": rf_model.threshold,
            "derived_threshold": rf_model.derived_threshold,
        },
        {
            "paper_model": "Signature-Snort",
            "model": "snort_signature",
            **binary_metrics(y_test, test_snort_pred, scores=test_snort_score),
        },
        {
            "paper_model": "RF-Conformal",
            "model": "rf_conformal",
            **binary_metrics(y_test, conformal_pred, scores=1.0 - test_pvals),
            "alpha": alpha_conformal,
            "derived_threshold": conformal.derived_threshold,
        },
        {
            "paper_model": paper_model_name,
            "model": "hybrid_cascade_fastsnort",
            **binary_metrics(y_test, final_pred, scores=cascade_score),
            "alpha_conformal": alpha_conformal,
            "alpha_escalate": alpha_escalate,
            "gate_threshold": gate_threshold,
            "calibration_fraction": calibration_fraction,
            "calibration_benign_n": int(len(cal_benign_df)),
            "gate_val_n": int(len(gate_val_df)),
            "gate_escalation_pool_size": n_esc,
            "gate_escalation_pool_frac": float(n_esc / len(gate_val_pvals)),
            "test_escalation_pool_size": n_test_esc,
            "test_escalation_pool_frac": float(n_test_esc / len(test_pvals)),
        },
    ]
    metrics_df = pd.DataFrame(rows)

    col_order = [
        "paper_model", "model",
        "accuracy", "precision", "recall", "f1",
        "far", "roc_auc", "pr_auc",
        "tp", "tn", "fp", "fn",
        "threshold", "derived_threshold",
        "alpha", "alpha_conformal", "alpha_escalate",
        "gate_threshold", "calibration_fraction",
        "calibration_benign_n", "gate_val_n",
        "gate_escalation_pool_size", "gate_escalation_pool_frac",
        "test_escalation_pool_size", "test_escalation_pool_frac",
    ]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]
    metrics_df.to_csv(out / "overall_metrics.csv", index=False)

    pred_df = splits.test_all.copy()
    pred_df["rf_score"] = test_scores
    pred_df["rf_pvalue"] = test_pvals
    pred_df["rf_pred"] = rf_pred
    pred_df["conformal_pred"] = conformal_pred
    pred_df["snort_pred"] = test_snort_pred
    pred_df["snort_score"] = test_snort_score
    pred_df["gate_prob"] = gate_probs_test
    pred_df["escalated"] = escalation_mask_test.astype(int)
    pred_df["cascade_pred"] = final_pred
    pred_df["cascade_score"] = cascade_score
    pred_df.to_csv(out / "cascade_predictions.csv", index=False)

    write_json(
        {
            "split_strategy": split_strategy,
            "alpha_conformal": alpha_conformal,
            "alpha_escalate": alpha_escalate,
            "gate_threshold": gate_threshold,
            "calibration_fraction": calibration_fraction,
            "rf_model_path": str(rf_model_path) if rf_model_path else None,
            "snort_predictions_path": str(snort_predictions_path),
            "n_total": int(len(cleaned)),
            "n_train": int(len(splits.train_all)),
            "n_val": int(len(splits.val_all)),
            "n_test": int(len(splits.test_all)),
            "n_train_benign": int(len(splits.train_benign)),
            "n_val_benign": int(len(splits.val_benign)),
            "calibration_benign_n": int(len(cal_benign_df)),
            "gate_val_n": int(len(gate_val_df)),
            "gate_val_benign_n": int((gate_val_df["binary_label"] == 0).sum()),
            "gate_val_attack_n": int((gate_val_df["binary_label"] == 1).sum()),
            "gate_escalation_pool_size": int(n_esc),
            "test_escalation_pool_size": int(n_test_esc),
            "snort_test_coverage": test_snort_cov,
            "signature_gate_val_coverage": gate_val_sig_cov,
            "gate_meta_columns": ["rf_score", "rf_pvalue", *gate_meta_cols_extra],
            "tier2_meta_columns_present": gate_meta_cols_extra,
        },
        out / "cascade_summary.json",
    )

    log(f"saved outputs to: {out}")
    print(metrics_df.to_string(index=False), flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run split-calibration cascade with FastSnort naming."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--snort-predictions", required=True)
    parser.add_argument("--rf-model", default=None)
    parser.add_argument("--output-dir", default="outputs_hybrid_cascade_splitcal_fastsnort")
    parser.add_argument("--alpha-conformal", type=float, default=0.05)
    parser.add_argument("--alpha-escalate", type=float, default=0.20)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--split-strategy",
        # Locked environment: temporal_by_file (day-level) split.
        default="temporal_by_file",
        choices=["random", "temporal", "temporal_by_file"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-max-iter", type=int, default=300)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--paper-model-name", default="Hybrid-Cascade-SplitCal-FastSnort")
    args = parser.parse_args()

    run_cascade(
        data_dir=args.data_dir,
        snort_predictions_path=args.snort_predictions,
        output_dir=args.output_dir,
        rf_model_path=args.rf_model,
        alpha_conformal=args.alpha_conformal,
        alpha_escalate=args.alpha_escalate,
        gate_threshold=args.gate_threshold,
        split_strategy=args.split_strategy,
        seed=args.seed,
        gate_max_iter=args.gate_max_iter,
        calibration_fraction=args.calibration_fraction,
        paper_model_name=args.paper_model_name,
    )
