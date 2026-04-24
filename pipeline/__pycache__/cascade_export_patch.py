"""
cascade_export_patch.py
=======================

Helper functions to export BOTH validation and test prediction tables from the
hybrid cascade pipeline, so they can be consumed by `proposed_method_valcal.py`.

How to use
----------
1) Copy the functions in this file into your hybrid cascade script
   (for example: hybrid_cascade.py / hybrid_cascade_splitcal_*.py).

2) After you have computed the following objects:
   - splits.val_all
   - splits.test_all
   - val_scores
   - val_pvals
   - test_scores
   - test_pvals
   - val_snort_pred
   - val_snort_score
   - test_snort_pred
   - test_snort_score
   - val_gate_prob
   - test_gate_prob
   - val_escalated
   - test_escalated
   - val_cascade_pred
   - test_cascade_pred
   - val_cascade_score
   - test_cascade_score

   call:

       export_cascade_split_predictions(
           out_dir=out_dir,
           val_df=splits.val_all,
           test_df=splits.test_all,
           val_rf_score=val_scores,
           val_rf_pvalue=val_pvals,
           val_snort_pred=val_snort_pred,
           val_snort_score=val_snort_score,
           val_gate_prob=val_gate_prob,
           val_escalated=val_escalated,
           val_cascade_pred=val_cascade_pred,
           val_cascade_score=val_cascade_score,
           test_rf_score=test_scores,
           test_rf_pvalue=test_pvals,
           test_snort_pred=test_snort_pred,
           test_snort_score=test_snort_score,
           test_gate_prob=test_gate_prob,
           test_escalated=test_escalated,
           test_cascade_pred=test_cascade_pred,
           test_cascade_score=test_cascade_score,
       )

3) Then run:
       proposed_method_valcal.py
   with:
       --val-csv  <out_dir>/val_cascade_predictions.csv
       --test-csv <out_dir>/test_cascade_predictions.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


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


INTEGRATION_SNIPPET = """
# ------------------------------------------------------------------
# After you have computed validation-side and test-side predictions:
# ------------------------------------------------------------------
val_csv_path, test_csv_path = export_cascade_split_predictions(
    out_dir=out_dir,
    val_df=splits.val_all,
    test_df=splits.test_all,
    val_rf_score=val_scores,
    val_rf_pvalue=val_pvals,
    val_snort_pred=val_snort_pred,
    val_snort_score=val_snort_score,
    val_gate_prob=val_gate_prob,
    val_escalated=val_escalated,
    val_cascade_pred=val_cascade_pred,
    val_cascade_score=val_cascade_score,
    test_rf_score=test_scores,
    test_rf_pvalue=test_pvals,
    test_snort_pred=test_snort_pred,
    test_snort_score=test_snort_score,
    test_gate_prob=test_gate_prob,
    test_escalated=test_escalated,
    test_cascade_pred=test_cascade_pred,
    test_cascade_score=test_cascade_score,
)

# Optional: keep old combined export too, if you still want it
# combined = pd.concat(
#     [
#         pd.read_csv(val_csv_path),
#         pd.read_csv(test_csv_path),
#     ],
#     ignore_index=True,
# )
# combined.to_csv(Path(out_dir) / "cascade_predictions.csv", index=False)
"""
