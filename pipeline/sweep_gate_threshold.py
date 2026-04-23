"""Sweep the escalation-gate decision threshold over an existing cascade run.

Reads cascade_predictions.csv (per-row outputs written by
hybrid_cascade_splitcal_fastsnort.py) and reconstructs the final cascade
decision as a function of gate threshold t:

    final_pred(t) = snort_pred OR (escalated AND (gate_prob >= t))

Rows whose signature fast-path already fires (snort_pred == 1) are attacks
regardless of t. Rows that were not escalated to the gate (escalated == 0)
keep their auto-benign decision regardless of t. Only escalated rows with
snort_pred == 0 are affected by the sweep.

Usage:
    python3 sweep_gate_threshold.py \
        --run-dir outputs_hybrid_cascade_splitcal_dualfast_temporal \
        --t-min 0.05 --t-max 0.60 --t-step 0.025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics_at_threshold(
    y: np.ndarray,
    sig_pred: np.ndarray,
    escalated: np.ndarray,
    gate_prob: np.ndarray,
    t: float,
) -> dict:
    gate_hit = (gate_prob >= t).astype(int)
    yhat = (sig_pred | (escalated & gate_hit)).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "t": round(t, 4),
        "acc": acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "far": far,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Cascade output directory containing cascade_predictions.csv")
    ap.add_argument("--pred-file", default="cascade_predictions.csv")
    ap.add_argument("--t-min", type=float, default=0.05)
    ap.add_argument("--t-max", type=float, default=0.60)
    ap.add_argument("--t-step", type=float, default=0.025)
    ap.add_argument("--target-acc", type=float, default=0.95,
                    help="Mark the lowest-t row that meets this accuracy.")
    ap.add_argument("--out-csv", default="gate_threshold_sweep.csv",
                    help="Written inside --run-dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / args.pred_file
    df = pd.read_csv(pred_path)

    required = ["binary_label", "snort_pred", "escalated", "gate_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{pred_path} missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    y = df["binary_label"].astype(int).to_numpy()
    sig_pred = df["snort_pred"].fillna(0).astype(int).to_numpy()
    escalated = df["escalated"].fillna(0).astype(int).to_numpy()
    gate_prob = df["gate_prob"].fillna(0.0).astype(float).to_numpy()

    print(f"loaded {len(df):,} test rows from {pred_path}")
    print(
        f"  benign={int((y == 0).sum()):,}  attack={int((y == 1).sum()):,}  "
        f"snort_hits={int(sig_pred.sum()):,}  escalated={int(escalated.sum()):,}"
    )
    print()

    ts = np.arange(args.t_min, args.t_max + 1e-9, args.t_step)
    rows = [metrics_at_threshold(y, sig_pred, escalated, gate_prob, float(t)) for t in ts]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(run_dir / args.out_csv, index=False)

    fmt = "{t:>5}  acc={acc:.4f}  rec={rec:.4f}  prec={prec:.4f}  f1={f1:.4f}  far={far:.2e}  tp={tp:>6}  fp={fp:>5}"
    hdr = "t        acc       rec       prec      f1        far          tp       fp"
    print(hdr)
    print("-" * len(hdr))
    best_f1 = max(rows, key=lambda r: r["f1"])
    first_target = next((r for r in rows if r["acc"] >= args.target_acc), None)
    for r in rows:
        marker = ""
        if r is best_f1:
            marker += "  <-- best F1"
        if first_target is not None and r is first_target:
            marker += f"  <-- first to hit acc>={args.target_acc}"
        print(fmt.format(**r) + marker)

    print()
    print(f"wrote sweep table -> {run_dir / args.out_csv}")
    print(f"best F1 at t={best_f1['t']}  (acc={best_f1['acc']:.4f}, rec={best_f1['rec']:.4f}, prec={best_f1['prec']:.4f})")
    if first_target is not None:
        print(
            f"first t meeting acc>={args.target_acc}: t={first_target['t']}  "
            f"(acc={first_target['acc']:.4f}, rec={first_target['rec']:.4f}, "
            f"prec={first_target['prec']:.4f}, far={first_target['far']:.2e})"
        )
    else:
        print(f"NO threshold in [{args.t_min}, {args.t_max}] reaches acc>={args.target_acc}.")
        print("Try widening --alpha-escalate or retraining; see discussion in chat.")


if __name__ == "__main__":
    main()
