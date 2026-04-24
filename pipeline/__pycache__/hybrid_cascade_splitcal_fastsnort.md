# `hybrid_cascade_splitcal_fastsnort.py` — reference guide

This is the **training + evaluation pipeline** that produces the
Hybrid-Cascade detector described in Section 4 of the paper. It is the
single script that trains every learnable block, wires them together in
cascade order, runs split-conformal calibration on a held-out validation
fold, and scores the held-out test fold.

It's the script you run when you want **fresh model artefacts** (`.joblib`)
and **prediction tables** for the val-calibrated threshold step done later
by `proposed_method_valcal.py`.

---

## 1. What the pipeline is

The detector is the 5-stage Hybrid-Cascade:

```
  raw flow ──► [1] Snort packet-signature σ_S ───────────┐
             │                                           │   (Tier-1 fast path:
             ├► [2] Rate rules R = {V, L, S, R, P, B} ───┤    short-circuit to
             │                                           │    alert, score=1.0)
             ├► [3] RF-anomaly s(x) ──► [4] Split-conformal p̂(x)
             │                                           │
             └──────────────────────────────────────────► [5] Escalation gate
                                                               g(z(x))  (HistGB)
                                                                 │
                                                          score_final ≥ τ*
                                                                 │
                                                                 ▼
                                                          alert / benign
```

### Two-tier signature layer — "FastSnort"

The signature layer is itself two-tiered:

| Tier | Which signals | Role |
|---|---|---|
| **Tier-1 (fast path)** | `signature_pred` (OR of Snort + high-precision rate rules V/S/R/B) | short-circuit straight to alert, score 1.0 |
| **Tier-2 (meta features)** | `rate_L`, `rate_P` (looser rules) | passed into the escalation gate alongside `rf_score` and `rf_pvalue` |

The word *FastSnort* in the name refers to the fact that the Snort signal
is **only** used on the fast path. The gate never re-reads σ_S as a
standalone meta-feature (that would double-count and could push precision
down). This avoids confusion with earlier `nosnort` / `gate+snort`
variants that tried both.

### Split-conformal calibration

To produce a proper p-value p̂(x) we hold out a benign-only subset of the
validation fold (`calibration_fraction`, default 50 %) as the
**calibration set**. The remaining validation rows become the
**gate-training set**. This is standard split-conformal — the calibration
scores are exchangeable with test scores under the null, so the reported
p-values are finite-sample valid.

---

## 2. The blocks, in order

| # | Block | Learnable? | Saved artefact | Produced by |
|---|---|---|---|---|
| 1 | `SelfSupervisedRFAnomaly` | yes | `rf_anomaly.joblib` | fresh training on `train_benign + val_benign`, or loaded via `--rf-model` |
| 2 | `ConformalAnomalyWrapper` | yes (parameter-free calibration) | `conformal_wrapper.joblib` | always trained on `cal_benign` subset of this run |
| 3 | Signature table merge | no | — | `load_signature_table(snort_predictions_path)` |
| 4 | Rate-rule columns (V/L/S/R/P/B) | no (hand-coded) | — | carried in the merged CSV when present |
| 5 | `EscalationGateFastSnort` (HistGradientBoostingClassifier) | yes | `escalation_gate_fastsnort.joblib` | always trained on the escalation pool of this run |

`rf_pvalue ≤ alpha_escalate` is the **gate in-scope filter**: only rows
where the conformal p-value is small enough are forwarded to the gate.
Everything else keeps the RF score.

---

## 3. Where each block is trained vs. applied

```
                           train                           apply
                   ┌───────────────────┐          ┌──────────────────────┐
  RF anomaly       │ train_benign ∪    │          │ cal_benign, gate_val │
                   │ val_benign        │          │ , test_all           │
                   └───────────────────┘          └──────────────────────┘

  Conformal        │ cal_benign scores │          │ gate_val, test       │

  Gate (HistGB)    │ escalation pool of│          │ escalation pool of   │
                   │ gate_val          │          │ test                 │

  Rate rules       │ —                 │          │ all rows (applied)   │

  Snort            │ — (offline pcap   │          │ all rows (joined by  │
                   │ replay, separate) │          │ row_id)              │
```

---

## 4. Inputs

- **`--data-dir`** — path to `csv_CIC_IDS2017/` (day-level CSVs).
- **`--snort-predictions`** — pre-computed signature table. Typical choice is
  `../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv` (or a
  merged CSV with the six `rate_*` columns). Must contain `row_id` and
  `signature_pred`; other columns are optional.
- **`--rf-model`** *(optional)* — reuse a previously trained RF `.joblib`
  instead of retraining. See §6 for safety conditions.

### Hyperparameters

| Flag | Default | Meaning |
|---|---|---|
| `--alpha-conformal` | 0.05 | conformal significance level (for reporting, not used in the cascade decision) |
| `--alpha-escalate` | 0.20 | **key:** p-value cutoff below which a flow is forwarded to the gate |
| `--gate-threshold` | 0.50 | operating threshold on gate probability (superseded by val-calibration downstream) |
| `--calibration-fraction` | 0.50 | share of validation benigns used as the split-conformal calibration set |
| `--split-strategy` | `temporal_by_file` | matches `Final_locked_environment.txt` |
| `--seed` | 42 | global random seed |
| `--gate-max-iter` | 300 | HistGB boosting iterations |

---

## 5. Outputs

Every output goes under `--output-dir`. One run writes:

### 5.1 Trained artefacts (reusable — see §6)

| File | Content |
|---|---|
| `rf_anomaly.joblib` | the self-supervised RF model + derived threshold |
| `conformal_wrapper.joblib` | calibration scores + `derived_threshold` for the chosen α |
| `escalation_gate_fastsnort.joblib` | fitted HistGB gate + its preprocessor config |

### 5.2 Prediction tables

| File | Rows | Columns of interest |
|---|---|---|
| `cascade_predictions.csv` | test set | `rf_score`, `rf_pvalue`, `rf_pred`, `conformal_pred`, `snort_pred`, `snort_score`, `gate_prob`, `escalated`, `cascade_pred`, `cascade_score` + original features |
| `val_cascade_predictions.csv` | validation set (post-calibration split) | same layout with `split=validation` |
| `test_cascade_predictions.csv` | test set | same layout with `split=test` |

The two `*_cascade_predictions.csv` files are **exactly the shape
`proposed_method_valcal.py` expects** for downstream val-calibrated
thresholding.

### 5.3 Metrics

| File | Content |
|---|---|
| `overall_metrics.csv` | four rows — RF, Signature-Snort, RF-Conformal, and the full Hybrid-Cascade — with Accuracy, Precision, Recall, F1, FAR, ROC-AUC, PR-AUC, TP/TN/FP/FN, plus the cascade-specific knobs (alpha_escalate, gate_threshold, pool sizes) |

### 5.4 Run manifest

| File | Content |
|---|---|
| `cascade_summary.json` | every hyperparameter, row counts for each split, escalation pool size, signature coverage, and which tier-2 meta-columns were seen in the signature CSV. Useful for audit and paper tables. |

---

## 6. Reusing `.joblib` artefacts instead of retraining

This is the main "cheat" — the **RF block is by far the slowest part**
(it fits many trees on 64-component SVD features over a few million
benign rows). Re-using it saves ~80 % of wall-clock time per run.

### 6.1 RF (`rf_anomaly.joblib`) — safe to reuse

Pass `--rf-model path/to/rf_anomaly.joblib`. Safe **only if** the reused
RF was trained under the same:

- `--data-dir`
- `--split-strategy`  (`temporal_by_file` in the locked env)
- `--seed`
- preprocessing config in `config.RFConfig`

If any of those change, the feature order or the benign-only train pool
changes, and the reused scores are meaningless. Fresh train is cheap if
in doubt.

### 6.2 Conformal (`conformal_wrapper.joblib`) — **do not reuse as-is**

The calibration set is drawn randomly from `val_benign` every run, so
the stored calibration scores are specific to that draw. If you reused
the `.joblib` without re-drawing `cal_benign`, the reported p-values
would still be valid exchangeably but would disagree with the `gate_val`
scores the same run computes. Best practice: **always re-fit** conformal
inside the current run. The saved `.joblib` is useful for forensic
analysis, not for skipping a stage.

### 6.3 Gate (`escalation_gate_fastsnort.joblib`) — reuse only for scoring

Training the gate depends on the exact escalation pool, which depends on
`alpha_escalate` **and** on the conformal p-values, which depend on the
current calibration draw. So the gate `.joblib` is reusable only if:

- same RF joblib,
- same calibration_fraction, seed, and split,
- same alpha_escalate.

In practice, the gate `.joblib` is most useful to the **prototype**
(`prototype/streaming_worker/cascade.py::_JoblibScorer`) which loads
exactly this bundle to score live flows without retraining anything.
See `cascade_export_patch.py` for the exporter that packages the three
joblibs into the single bundle the prototype expects.

### 6.4 Summary cheat-sheet

| Want to change | Can reuse RF? | Can reuse Conformal? | Can reuse Gate? |
|---|---|---|---|
| Seed only | ✗ | ✗ | ✗ |
| `alpha_escalate` | ✓ | ✓ | ✗ |
| `gate_threshold` (downstream) | ✓ | ✓ | ✓ (re-threshold `gate_prob`) |
| `calibration_fraction` | ✓ | ✗ | ✗ |
| Split strategy | ✗ | ✗ | ✗ |
| Nothing (just re-running) | ✓ | ✓ (deterministic with same seed) | ✓ |

---

## 7. Locked-environment sweep

From `Final_locked_environment.txt`:

- split: `temporal_by_file` (64/16/20 day-level)
- seed: 42
- calibration_fraction: 0.50
- threshold selection: **val-calibrated**, done downstream by
  `proposed_method_valcal.py`, **not** by the `--gate-threshold` flag here.

The four runs below are the full α / gate-threshold grid used for the
paper's sensitivity table. Only the first one is the locked operating
point; the rest are ablation rows.

### Run 1 — locked main

```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50 \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.20 \
  --gate-threshold 0.50 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

### Run 2 — loosen α_escalate

```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g50 \
  --rf-model outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50/rf_anomaly.joblib \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.30 \
  --gate-threshold 0.50 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

### Run 3 — loosen further

```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a40_g50 \
  --rf-model outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50/rf_anomaly.joblib \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.40 \
  --gate-threshold 0.50 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

### Run 4 — stricter gate threshold

```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g55 \
  --rf-model outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50/rf_anomaly.joblib \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.30 \
  --gate-threshold 0.55 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

> **Tip:** runs 2–4 pass `--rf-model` pointing at the artefact produced
> by run 1. They reuse the trained RF and only re-fit the conformal and
> gate stages. Wall-clock time drops from ~45 min to ~8 min per run.

### Check results

```bash
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a40_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g55/overall_metrics.csv
```

---

## 8. Downstream consumers

The prediction CSVs from this script are the input to several follow-up
scripts. Understanding which file feeds which step prevents duplicate
training.

| Consumer | Reads | Produces |
|---|---|---|
| `proposed_method_valcal.py` | `val_cascade_predictions.csv`, `test_cascade_predictions.csv` | val-calibrated τ\* on `final_score = max(snort, gate_prob)` → `outputs_proposed_locked_rate_promoted/overall_metrics_proposed_valcal.csv` |
| `compare_anomaly_baselines_valcal.py` | scores of RF, RF-Conformal, LSTM autoencoder | baseline comparison table |
| `rate_rules_baseline_valcal.py` | `signature_merged_predictions.csv` | rate-rule ablation table |
| `rf_baseline_valcal.py` | raw data + RF (`rf_anomaly.joblib` optional) | RF-only val-cal baseline |
| `cascade_export_patch.py` | the three `.joblib` files | single bundle for `prototype/streaming_worker/` |
| `prototype/streaming_worker/cascade.py` | bundle from `cascade_export_patch.py` | live per-flow scoring inside Kafka streaming worker |

---

## 9. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Escalation pool too small or single-class` | `alpha_escalate` too tight | raise `--alpha-escalate` to 0.30 or widen `--calibration-fraction` |
| `Signature predictions CSV must contain row_id` | wrong snort CSV — the runner output, not the eval output | pass the file from `snort/outputs_snort_eval_v4a/` |
| Reused `--rf-model` produces nonsense metrics | split strategy or seed changed | retrain RF from scratch |
| Gate `.joblib` size huge | HistGB trees saved by default; fine | — |
| Script finishes but `overall_metrics.csv` is empty | test set is single-class; check the split | inspect `cascade_summary.json` |

---

## 10. What goes into the paper

From a single locked run you can pull:

- **Main results row** (`paper_model = Hybrid-Cascade (ours)` in
  `overall_metrics_proposed_valcal.csv` after `proposed_method_valcal.py`)
- **Ablation rows** for RF, Signature-Snort, RF-Conformal from
  `overall_metrics.csv`
- **Threshold sensitivity figure** by concatenating the four α/gate runs
  and plotting Accuracy vs. FAR
- **Confusion matrix** from `cascade_predictions.csv`
- **Coverage statistics** from `cascade_summary.json` (signature coverage,
  escalation pool sizes) for the paper's reproducibility appendix

These feed the tables in `NIDAAS_overleaf/first_Idea/5_experiment.tex`.
