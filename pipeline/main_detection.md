# main_detection.md

## Purpose

`main_detection.py` is the main entry point for the detection-side experiments in this project.

It is responsible for:
- loading and preparing the CIC-IDS2017 CSV data
- creating train / validation / test splits
- running one or more detectors
- saving metrics and prediction files
- exporting split metadata for later reuse

The current version supports these experiment modes:
- `signature`
- `rf`
- `lstm`
- `hybrid`
- `all`

---

## Supported modes

### `rf`
Runs only the RF anomaly detector.

Use this when you want:
- the RF baseline
- `rf_predictions.csv`
- the RF model artifact

### `signature`
Runs only the signature prefilter.

Use this when you want:
- the signature-only baseline
- `signature_predictions.csv`
- signature hit summaries

### `lstm`
Runs only the LSTM anomaly detector.

Use this when you want:
- the LSTM baseline
- `lstm_predictions.csv`
- the LSTM model artifact

### `hybrid`
Runs:
- signature
- RF
- OR-based hybrid merge

Use this when you want the original detection-side OR hybrid baseline.

### `all`
Runs:
- signature
- RF
- LSTM
- OR hybrid

Use this when you want all supported baselines in one run.

---

## Important implementation behavior

This updated version uses lazy imports for optional modules.

That means:
- `signature.py` is imported only if `mode` needs signature
- `lstm_anomaly.py` is imported only if `mode` needs LSTM
- `hybrid.py` is imported only if `mode` needs OR hybrid
- `matplotlib` is imported only when saving the plot

This is useful because:
- `--mode rf` can run even if `hybrid.py` is missing
- `--mode rf` can run even if `lstm_anomaly.py` is missing
- missing `matplotlib` will not crash the experiment; it will only skip plotting

---

## Basic command format

Run from the project root:

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_detection_v2 \
  --mode rf
```

General syntax:

```bash
python3 pipeline/main_detection.py \
  --data-dir <csv_folder> \
  --output-dir <output_folder> \
  --mode <mode> \
  --seed <seed>
```

---

## Arguments

### `--data-dir`
Path to the folder containing the CIC-IDS2017 CSV files.

Example:
```bash
--data-dir csv_CIC_IDS2017
```

### `--output-dir`
Path to the folder where results will be saved.

Example:
```bash
--output-dir pipeline/outputs_detection_v2
```

### `--mode`
Which experiment to run.

Allowed values:
- `all`
- `signature`
- `rf`
- `lstm`
- `hybrid`

### `--seed`
Random seed used for data splitting and model-related randomness.

Default:
```text
42
```

---

## Recommended usage

### RF only
Use this for the final assisted hybrid workflow.

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_detection_v2 \
  --mode rf
```

Expected important outputs:
- `rf_predictions.csv`
- `rf_anomaly.joblib`
- `overall_metrics.csv`
- `split_summary.json`
- optionally `split_row_ids.json`

### Signature only
```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_signature_only \
  --mode signature
```

Expected important outputs:
- `signature_predictions.csv`
- `signature_rule_hits_test.csv`
- `signature_rule_summary.csv`
- `signature_config.json`

### LSTM only
Use this only after `lstm_anomaly.py` is added and dependencies are installed.

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_lstm_only \
  --mode lstm
```

Expected important outputs:
- `lstm_predictions.csv`
- `lstm_anomaly.joblib`
- `overall_metrics.csv`

### Hybrid OR baseline
This runs signature + RF + OR hybrid.

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_hybrid_or \
  --mode hybrid
```

Expected important outputs:
- `signature_predictions.csv`
- `rf_predictions.csv`
- `hybrid_predictions.csv`
- `classwise_hybrid.csv`
- `overall_metrics.csv`

### All modes
```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_all_models \
  --mode all
```

Use this only if:
- all modules exist
- all dependencies are installed
- you really want every baseline in one run

---

## Output files

Depending on the selected mode, `main_detection.py` may create the following files.

### Always expected
- `overall_metrics.csv`
- `split_summary.json`

### If split indices are enabled
- `split_row_ids.json`

### If RF runs
- `rf_predictions.csv`
- `rf_anomaly.joblib`

### If signature runs
- `signature_predictions.csv`
- `signature_rule_hits_test.csv`
- `signature_rule_summary.csv`
- `signature_config.json`

### If LSTM runs
- `lstm_predictions.csv`
- `lstm_anomaly.joblib`

### If OR hybrid runs
- `hybrid_predictions.csv`
- `classwise_hybrid.csv`

### Optional plot
- `detection_performance.png`

If `matplotlib` is not installed, plotting is skipped safely.

---

## Output file meanings

### `rf_predictions.csv`
Per-row RF predictions on the test split.

This is the key file needed by the final assisted hybrid pipeline.

Important columns usually include:
- original test-set columns
- `rf_pred`
- `rf_score`

### `signature_predictions.csv`
Per-row signature baseline predictions on the test split.

Important column:
- `signature_pred`

### `lstm_predictions.csv`
Per-row LSTM predictions on the test split.

Important columns:
- `lstm_pred`
- `lstm_score`

### `hybrid_predictions.csv`
Per-row OR-hybrid predictions.

Important column:
- `hybrid_pred`

### `overall_metrics.csv`
Summary table of model performance for the selected mode.

Typical columns:
- `paper_model`
- `model`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `far`
- `roc_auc`
- `pr_auc`
- `tp`
- `tn`
- `fp`
- `fn`
- `threshold`
- `derived_threshold`
- `total_time_s`

---

## Typical workflow in this project

### Final deployment-oriented workflow
For the final selected model in this project, `main_detection.py` is used only to generate the RF side outputs.

Recommended sequence:
1. run RF with `main_detection.py`
2. run Snort replay and parse alerts
3. filter alerts with `v4a_keep_sids.txt`
4. map Snort alerts with `snort_eval_fixed_v3.py`
5. merge RF + Snort using `hybrid_assisted_from_snort_rf.py`

### Final RF command
```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_detection_v2 \
  --mode rf
```

This produces:
```text
pipeline/outputs_detection_v2/rf_predictions.csv
```

That file is then used by the final assisted hybrid script.

---

## When to use each mode in the paper

### Use `rf`
For:
- RF baseline
- final assisted hybrid input

### Use `signature`
For:
- signature-only baseline
- signature rule inspection

### Use `lstm`
For:
- LSTM comparison baseline

### Use `hybrid`
For:
- OR-based hybrid ablation
- class-wise hybrid detection analysis

### Use `all`
For:
- one-shot baseline generation when all modules are ready

---

## Common problems

### Problem: `ModuleNotFoundError: No module named 'hybrid'`
Cause:
- `hybrid.py` is missing
- but `mode` requires hybrid

Fix:
- use `--mode rf` if you only need RF
- or add `hybrid.py` back

### Problem: `ModuleNotFoundError: No module named 'lstm_anomaly'`
Cause:
- `lstm_anomaly.py` is missing
- but `mode` requires LSTM

Fix:
- use `--mode rf` or `--mode signature`
- or add `lstm_anomaly.py`

### Problem: `ModuleNotFoundError: No module named 'matplotlib'`
Cause:
- plotting dependency not installed

Fix:
- install `matplotlib`
- or continue, since plotting is skipped safely in the updated version

### Problem: `rf_predictions.csv` not found after running RF mode
Fix:
- confirm you replaced the old script with the updated one
- confirm the output path is correct
- confirm the RF block completed successfully

### Problem: run command works only from some directories
Fix:
- run from the project root
- use full paths when needed

---

## Suggested future extension

When `lstm_anomaly.py` is added later, this same script can be reused without structural changes.

Recommended steps:
1. add `lstm_anomaly.py`
2. install required dependencies, including `torch`
3. run:

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_lstm_only \
  --mode lstm
```

Then compare:
- RF
- LSTM
- Signature
- OR Hybrid
- final Assisted Hybrid

---

## Final note

For this project's final result, the most important use of `main_detection.py` is:
- generate `rf_predictions.csv`
- keep the RF baseline reproducible
- optionally generate signature / LSTM / OR-hybrid baselines for comparison

The final selected deployment-oriented model is not produced directly inside `main_detection.py`.

Instead, it is produced later by:
- `snort_eval_fixed_v3.py`
- `hybrid_assisted_from_snort_rf.py`
