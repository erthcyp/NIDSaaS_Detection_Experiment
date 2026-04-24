# rate_rules_baseline_valcal.md

## Purpose

Rate-rule-only ablation for the comparison table. Answers the reviewer
question *"what if you used only the rate rules, with no ML gate and no
Snort?"* Emits four rows that drop straight into the val-calibrated
master table:

| Method                       | What it tests                                          |
| ---------------------------- | ------------------------------------------------------ |
| Rate Rules (V\|S\|P)         | The Tier-1 promotion set actually used in the cascade  |
| Rate Rules (all 6, OR)       | Full union of all six rate rules                       |
| Rate Rules (count)           | Sum of fires (0..6); val-cal picks the cut             |
| Snort + Rate Rules           | Full signature stack, no ML at all (rate OR snort)     |

All four are deterministic functions of the cleaned flow features (and
Snort fires). No retraining is required; the script only re-uses the
cascade's val/test prediction CSVs (for labels and Snort fires) plus the
rate-rule fires from `signature_merged_predictions.csv`.

## Run

From `pipeline/`:

```bash
python3 rate_rules_baseline_valcal.py \
  --val-csv  outputs_hybrid_cascade_splitcal_fastsnort_temporal/val_cascade_predictions.csv \
  --test-csv outputs_hybrid_cascade_splitcal_fastsnort_temporal/test_cascade_predictions.csv \
  --rate-csv signature_merged_predictions.csv \
  --out-dir  outputs_rate_rules_baseline_valcal
```

Defaults match the headline cascade run, so just:

```bash
python3 rate_rules_baseline_valcal.py
```

is enough.

## Outputs

`outputs_rate_rules_baseline_valcal/`

```
overall_metrics_rate_rules_valcal.csv   # one row per (method, operating_point)
rate_rules_table_fragment.tex           # LaTeX rows ready for the master table
per_class_rate_VSP.csv                  # which attack classes V|S|P catches/misses
per_class_rate_OR.csv                   # same, for the full OR
per_class_rate_count.csv                # same, for the count score
per_class_snort_plus_rates.csv          # signature stack vs each attack class
run_config.json                         # provenance + protocol blurb
```

## What to do with the results

1. **Open `overall_metrics_rate_rules_valcal.csv`.** The four `method`
   rows at `operating_point = val_accuracy_calibrated` go into the master
   comparison table directly under the existing baselines.

2. **Open the per-class CSVs.** This is where the ablation story lives:
   you should see *high* detection rate on DoS/DDoS/PortScan/Patator and
   *near-zero* detection on Web Attack / Infiltration / Bot / Heartbleed.
   That contrast is the headline figure for "why the gate is necessary."

3. **Add the LaTeX fragment** at `rate_rules_table_fragment.tex` to the
   ablation block of the paper. Same column schema as the existing
   baseline fragments, so it lines up with no edits.

## Why the protocol matches the headline run

- Same split strategy (`temporal_by_file`, seed 42, 64/16/20)
- Same labels (binary_label as in load_data.py line 92: 1=attack, 0=benign)
- Same threshold-selection rule (val-accuracy-calibrated, tie-safe
  achievable-cut search; identical implementation to
  `rf_baseline_valcal.py`)
- Test set used exactly once per method, never for tuning

So the rate-rule-only rows are directly comparable to the RF, RF+Conformal,
HistGB, and full Hybrid-Cascade rows already in the master table.
