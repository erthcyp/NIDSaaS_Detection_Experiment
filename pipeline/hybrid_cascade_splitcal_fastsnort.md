# hybrid_cascade_splitcal_fastsnort.md

## Meaning of FastSnort

FastSnort means:
- Snort is still used
- but only as the fast-path short-circuit
- the escalation gate does not use Snort meta-features

This avoids confusion from the older `nosnort` name.

## Recommended temporal_by_file sweep

Run these from `pipeline/`:

### Run 1
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

### Run 2
```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g50 \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.30 \
  --gate-threshold 0.50 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

### Run 3
```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a40_g50 \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.40 \
  --gate-threshold 0.50 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

### Run 4
```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g55 \
  --alpha-conformal 0.05 \
  --alpha-escalate 0.30 \
  --gate-threshold 0.55 \
  --calibration-fraction 0.50 \
  --split-strategy temporal_by_file
```

## Check results

```bash
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a20_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a40_g50/overall_metrics.csv
cat outputs_hybrid_cascade_splitcal_fastsnort_temporal_a30_g55/overall_metrics.csv
```
