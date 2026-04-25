# NIDSaaS — Full Manual (cold start to paper reproduction)

This manual walks a brand-new collaborator from **zero** to
(a) reproducing every number in the paper and
(b) bringing up the live multi-tenant prototype.

If you have never cloned this repo before, start at Step 0 and do the
steps in order. Every step has a "Done when" check so you know you can
move on.

Per-script manuals live next to the scripts themselves; this document
is the spine that links them together.

---

> **Just want the detection numbers? Skip Docker and use
> [`DETECTION_QUICKSTART.md`](DETECTION_QUICKSTART.md) instead.**
>
> The quickstart reproduces the paper's full detection experiment
> (cascade + baselines + Snort) in **~4 hours on WSL2** — no Docker,
> no live prototype. It walks you through the recommended file
> placement (repo + CSVs on Windows fs, pcaps copied into WSL Linux
> fs for native Snort speed) and links to the fast 1.5-hour path that
> skips Snort entirely if needed.
>
> Run `bash scripts/precheck_detection.sh` first to confirm your
> machine is set up correctly.

---

## 0. What you will end up with

When you finish the whole manual you will have:

1. A Python environment that can train every detection component in
   the paper.
2. The CIC-IDS2017 flow CSVs and raw pcaps sitting at the paths the
   scripts expect.
3. Locked-environment output directories (`pipeline/outputs_*`) that
   contain the exact numbers reported in the paper's tables.
4. A trained `gate.joblib` bundle the live detector can load.
5. A running Docker Compose stack (Kafka + gateway + detector + Snort
   sidecar + alert fan-out + 3 simulated tenants) producing real
   alerts on real webhook endpoints.

Total wall-clock on a modern laptop (16 GB RAM, 8 cores, no GPU):
roughly **2–3 hours the first time**, dominated by the Snort 3 sidecar
image build (~10 min) and the anomaly-baseline comparison (~30 min).
Subsequent runs are much faster because the big `.joblib` artifacts
can be reused (see "Reusing artifacts" in each per-script manual).

---

## 1. System requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / WSL2 / macOS 13+ | Windows-native works only for the Docker prototype, not for the research scripts (pcap tooling assumes POSIX). |
| CPU | 4 cores | 8+ recommended for LSTM baseline. |
| RAM | 12 GB | 16 GB recommended; the anomaly comparison peaks around 10 GB. |
| Disk | 40 GB free | Dataset (~12 GB pcap + 2.7 GB CSV) + outputs + Docker images. |
| Python | 3.10 or 3.11 | 3.12 works but torch wheels may lag. |
| Docker | 24+ with Compose v2 | Only needed for the prototype (Step 8). Docker Desktop on Windows is fine. |
| Snort 3 | 3.1.x+ | Only needed for offline Snort evaluation (Step 3). The prototype bundles its own Snort in a container. |

No GPU is required. Torch is CPU-only by default.

---

## 2. Directory layout (after you finish Step 1)

```
NIDSaaS_Experiment/
├── MANUAL.md                     <-- this file
├── requirements.txt
├── Final_locked_environment.txt  <-- pinned protocol (split, seed, etc.)
├── paper_contributions.md
│
├── csv_CIC_IDS2017/              <-- flow-level CSVs you download in Step 1
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│
├── pcap_CIC_IDS2017/             <-- raw pcaps you download in Step 1
│   ├── Monday-WorkingHours.pcap
│   ├── Tuesday-WorkingHours.pcap
│   ├── Wednesday-workingHours.pcap
│   ├── Thursday-WorkingHours.pcap
│   └── Friday-WorkingHours.pcap
│
├── pipeline/                     <-- research scripts (train + eval)
├── snort/                        <-- offline Snort evaluation
├── prototype/                    <-- live Docker stack
└── unused_files_laew/            <-- archived older versions; ignore
```

Every script below assumes you run it from inside its own folder
(`cd pipeline/`, `cd snort/`, or `cd prototype/`) unless noted.

---

## 3. Step 0 — Clone, Python env, install dependencies

```bash
# 0.1 Clone (or unzip the share) and cd in
git clone <your-repo-url> NIDSaaS_Experiment
cd NIDSaaS_Experiment

# 0.2 Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 0.3 Install deps
pip install -r requirements.txt

# 0.4 Verify
python -c "import numpy, pandas, sklearn, torch, joblib; print('OK')"
```

**Done when:** `python -c "import numpy, pandas, sklearn, torch, joblib; print('OK')"` prints `OK`.

---

## 4. Step 1 — Obtain the CIC-IDS2017 dataset

You need two artefacts from the **Canadian Institute for Cybersecurity
CIC-IDS2017** public release:

1. **Flow-level CSVs** (the 8 `*.pcap_ISCX.csv` files produced by
   CICFlowMeter). These drive every training script.
2. **Raw pcaps** (the 5 daily capture files). These drive Snort and
   the prototype sidecar.

### 1.1 Register and download

Go to https://www.unb.ca/cic/datasets/ids-2017.html, follow the
download instructions, and grab:

- `MachineLearningCSV.zip` (~220 MB compressed, ~2.7 GB extracted)
- `PCAPs.zip` or the individual daily pcaps (~12 GB total).

### 1.2 Place them in the expected folders

```bash
# from the repo root
mkdir -p csv_CIC_IDS2017 pcap_CIC_IDS2017

# unzip CSVs into csv_CIC_IDS2017/  -- one CSV per day/part
# unzip PCAPs into pcap_CIC_IDS2017/ -- one pcap per weekday
```

When you're done the layout must match the tree in section 2.
File **names** matter — scripts key off them to construct labels and
splits.

**Done when:**

```bash
ls csv_CIC_IDS2017 | wc -l   # -> 8
ls pcap_CIC_IDS2017 | wc -l  # -> 5
```

---

## 5. Step 2 — Install Snort 3 (only if you want Step 3)

The **research** Snort evaluation in `snort/` runs Snort 3 on the
host. If you do not care about reproducing the Snort row of the paper
you can skip this step — the hybrid cascade in Step 4 does not need a
host Snort install because the `σ_S` signature signal is produced
from a pre-computed `signature_merged_predictions.csv` in `pipeline/`.

The **prototype** (Step 8) bundles its own Snort 3 inside a Docker
image, so you don't need to install Snort on the host for Step 8
either.

### 2.1 Install (Ubuntu / WSL)

Follow https://docs.snort.org/start/installation or use the prebuilt
binary from your distro. Verify with:

```bash
snort -V
# Expected: Snort 3.1.x
```

Then fetch the community rules:

```bash
cd NIDSaaS_Experiment/snort
mkdir -p rules/community
# Download snort3-community-rules.tar.gz from snort.org, extract:
#   rules/community/snort3-community.rules
#   rules/community/sid-msg.map
```

See `snort/README_SNORT_updated.md` for the rules folder conventions.

**Done when:** `snort -V` prints a 3.x version and
`snort/rules/community/snort3-community.rules` exists.

---

## 6. Step 3 — Offline Snort evaluation (optional)

This produces the Snort-alone row reported in the paper. It also
produces `signature_merged_predictions.csv`, which the hybrid cascade
**already has a cached copy of** in `pipeline/`, so you can skip Step
3 if you trust the cached copy.

Run this chain (see each per-script `.md` for details):

```bash
cd snort

# 3.1 Replay every daily pcap through Snort
python3 snort_runner.py

# 3.2 Parse alert_fast.txt into structured CSV
python3 parse_fast_alerts.py

# 3.3 (optional) Filter SIDs by policy
python3 filter_policy_snort.py

# 3.4 Match alerts to CIC-IDS2017 labels using the locked split
python3 snort_eval_fixed_v3_splitstrategy.py
```

Per-script manuals:
- `snort/snort_runner.md`
- `snort/parse_fast_alerts.md`
- `snort/filter_policy_snort.md`
- `snort/snort_eval_fixed_v3_splitstrategy.md`

**Done when:** `snort/outputs_snort_eval_v4a_temporal/` contains the
per-class metrics CSV and a `signature_merged_predictions.csv` (or
confirm the cached copy at
`pipeline/signature_merged_predictions.csv` is present).

---

## 7. Step 4 — Train the main detector (Hybrid-Cascade)

This is the headline model. It trains all five stages end-to-end:
rate rules (hand-coded) → RF anomaly → split-conformal → HistGB
escalation gate, with the val-calibrated τ\* applied on top.

```bash
cd pipeline

python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 \
    --alpha-escalate 0.20 \
    --calibration-fraction 0.50 \
    --split-strategy temporal_by_file \
    --seed 42 \
    --output-dir outputs_proposed_locked_a20_g50
```

Full detail: `pipeline/hybrid_cascade_splitcal_fastsnort.md`.

**Done when:** `pipeline/outputs_proposed_locked_a20_g50/` contains

```
rf_anomaly.joblib          <-- reusable, see below
conformal.joblib
gate.joblib
feature_order.json
run_config.json
overall_metrics.csv
per_class_metrics.csv
val_scores_with_predictions.csv
test_scores_with_predictions.csv
```

### 4.1 Locked-environment sweep (the 4 rows in the main table)

The paper reports four (α, calibration-fraction) settings. Runs 2–4
reuse the RF from run 1 with `--rf-model`, which cuts total
wall-clock from ~45 min to ~8 min:

```bash
# run 1: baseline locked operating point (trains RF)
python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.20 \
    --calibration-fraction 0.50 --output-dir outputs_proposed_locked_a20_g50

# runs 2–4: reuse RF, vary α or gate calibration
python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.10 \
    --calibration-fraction 0.50 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a10_g50

python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.20 \
    --calibration-fraction 0.25 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a20_g25

python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.05 \
    --calibration-fraction 0.50 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a05_g50
```

### 4.2 Apply the val-calibrated operating point

The hybrid cascade outputs a raw score per flow. To get the headline
test row under val-accuracy-calibrated τ\*, run:

```bash
python3 proposed_method_valcal.py \
    --scored outputs_proposed_locked_a20_g50/test_scores_with_predictions.csv \
    --val-scored outputs_proposed_locked_a20_g50/val_scores_with_predictions.csv \
    --output-dir outputs_proposed_locked_rate_promoted
```

Full detail: `pipeline/proposed_method_valcal.md`.

**Done when:** `pipeline/outputs_proposed_locked_rate_promoted/overall_metrics_proposed_valcal.csv`
exists. The τ\* it chose is also written to `run_config.json` (currently
τ\* = **0.0642566028502602**) — remember this number; the prototype's
`.env` uses it.

---

## 8. Step 5 — Train the baselines (for the paper's comparison tables)

Each baseline has its own manual in `pipeline/`. Run in this order so
outputs can feed into the comparison script:

```bash
cd pipeline

# 5.1 RF-only and RF+Conformal — reuses cascade scores
python3 rf_baseline_valcal.py \
    --cascade-dir outputs_proposed_locked_a20_g50 \
    --output-dir outputs_rf_baseline_valcal

# 5.2 Rate-rules-only ablation
python3 rate_rules_baseline_valcal.py \
    --data-dir ../csv_CIC_IDS2017 \
    --output-dir outputs_rate_rules_baseline_valcal

# 5.3 Anomaly baselines comparison (RF, Isolation Forest, OCSVM, LSTM-AE)
python3 compare_anomaly_baselines_valcal.py \
    --data-dir ../csv_CIC_IDS2017 \
    --output-dir outputs_baselines_temporal_by_file_valcal_iso
```

Manuals:
- `pipeline/rf_baseline_valcal.md`
- `pipeline/rate_rules_baseline_valcal.md`
- `pipeline/compare_anomaly_baselines_valcal.md`
- `pipeline/compare_anomaly_baselines.md` *(legacy non-valcal; kept
  for reference only)*
- `pipeline/lstm_autoencoder_baseline.md` *(library module — used by
  `compare_anomaly_baselines_valcal.py`, not run directly)*
- `pipeline/signature_rate_rules.md` *(library — already consumed
  inside the hybrid cascade)*

**Done when:** three `outputs_*` directories above each contain their
`overall_metrics_*.csv` file.

---

## 9. Step 6 — Build the paper tables

With all six output folders in place you have every number referenced
in the paper. The mapping:

| Paper table row | Comes from |
|---|---|
| Proposed (val-calibrated) | `outputs_proposed_locked_rate_promoted/overall_metrics_proposed_valcal.csv` |
| Proposed sweep (α × g) | `outputs_proposed_locked_a*_g*/overall_metrics.csv` × 4 |
| RF-only | `outputs_rf_baseline_valcal/overall_metrics_rf_only.csv` |
| RF + Conformal | `outputs_rf_baseline_valcal/overall_metrics_rf_conformal.csv` |
| Rate rules only | `outputs_rate_rules_baseline_valcal/overall_metrics_rate_rules.csv` |
| Isolation Forest / OCSVM / LSTM-AE | `outputs_baselines_temporal_by_file_valcal_iso/overall_metrics_*.csv` |
| Snort only | `snort/outputs_snort_eval_v4a_temporal/` |

---

## 10. Step 7 — Export the trained bundle for the live detector

The prototype's streaming detector loads a single `gate.joblib` that
packs `{rf, conformal, gate, feature_order, tau_star}`. Build it from
your Step 4 outputs:

```bash
cd pipeline

python3 cascade_export_patch.py \
    --data-dir ../csv_CIC_IDS2017 \
    --rf outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --conformal outputs_proposed_locked_a20_g50/conformal.joblib \
    --gate outputs_proposed_locked_a20_g50/gate.joblib \
    --tau-star 0.0642566028502602 \
    --output ../prototype/models/gate.joblib
```

Full detail: `pipeline/cascade_export_patch.md`.

**Done when:** `prototype/models/gate.joblib` exists and is ~50–150
MB. The detector will load it automatically on next restart; otherwise
it falls back to a conservative statistical scorer so the stack still
runs end-to-end.

---

## 11. Step 8 — Bring up the live prototype (Docker)

### 8.1 Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)

- Windows: Docker Desktop + enable WSL2 integration with your Ubuntu
  distro. See
  https://docs.docker.com/desktop/install/windows-install/.
- Linux: `sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin`
  via Docker's official apt repo.

Verify:

```bash
docker --version          # >= 24.0
docker compose version    # v2+
```

### 8.2 Bring up the stack

```bash
cd prototype

# First time only — build images (~10 min for the Snort sidecar)
docker compose build

# Start everything
docker compose up -d

# Verify all services came up healthy
docker compose ps
```

### 8.3 Fire a synthetic attack and watch the alert fan out

```bash
./scripts/demo_attack.sh
```

Expected final output: a JSON alert body with `"tier": "tier1_rate"`
and `"score": 1.0` — the Tier-1 rate rule path catching a synthetic
SYN flood.

Follow the alerts topic live:

```bash
docker exec -it nidsaas_kafka kafka-console-consumer.sh \
    --bootstrap-server kafka:9092 \
    --topic tenant.acme.alerts --from-beginning
```

Or query the in-memory receiver:

```bash
curl -s localhost:9000/alerts | jq 'to_entries[] | {t:.key, n:(.value|length)}'
```

### 8.4 (Optional) Switch to pcap mode — exercise the full Figure-1 pipeline

CSV mode (default) POSTs pre-extracted flow rows straight to the
detector. Pcap mode instead ships raw pcap bytes through the
`flow_extractor` sidecar (CICFlowMeter v4), which is the path drawn in
Figure 1 of the paper. Downstream of the raw topic the two modes are
identical.

```bash
cd prototype

# 8.4.1 flip the simulator to pcap mode (edit .env)
sed -i 's/^SIM_MODE=csv/SIM_MODE=pcap/' .env

# 8.4.2 build the CICFlowMeter sidecar (first time only: ~8 min)
docker compose build flow_extractor

# 8.4.3 restart the stack with pcap mode active
docker compose down -v
docker compose up -d

# 8.4.4 follow extraction + detection
docker compose logs -f flow_extractor detector
```

Expected log sequence:

```
[sim] SIM_MODE=pcap: using /ingest_pcap + flow_extractor path
[gateway] 202 /ingest_pcap chunk_id=Monday-WorkingHours-00000 bytes=4982312
[extractor] [acme] chunk=Monday-WorkingHours-00000 bytes=4982312 flows=1847 elapsed=4.3s
[detector] ... tier=tier2_gate score=... decision=0|1
```

Knobs in `.env` if the default run is too slow / fast:

| Var | Meaning |
|---|---|
| `SIM_PCAP_PACKETS_PER_CHUNK` | Packets per chunk (default 5000). |
| `SIM_PCAP_MAX_CHUNKS_PER_FILE` | Cap so the demo finishes in minutes (default 40). |
| `SIM_PCAP_CHUNK_GAP_SEC` | Gap between chunks (default 1.0s). |
| `PCAP_CHUNK_MAX_BYTES` | Broker/topic/consumer message-size ceiling (default 16 MB). |

Important caveats:

- Pcap mode has **no ground-truth labels**. Alerts reflect detector
  decisions only. Use CSV mode when you need label-based metrics.
- CICFlowMeter feature names must match the RF training distribution.
  Do not swap the extractor to nfstream / a custom port without
  retraining the detector.
- Per-tenant pcaps: drop captures under
  `pcap_CIC_IDS2017/<tenant>/*.pcap` to give different tenants
  different traffic. Otherwise all tenants share the same pcap menu.

### 8.5 Teardown

```bash
docker compose down -v       # -v also drops the Kafka volume
```

Service-level details:
- `prototype/README.md` (overall)
- `prototype/gateway/README.md`
- `prototype/streaming_worker/README.md`
- `prototype/flow_extractor/README.md` *(pcap mode, CICFlowMeter v4)*
- `prototype/snort_sidecar/README.md`
- `prototype/tenant_simulator/README.md`
- `prototype/alert_fanout/README.md`
- `prototype/webhook_receiver/README.md`
- `prototype/init/README.md`
- `prototype/scripts/README.md`

**Done when:** `demo_attack.sh` prints a JSON alert with `"tier":
"tier1_rate"`, and `curl localhost:9000/alerts` returns non-empty
buffers for all three tenants after the simulator finishes.

---

## 12. Quick-reference runbook

Once you have done the full install once, day-to-day work looks like:

```bash
# edit a rate rule or gate hyperparameter, re-train fast using cached RF
cd pipeline
python3 hybrid_cascade_splitcal_fastsnort.py \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --alpha-escalate 0.15 \
    --output-dir outputs_experiment_a15

# re-apply val calibration
python3 proposed_method_valcal.py \
    --scored outputs_experiment_a15/test_scores_with_predictions.csv \
    --val-scored outputs_experiment_a15/val_scores_with_predictions.csv \
    --output-dir outputs_experiment_a15_promoted

# (optional) push new bundle to the prototype detector
python3 cascade_export_patch.py \
    --data-dir ../csv_CIC_IDS2017 \
    --rf outputs_experiment_a15/rf_anomaly.joblib \
    --conformal outputs_experiment_a15/conformal.joblib \
    --gate outputs_experiment_a15/gate.joblib \
    --tau-star "$(jq -r .tau_star outputs_experiment_a15_promoted/run_config.json)" \
    --output ../prototype/models/gate.joblib

cd ../prototype
docker compose restart detector
```

---

## 13. Troubleshooting index

| Symptom | Where to look |
|---|---|
| `FileNotFoundError: ../csv_CIC_IDS2017/Monday-...csv` | You skipped Step 1 or the CSVs are in the wrong folder. File **names** must match exactly. |
| Hybrid cascade runs OOM | Lower `--sample-benign` or close other apps; peak is ~10 GB. |
| `outputs_*/test_scores_with_predictions.csv` empty | Split strategy produced an empty test fold. Check `--split-strategy temporal_by_file` is set and all 8 CSVs are present. |
| Reused RF gives different numbers | You changed `--seed` or `--split-strategy`; RF cache is only valid under the same seed + split + data. See the cheat-sheet in `hybrid_cascade_splitcal_fastsnort.md`. |
| Prototype `Image bitnami/kafka:3.7 not found` | Old compose file. Pull latest — it now uses `apache/kafka:3.9.0`. |
| Docker Desktop "C:\ProgramData\DockerDesktop must be owned by an elevated account" | Uninstalled without admin rights. Delete `C:\ProgramData\DockerDesktop` and `C:\ProgramData\Docker` in an elevated PowerShell, then run Docker Desktop **as administrator** once. |
| `demo_attack.sh` prints no alert | Check `docker compose logs detector` — likely `gate.joblib` is missing and the fallback scorer is running with a very strict τ\*. Re-run Step 7 or lower τ\* temporarily. |
| Snort sidecar idle | No pcap directory for that tenant. Place a pcap at `pcap_CIC_IDS2017/<tenant>/<whatever>.pcap` and `docker compose restart snort_sidecar`. |
| JWT "token expired" on `/ingest` | Client ran longer than the gateway's JWT TTL (default 1h). Re-POST to `/oauth/token` and retry. |

---

## 14. Paper deliverables map (what each step produces)

| Paper deliverable | Produced by | Step |
|---|---|---|
| Main table: proposed (val-calibrated) row | `proposed_method_valcal.py` | 4.2 |
| Sweep table: (α, calibration-fraction) grid | `hybrid_cascade_splitcal_fastsnort.py` × 4 | 4.1 |
| Baselines table | `compare_anomaly_baselines_valcal.py` + `rf_baseline_valcal.py` + `rate_rules_baseline_valcal.py` | 5 |
| Snort row | `snort_eval_fixed_v3_splitstrategy.py` | 3 |
| Architecture figure services | Docker Compose stack under `prototype/` | 8 |
| Threshold τ\* justification | `outputs_proposed_locked_rate_promoted/run_config.json` | 4.2 |
| Rate-rule promotion justification | `outputs_rate_rules_baseline_valcal/` per-class breakdown | 5 |

---

**Stuck?** Each per-script / per-service manual has its own
"Common problems" section that covers the 90% case. Fall back to this
document only for cross-cutting setup issues.
