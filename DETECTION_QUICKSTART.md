# DETECTION_QUICKSTART.md

**Reproduce the paper's detection experiment end-to-end on WSL2.
No Docker. No prototype.**

This guide is a focused subset of `MANUAL.md` for collaborators who:

- already have the CIC-IDS2017 raw `.pcap` files and `.csv` flow tables,
- want to run on **WSL2 + Ubuntu** (the recommended setup) or native Linux, and
- only need the **detection numbers** in the paper — not the live SaaS
  prototype under `prototype/`.

The full reproduction includes Snort 3 (run on raw pcaps) and the ML
cascade + baselines (run on flow CSVs). Total wall-clock is **~5 hours**
on a typical i7 / 16 GB laptop, dominated by Snort replay.

If you are missing the pcaps, or want the fast ML-only path that skips
Snort entirely, see "Skipping Snort" at the end of this document.

---

## §0  Audience assumption

You have:

- Windows 10/11 with **WSL2 + Ubuntu 22.04**, *or* native Ubuntu 22.04 / Debian 12
- Python 3.10 or 3.11
- ≥12 GB host RAM (16 GB recommended)
- ≥40 GB free disk (Snort outputs + ML artifacts + pcap copy)
- The CIC-IDS2017 archives extracted somewhere on your machine:
  - `MachineLearningCSV/` — 8 flow CSVs (~2.7 GB extracted)
  - `PCAPs/` — 5 daily pcap files (~12 GB extracted)

You do **not** need:

- Docker / Docker Desktop
- A separate VM — VS Code's WSL extension is enough
- Any Snort binary pre-installed (we install it in `§4`)

---

## §1  WSL2 sanity check

Skip to `§2` if you are on native Linux.

```bash
# inside your WSL Ubuntu shell
uname -a                              # Linux x86_64
free -h                               # mem.total >= 12 GB
python3 --version                     # 3.10.x or 3.11.x
df -h ~                               # >= 40 GB free in $HOME
```

> **WSL pitfall — RAM cap.**
> WSL2 caps memory at 50% of the host by default (so 8 GB on a 16 GB
> laptop). The anomaly-baselines comparison peaks ~10 GB and will get
> OOM-killed. Fix on the **Windows** side:
>
> ```ini
> # %UserProfile%\.wslconfig
> [wsl2]
> memory=14GB
> swap=8GB
> ```
>
> Then in PowerShell: `wsl --shutdown` and re-open Ubuntu.

---

## §2  File layout — the recommended split

You almost certainly opened the repo in **VS Code on Windows** (e.g.
`C:\Users\you\NIDSaaS_Experiment`). That's fine — but Snort reads the
pcaps **packet-by-packet** through the Linux filesystem, and the
Windows ↔ Linux bridge (`/mnt/c/...`) is 5–10× slower than native
Linux fs. On a 12 GB pcap that turns a 2-hour Snort run into 8+
hours.

The trick: **keep the repo + CSVs on Windows, copy only the pcaps into
WSL Linux fs**. Editing in VS Code works exactly as before; Snort gets
native disk speed; nothing else changes.

| What | Where | Why |
|---|---|---|
| Repo (`NIDSaaS_Experiment/`) | Windows fs — wherever you cloned it (e.g. `C:\Users\you\NIDSaaS_Experiment`) | VS Code edits it normally; ML scripts only read CSVs once at startup so the bridge cost is negligible |
| CSV dataset (`csv_CIC_IDS2017/`) | Inside the repo on Windows fs | Same |
| **PCAP dataset** (`~/pcap_CIC_IDS2017/`) | **Inside WSL Linux fs** (`/home/<you>/pcap_CIC_IDS2017/`) | Snort hits this thousands of times during replay — must be on native Linux fs |

### 2.1  Place the CSVs (Windows side, inside the repo)

```text
NIDSaaS_Experiment/                            ← on Windows, e.g. C:\Users\you\
└── csv_CIC_IDS2017/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

In a WSL terminal:

```bash
cd /mnt/c/Users/<you>/NIDSaaS_Experiment       # adjust to your path
mkdir -p csv_CIC_IDS2017
cp /mnt/c/Users/<you>/Downloads/MachineLearningCSV/*.csv csv_CIC_IDS2017/
```

### 2.2  Place the PCAPs (WSL side, outside the repo)

```bash
mkdir -p ~/pcap_CIC_IDS2017
cp /mnt/c/Users/<you>/Downloads/PCAPs/*.pcap ~/pcap_CIC_IDS2017/

ls ~/pcap_CIC_IDS2017
# Expected: Monday-WorkingHours.pcap, Tuesday-..., Wednesday-..., Thursday-..., Friday-...
```

This is a one-time copy of ~12 GB. After this you never touch
`/mnt/c/.../PCAPs/` again — the rest of the guide always points to
`~/pcap_CIC_IDS2017/`.

> **Why not just symlink to `/mnt/c/...`?** A symlink doesn't help —
> the slowness is in the underlying filesystem bridge, not the path.
> The copy is the fix.

### 2.3  VS Code workflow

Both editing styles work; pick whichever is more familiar. **Always
use a WSL bash terminal**, not PowerShell, for running the scripts.

| You usually | What to do |
|---|---|
| Open VS Code on Windows pointing at `C:\Users\you\NIDSaaS_Experiment` | **Terminal → New Terminal** → click the `v` next to `+` → pick **Ubuntu (WSL)** |
| Use the VS Code WSL extension and "Reopen Folder in WSL" | Already on WSL — terminal opens in WSL by default |

---

## §3  Python environment

Any Python environment works as long as the deps from
`requirements.txt` import cleanly. Pick whichever you already use:

```bash
cd /mnt/c/Users/<you>/NIDSaaS_Experiment    # or wherever your repo is

# Option A — venv (recommended for clean isolation)
python3 -m venv .venv
source .venv/bin/activate

# Option B — system Python
# (no venv command needed; just install deps below)

# Option C — conda / mamba
# conda create -n nidsaas python=3.11 && conda activate nidsaas

# install deps (any option above)
pip install --upgrade pip
pip install -r requirements.txt

# verify
python -c "import numpy, pandas, sklearn, torch, joblib; print('OK')"
```

**Done when:** the import line prints `OK`.

Run the pre-flight check:

```bash
bash scripts/precheck_detection.sh
```

It should report `READY`. If anything is in red, fix it before going on
— the Snort and ML steps each take 30+ minutes and you do not want to
discover a missing CSV halfway through.

---

## §4  Install Snort 3 in WSL

This is a **native Linux build**, not a container. ~30 minutes
one-time.

The full recipe lives in `snort/README_SNORT_updated.md` § "Snort 3
installation summary". The condensed version:

```bash
# 4.1 — apt deps
sudo apt update && sudo apt install -y \
    build-essential cmake git pkg-config flex bison \
    libpcap-dev libpcre2-dev zlib1g-dev libdumbnet-dev \
    libluajit-5.1-dev libssl-dev libhwloc-dev \
    autotools-dev libtool liblzma-dev libhyperscan-dev

# 4.2 — build LibDAQ
cd ~ && git clone https://github.com/snort3/libdaq.git
cd libdaq && ./bootstrap
./configure --prefix=/usr/local/lib/daq_s3
make -j"$(nproc)" && sudo make install
echo '/usr/local/lib/daq_s3/lib/' | sudo tee /etc/ld.so.conf.d/libdaq3.conf
sudo ldconfig

# 4.3 — build Snort 3
cd ~ && git clone https://github.com/snort3/snort3.git
cd snort3
./configure_cmake.sh \
    --prefix=/opt/snort3 \
    --with-daq-includes=/usr/local/lib/daq_s3/include/ \
    --with-daq-libraries=/usr/local/lib/daq_s3/lib/
cd build && make -j"$(nproc)" && sudo make install
sudo ldconfig

# 4.4 — verify
/opt/snort3/bin/snort -V        # → "Snort++ 3.1.x"
```

Community rules are already committed at
`snort/rules/community/snort3-community.rules`, so no separate download
is required.

**Done when:** `/opt/snort3/bin/snort -V` prints a 3.x version string.

---

## §5  Run Snort offline evaluation

All commands run from inside the repo's `snort/` folder. Note that
`--pcap-dir` points at the Linux-fs copy from `§2.2`, **not** the
Windows-fs original.

```bash
cd snort                      # i.e. /mnt/c/.../NIDSaaS_Experiment/snort

# 5.1 — replay all daily pcaps through Snort  (~2–3 hr)
python3 snort_runner.py \
    --snort-exe   /opt/snort3/bin/snort \
    --pcap-dir    ~/pcap_CIC_IDS2017 \
    --rules       /opt/snort3/etc/snort/snort.lua \
    --extra-rules rules/community/snort3-community.rules \
    --out-dir     ~/snort_outputs

# 5.2 — parse alert_fast.txt → CSV  (~5 min)
python3 parse_fast_alerts.py \
    --input-dir  ~/snort_outputs \
    --output-csv ~/snort_alerts.csv

# 5.3 — apply experiment SID policy v4a (final paper policy)
python3 filter_policy_snort.py \
    --input-csv   ~/snort_alerts.csv \
    --policy-file rules/policy/v4a_keep_sids.txt \
    --output-csv  ~/snort_alerts_v4a.csv

# 5.4 — map filtered alerts to the locked CIC-IDS2017 split  (~10 min)
python3 snort_eval_fixed_v3_splitstrategy.py \
    --data-dir     ../csv_CIC_IDS2017 \
    --snort-alerts ~/snort_alerts_v4a.csv \
    --output-dir   outputs_snort_eval_v4a_temporal \
    --ignore-time
```

**Done when:** `snort/outputs_snort_eval_v4a_temporal/` contains:

```
snort_signature_predictions.csv
snort_signature_metrics.csv
```

The `snort_signature_predictions.csv` is the input to the cascade in
`§6`.

---

## §6  Run the ML detection pipeline

All commands run from `pipeline/`.

### 6.1 — Train the headline Hybrid-Cascade  (~30 min)

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

### 6.2 — Apply the val-calibrated operating point  (~1 min)

```bash
python3 proposed_method_valcal.py \
    --scored     outputs_proposed_locked_a20_g50/test_scores_with_predictions.csv \
    --val-scored outputs_proposed_locked_a20_g50/val_scores_with_predictions.csv \
    --output-dir outputs_proposed_locked_rate_promoted
```

Produces `overall_metrics_proposed_valcal.csv` — the **headline row**
of the paper's main detection table.

### 6.3 — Train the comparison baselines  (~30 min total)

```bash
# 6.3a — RF-only and RF+Conformal (re-uses cascade scores, fast)
python3 rf_baseline_valcal.py \
    --cascade-dir outputs_proposed_locked_a20_g50 \
    --output-dir  outputs_rf_baseline_valcal

# 6.3b — Rate-rules-only ablation
python3 rate_rules_baseline_valcal.py \
    --data-dir   ../csv_CIC_IDS2017 \
    --output-dir outputs_rate_rules_baseline_valcal

# 6.3c — Anomaly-detector baselines (Isolation Forest, OCSVM, LSTM-AE)
python3 compare_anomaly_baselines_valcal.py \
    --data-dir   ../csv_CIC_IDS2017 \
    --output-dir outputs_baselines_temporal_by_file_valcal_iso
```

### 6.4 — (Optional) the (α × calibration-fraction) sweep  (~25 min)

The paper reports four operating points. Re-use the trained RF from
`6.1`:

```bash
python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.10 \
    --calibration-fraction 0.50 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a10_g50

python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.05 \
    --calibration-fraction 0.50 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a05_g50

python3 hybrid_cascade_splitcal_fastsnort.py \
    --data-dir ../csv_CIC_IDS2017 --alpha-escalate 0.20 \
    --calibration-fraction 0.25 \
    --rf-model outputs_proposed_locked_a20_g50/rf_anomaly.joblib \
    --output-dir outputs_proposed_locked_a20_g25
```

**Done when:** every directory below contains its `overall_metrics_*.csv`:

```
pipeline/outputs_proposed_locked_rate_promoted/
pipeline/outputs_rf_baseline_valcal/
pipeline/outputs_rate_rules_baseline_valcal/
pipeline/outputs_baselines_temporal_by_file_valcal_iso/
```

---

## §7  Map outputs to paper tables

| Paper table row | File |
|---|---|
| **Proposed (val-calibrated)** | `pipeline/outputs_proposed_locked_rate_promoted/overall_metrics_proposed_valcal.csv` |
| Proposed sweep — (α × g) grid | `pipeline/outputs_proposed_locked_a*_g*/overall_metrics.csv` |
| RF-only | `pipeline/outputs_rf_baseline_valcal/overall_metrics_rf_only.csv` |
| RF + Conformal | `pipeline/outputs_rf_baseline_valcal/overall_metrics_rf_conformal.csv` |
| Rate rules only | `pipeline/outputs_rate_rules_baseline_valcal/overall_metrics_rate_rules.csv` |
| Isolation Forest / OCSVM / LSTM-AE | `pipeline/outputs_baselines_temporal_by_file_valcal_iso/overall_metrics_baselines.csv` |
| Snort only | `snort/outputs_snort_eval_v4a_temporal/snort_signature_metrics.csv` |

The chosen operating threshold τ\* is recorded in
`pipeline/outputs_proposed_locked_rate_promoted/run_config.json`
(currently τ\* ≈ **0.0642566028502602**).

---

## §8  Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError: ../csv_CIC_IDS2017/Monday-...csv` | Files are not in the expected folder *or* names do not match exactly. The 8 CSVs must keep their original `*_WorkingHours*.pcap_ISCX.csv` names. |
| `Killed` mid-run on `compare_anomaly_baselines_valcal.py` | OOM. Raise the WSL mem cap (`§1`) or pass `--sample-benign 250000` to subsample. |
| Snort runner is extremely slow (< 100 kpps) | Pcap is on `/mnt/c/`. Move it to `~/pcap_CIC_IDS2017/` per `§2.2`. |
| Snort segfault on `Friday-WorkingHours.pcap` | Known issue with very large pcaps. Pass `--packet-limit 5000000` to `snort_runner.py`. |
| Snort prints `ERROR: can't find daq` | `ldconfig` not run after LibDAQ install. `sudo ldconfig` and retry. |
| `outputs_*/test_scores_with_predictions.csv` empty | Wrong split strategy or missing CSV. Ensure all 8 CSVs are present and `--split-strategy temporal_by_file` is set. |
| Reused RF (`--rf-model`) gives different numbers | Cache valid only under the same `--seed`, `--split-strategy`, and dataset. Retrain with the same flags. |
| `proposed_method_valcal.py` fails: missing column `gate_prob` | You pointed `--scored` at a baseline output, not the cascade output. Re-check paths from `§6.1`. |
| `ModuleNotFoundError: torch` (or sklearn, pandas, …) | Wrong Python environment. `which python3` and confirm it points at the env where you installed `requirements.txt`. |

---

## §9  Expected runtime

Reference: Intel i7-1165G7 (8 logical cores) / 16 GB RAM / NVMe SSD,
WSL2 with `memory=14GB`, pcaps on Linux fs per `§2.2`.

| Step | Time |
|---|---:|
| §1–§3   setup + venv install | 10 min |
| §4      Snort build | 30 min |
| §5.1    Snort replay (all pcaps) | 150 min |
| §5.2–§5.4  alert parsing + mapping | 15 min |
| §6.1    train cascade | 30 min |
| §6.2    val-cal headline | 1 min |
| §6.3    train baselines | 30 min |
| §6.4    (α × g) sweep *(optional)* | 25 min |
| **Total (without sweep)** | **~4 h 10 m** |
| **Total (with sweep)** | **~4 h 35 m** |

If pcaps are still on `/mnt/c/`, add 4–6 hours to `§5.1`.

---

## §10  Skipping Snort

If you cannot or do not want to install Snort and re-run pcaps, you
can still reproduce 80% of the paper using the **committed**
`snort_signature_predictions.csv` at
`snort/outputs_snort_eval_v4a_temporal/`. Skip `§4` and `§5` entirely
and jump to `§6`. The Snort row in `§7` will be served from the
committed CSV. Total drops to ~1.5 hours.

This is the best path for first-time reviewers — get the headline
numbers fast, then come back to `§4` and `§5` if a deeper audit is
needed.

---

## §11  Where to go next

After §6 finishes you have every detection number in the paper. From here:

- **Run the live SaaS prototype** (Docker Compose stack with Kafka +
  multi-tenant simulator + alert fan-out): see `MANUAL.md` starting at
  "Step 7 — Export the trained bundle". That path **does** require
  Docker.
- **Run the transport-overhead load test** (Kafka vs Direct-HTTP
  numbers reported as the SaaS architecture contribution): see
  `prototype/loadtest/README.md`. Also requires Docker.

The `gate.joblib` produced in `§6.1` is the input to both follow-ups,
so nothing here is wasted.
