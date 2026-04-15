# README_SNORT.md

## Snort module for NIDSAAS_EXPERIMENT

This folder contains the Snort-side workflow used in the experiment:

1. replay PCAP files with Snort 3  
2. parse `alert_fast.txt` into CSV  
3. filter alerts using experiment policy SID lists  
4. map filtered Snort alerts to the RF test split  
5. export `snort_signature_predictions.csv` for hybrid fusion

---

## Recommended structure

```text
snort/
├── parse_fast_alerts.py
├── snort_eval_fixed_v3.py
├── snort_runner.py
├── README_SNORT.md
└── rules/
    ├── community/
    │   ├── snort3-community.rules
    │   └── sid-msg.map
    ├── local/
    │   └── local.rules
    └── policy/
        ├── v4a_keep_sids.txt
        ├── v4b_keep_sids.txt
        └── policy_notes.md
```

### Meaning of each rules directory

#### `rules/community/`
Contains upstream Snort 3 rules downloaded from the official community rules package.

Use this for actual Snort replay.

#### `rules/local/`
Contains custom Snort runtime rules written for debugging or local experiments.

These are optional and not required for the final selected model.

#### `rules/policy/`
Contains **post-parse filtering policies** for the experiment.

These are **not** Snort `.rules` files.  
They are SID lists used after parsing alerts into CSV.

Example:
- `v4a_keep_sids.txt` = final selected policy
- `v4b_keep_sids.txt` = ablation / comparison policy

---

## Final selected policy

### Final signature policy
`v4a_keep_sids.txt`

Contents:
- `491`
- `553`

This is the policy used for the final deployment-oriented assisted hybrid.

---

## Required external paths

The current workflow assumes:

- Snort binary: `/opt/snort3/bin/snort`
- Snort Lua config: `/opt/snort3/etc/snort/snort.lua`

If your installation uses a different path, update the command arguments accordingly.

---

## Snort 3 installation summary

This project expects **Snort 3** with **LibDAQ** installed.

### 1) Install dependencies
```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  flex \
  bison \
  libpcap-dev \
  libpcre2-dev \
  zlib1g-dev \
  libdumbnet-dev \
  libluajit-5.1-dev \
  libssl-dev \
  libhwloc-dev \
  autotools-dev \
  libtool \
  liblzma-dev
```

Optional but recommended:
```bash
sudo apt install -y libhyperscan-dev
```

### 2) Build LibDAQ
```bash
cd ~
git clone https://github.com/snort3/libdaq.git
cd libdaq
./bootstrap
./configure --prefix=/usr/local/lib/daq_s3
make -j"$(nproc)"
sudo make install
```

Register the library path:
```bash
echo '/usr/local/lib/daq_s3/lib/' | sudo tee /etc/ld.so.conf.d/libdaq3.conf
sudo ldconfig
```

### 3) Build Snort 3
```bash
cd ~
git clone https://github.com/snort3/snort3.git
cd snort3

./configure_cmake.sh \
  --prefix=/opt/snort3 \
  --with-daq-includes=/usr/local/lib/daq_s3/include/ \
  --with-daq-libraries=/usr/local/lib/daq_s3/lib/

cd build
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

### 4) Check installation
```bash
/opt/snort3/bin/snort -V
```

---

## Community rules setup

Download and extract community rules:

```bash
cd ~
wget -O snort3-community-rules.tar.gz https://www.snort.org/downloads/community/snort3-community-rules.tar.gz
tar -xzf snort3-community-rules.tar.gz
```

Copy into the project:

```bash
mkdir -p snort/rules/community
cp ~/snort3-community-rules/snort3-community.rules snort/rules/community/
cp ~/snort3-community-rules/sid-msg.map snort/rules/community/
```

---

## Dataset placement

### CSV files
Put CIC-IDS2017 CSV files in:

```text
csv_CIC_IDS2017/
```

### PCAP files
Put PCAP files in:

```text
pcap_CIC_IDS2017/
```

For better replay speed in WSL, copy PCAPs into the Linux filesystem, for example:

```bash
mkdir -p ~/pcap_CIC_IDS2017
cp /mnt/c/path/to/pcap_CIC_IDS2017/*.pcap ~/pcap_CIC_IDS2017/
```

---

## Step-by-step run

### 1) Run Snort over the PCAPs
From the `snort/` folder:

```bash
python3 snort_runner.py \
  --snort-exe "/opt/snort3/bin/snort" \
  --pcap-dir "/home/$USER/pcap_CIC_IDS2017" \
  --rules "/opt/snort3/etc/snort/snort.lua" \
  --extra-rules "rules/community/snort3-community.rules" \
  --out-dir "/home/$USER/snort_outputs"
```

### 2) Parse `alert_fast.txt` into CSV
```bash
python3 parse_fast_alerts.py \
  --input-dir "/home/$USER/snort_outputs" \
  --output-csv "/home/$USER/snort_alerts.csv"
```

### 3) Filter by policy SID list
Example using final `v4a` policy:

```python
from pathlib import Path
import pandas as pd

def load_sid_list(path: str) -> set[int]:
    sids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sids.add(int(line))
    return sids

df = pd.read_csv("/home/$USER/snort_alerts.csv")
keep_sids = load_sid_list("rules/policy/v4a_keep_sids.txt")
filtered = df[df["sid"].isin(keep_sids)].copy()
filtered.to_csv("/home/$USER/snort_alerts_v4a_ftp_only.csv", index=False)
```

### 4) Map Snort alerts to the RF test split
```bash
python3 snort_eval_fixed_v3.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-alerts /home/$USER/snort_alerts_v4a_ftp_only.csv \
  --output-dir outputs_snort_eval_v4a \
  --ignore-time
```

Expected output:
- `outputs_snort_eval_v4a/snort_signature_predictions.csv`

This file is the Snort-side input for the final assisted hybrid.

---

## Final assisted hybrid link

The final selected model is:

**Hybrid-Assisted-Snort+RF-v4a-F1push**

It uses:
- Snort policy = `v4a`
- base threshold = `0.875`
- assist threshold = `0.55`

The final fusion step is run from the pipeline side with:
- `rf_predictions.csv`
- `snort_signature_predictions.csv`

---

## Export checklist

To let another person rerun the Snort side successfully, include:

- `snort_runner.py`
- `parse_fast_alerts.py`
- `snort_eval_fixed_v3.py`
- `README_SNORT.md`
- `rules/community/snort3-community.rules`
- `rules/community/sid-msg.map`
- `rules/policy/v4a_keep_sids.txt`
- `rules/policy/v4b_keep_sids.txt`
- `rules/policy/policy_notes.md`

Optional:
- parsed `snort_alerts.csv`
- filtered `snort_alerts_v4a_ftp_only.csv`
- `outputs_snort_eval_v4a/snort_signature_predictions.csv`

Sharing the pre-generated prediction file lets others reproduce the final hybrid without replaying all PCAPs.
