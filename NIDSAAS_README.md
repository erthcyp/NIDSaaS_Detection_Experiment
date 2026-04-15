# NIDSAAS_EXPERIMENT

Final experiment package for the **multi-tenant NIDSaaS detection study** using:

- **RF anomaly detector**
- **Snort signature detection**
- **Hybrid-Assisted-Snort+RF-v4a-F1push** as the final selected model

---

## 1) Recommended project structure

Your current structure is already good enough for export and rerun.

```text
NIDSAAS_EXPERIMENT/
├── csv_CIC_IDS2017/
├── pcap_CIC_IDS2017/
├── pipeline/
│   ├── config.py
│   ├── features.py
│   ├── hybrid_assisted_from_snort_rf.py
│   ├── load_data.py
│   ├── main_detection.py
│   ├── metrics.py
│   ├── rf_anomaly.py
│   └── utils.py
└── snort/
    ├── hybrid_merge.py
    ├── parse_fast_alerts.py
    ├── README_SNORT.md
    └── snort_runner.py
```

### Suggested small improvements
For cleaner export, consider adding these files too:

```text
NIDSAAS_EXPERIMENT/
├── README.md
├── requirements.txt
├── csv_CIC_IDS2017/
├── pcap_CIC_IDS2017/
├── pipeline/
│   ├── config.py
│   ├── features.py
│   ├── hybrid_assisted_from_snort_rf.py
│   ├── hybrid_score_boost_from_snort_rf.py
│   ├── hybrid_three_zone_from_snort_rf.py
│   ├── load_data.py
│   ├── main_detection.py
│   ├── metrics.py
│   ├── rf_anomaly.py
│   └── utils.py
└── snort/
    ├── parse_fast_alerts.py
    ├── snort_eval_fixed_v3.py
    ├── snort_runner.py
    ├── README_SNORT.md
    └── rules/
        ├── snort3-community.rules
        └── policy_notes.md
```

---

## 2) Final selected model

### Final deployment-oriented model
**Hybrid-Assisted-Snort+RF-v4a-F1push**

### Final operating point
- Snort policy: **v4a**
- Hybrid type: **assisted**
- Base threshold: **0.875**
- Assist threshold: **0.55**

### Why this was selected
Compared with the RF-only baseline, the final assisted hybrid:
- improves **accuracy**
- improves **precision**
- improves **F1-score**
- reduces **false alarm rate**
- keeps recall close to RF

This makes it more suitable for **multi-tenant NIDSaaS deployment**, where reducing false alarms and alert fatigue is important.

---

## 3) Python environment

Tested with Python 3.12 in WSL Ubuntu.

Create a virtual environment if desired:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -U pip
pip install pandas numpy scipy scikit-learn matplotlib joblib
```

If your code path still imports the LSTM module at import time, install PyTorch too:

```bash
pip install torch torchvision torchaudio
```

### Optional requirements.txt
You can create a simple `requirements.txt` like this:

```txt
pandas
numpy
scipy
scikit-learn
matplotlib
joblib
torch
torchvision
torchaudio
```

Then install with:

```bash
pip install -r requirements.txt
```

---

## 4) Snort 3 installation guide (WSL Ubuntu / Ubuntu Linux)

This project uses **Snort 3** built from source.

### 4.1 Install build dependencies
Example Ubuntu/WSL setup:

```bash
sudo apt update

sudo apt install -y   build-essential   cmake   git   pkg-config   flex   bison   libpcap-dev   libpcre2-dev   zlib1g-dev   libdumbnet-dev   libluajit-5.1-dev   libssl-dev   libhwloc-dev   autotools-dev   libtool   liblzma-dev
```

### 4.2 Optional but recommended dependency
Install **Hyperscan** support if available in your environment:

```bash
sudo apt install -y libhyperscan-dev
```

If your distro does not provide it, you can build Hyperscan separately or omit it.

### 4.3 Build and install LibDAQ
```bash
cd ~
git clone https://github.com/snort3/libdaq.git
cd libdaq
./bootstrap
./configure --prefix=/usr/local/lib/daq_s3
make -j"$(nproc)"
sudo make install
```

Register the shared library path:

```bash
echo '/usr/local/lib/daq_s3/lib/' | sudo tee /etc/ld.so.conf.d/libdaq3.conf
sudo ldconfig
```

### 4.4 Build and install Snort 3
```bash
cd ~
git clone https://github.com/snort3/snort3.git
cd snort3

./configure_cmake.sh   --prefix=/opt/snort3   --with-daq-includes=/usr/local/lib/daq_s3/include/   --with-daq-libraries=/usr/local/lib/daq_s3/lib/

cd build
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

### 4.5 Check the installation
```bash
/opt/snort3/bin/snort -V
```

Expected output should show the installed Snort version and enabled libraries.

---

## 5) Snort rules

### Community rules
Download the Snort 3 Community Rules and extract them.

Example:

```bash
cd ~
wget -O snort3-community-rules.tar.gz https://www.snort.org/downloads/community/snort3-community-rules.tar.gz
tar -xzf snort3-community-rules.tar.gz
```

This should give you a directory containing:
- `snort3-community.rules`
- `sid-msg.map`

### Final project policy
This project uses a **filtered Snort alert policy** derived from parsed alerts, with the final selected policy being **v4a**.

For reproducibility, keep:
- raw parsed alerts CSV
- filtered alerts CSV for v4a
- `snort_signature_predictions.csv`

---

## 6) Dataset placement

### CSV dataset
Put CIC-IDS2017 CSV files under:

```text
csv_CIC_IDS2017/
```

### PCAP dataset
Put PCAP files under:

```text
pcap_CIC_IDS2017/
```

For faster Snort replay in WSL, copy PCAP files into the Linux filesystem instead of reading them from `/mnt/c/...`.

Example:

```bash
mkdir -p ~/pcap_CIC_IDS2017
cp /mnt/c/path/to/pcap_CIC_IDS2017/*.pcap ~/pcap_CIC_IDS2017/
```

---

## 7) Run the RF pipeline

Go to the pipeline folder:

```bash
cd pipeline
```

Run RF experiment:

```bash
python3 main_detection.py   --data-dir ../csv_CIC_IDS2017   --output-dir outputs_detection_v2   --mode rf
```

Important output:
- `outputs_detection_v2/rf_predictions.csv`

This file is required for the final hybrid run.

---

## 8) Run Snort on PCAP files

Go to the snort folder:

```bash
cd ../snort
```

### 8.1 Replay PCAPs with Snort
Example:

```bash
python3 snort_runner.py   --snort-exe "/opt/snort3/bin/snort"   --pcap-dir "/home/$USER/pcap_CIC_IDS2017"   --rules "/opt/snort3/etc/snort/snort.lua"   --extra-rules "/home/$USER/snort-src/snort3-community-rules/snort3-community.rules"   --out-dir "/home/$USER/snort_outputs"
```

### 8.2 Parse Snort alerts
```bash
python3 parse_fast_alerts.py   --input-dir "/home/$USER/snort_outputs"   --output-csv "/home/$USER/snort_alerts.csv"
```

### 8.3 Evaluate filtered Snort alerts against the test split
Use the final `v4a` policy file and the fixed evaluator:

```bash
python3 snort_eval_fixed_v3.py   --data-dir ../csv_CIC_IDS2017   --snort-alerts /home/$USER/snort_alerts_v4a_ftp_only.csv   --output-dir outputs_snort_eval_v4a   --ignore-time
```

Important output:
- `outputs_snort_eval_v4a/snort_signature_predictions.csv`

---

## 9) Run the final assisted hybrid

Go back to the pipeline folder:

```bash
cd ../pipeline
```

Run the final selected hybrid:

```bash
python3 hybrid_assisted_from_snort_rf.py   --rf-predictions outputs_detection_v2/rf_predictions.csv   --snort-predictions ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv   --output-dir outputs_hybrid_assisted_v4a   --paper-model-name Hybrid-Assisted-Snort+RF-v4a-F1push   --base-threshold 0.875   --assist-threshold 0.55
```

Important output:
- `outputs_hybrid_assisted_v4a/assisted_hybrid_metrics_comparison.csv`
- `outputs_hybrid_assisted_v4a/assisted_hybrid_predictions.csv`

---

## 10) Final files required for full reproduction

### Pipeline side
- `pipeline/load_data.py`
- `pipeline/main_detection.py`
- `pipeline/rf_anomaly.py`
- `pipeline/features.py`
- `pipeline/config.py`
- `pipeline/utils.py`
- `pipeline/metrics.py`
- `pipeline/hybrid_assisted_from_snort_rf.py`

### Snort side
- `snort/snort_runner.py`
- `snort/parse_fast_alerts.py`
- `snort/snort_eval_fixed_v3.py`
- `snort/rules/snort3-community.rules`
- Snort binary and config:
  - `/opt/snort3/bin/snort`
  - `/opt/snort3/etc/snort/snort.lua`

### Generated outputs
- `pipeline/outputs_detection_v2/rf_predictions.csv`
- `snort/outputs_snort_eval_v4a/snort_signature_predictions.csv`
- `pipeline/outputs_hybrid_assisted_v4a/assisted_hybrid_metrics_comparison.csv`

---

## 11) Notes for exporting this project to others

If you want another person to rerun everything successfully, export:

1. Source code folders:
   - `pipeline/`
   - `snort/`

2. This README:
   - `README.md`

3. Python dependency list:
   - `requirements.txt`

4. A short note about Snort installation path:
   - Snort binary expected at `/opt/snort3/bin/snort`
   - Config expected at `/opt/snort3/etc/snort/snort.lua`

5. Dataset placement instructions:
   - where to place CSV files
   - where to place PCAP files

6. Optional pre-generated outputs:
   - `rf_predictions.csv`
   - `snort_signature_predictions.csv`
   - final metric CSVs

If you also share the pre-generated outputs, another user can reproduce the final tables and plots without rerunning the full Snort replay.

---

## 12) Recommended final packaging

```text
NIDSAAS_EXPERIMENT/
├── README.md
├── requirements.txt
├── csv_CIC_IDS2017/
├── pcap_CIC_IDS2017/
├── pipeline/
├── snort/
└── outputs/
```

A clean `outputs/` folder may include:
- RF outputs
- Snort evaluation outputs
- final assisted hybrid outputs
- final figures

---

## 13) Final recommendation

For the paper and for practical deployment-oriented experiments, use:

**Hybrid-Assisted-Snort+RF-v4a-F1push**

because it gives the best operational trade-off between:
- detection effectiveness
- precision
- false alarm control
- suitability for multi-tenant NIDSaaS deployment
