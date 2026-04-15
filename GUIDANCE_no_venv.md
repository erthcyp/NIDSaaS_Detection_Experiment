# GUIDANCE_no_venv.md

## End-to-end run guide without `.venv`

This version is for users who want to run the project using the system Python in WSL / Ubuntu, without creating a virtual environment.

Final selected model:
- **Hybrid-Assisted-Snort+RF-v4a-F1push**
- Snort policy: **v4a**
- Base threshold: **0.875**
- Assist threshold: **0.55**

---

## 0) When to use this guide

Use this guide if:
- you do not want to use `.venv`
- your old setup already worked with system Python
- you are okay installing packages with `--break-system-packages`

Do **not** use this approach if you want the cleanest isolation.  
For that, use the venv-based guide instead.

---

## 1) Open the project root

Example:

```bash
cd /mnt/c/Users/<your_user>/Downloads/NIDSaaS_Experiment
```

Check that you are in the correct place:

```bash
pwd
ls
```

You should see:
- `pipeline`
- `snort`
- `csv_CIC_IDS2017`
- `pcap_CIC_IDS2017`
- `requirements.txt`

---

## 2) Check Python version

```bash
python3 --version
```

Recommended:
- Python 3.12 on WSL Ubuntu

---

## 3) Install Python packages without `.venv`

Because Ubuntu / Debian may block normal pip installs under PEP 668, use:

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

If you want to upgrade pip first:

```bash
python3 -m pip install --break-system-packages -U pip
python3 -m pip install --break-system-packages -r requirements.txt
```

### Quick sanity check
```bash
python3 -c "import pandas, sklearn, matplotlib; print('python packages ok')"
```

If your code path still imports LSTM modules, install PyTorch too:

```bash
python3 -m pip install --break-system-packages torch torchvision torchaudio
```

---

## 4) Install Snort 3

If Snort is already installed and this works:

```bash
/opt/snort3/bin/snort -V
```

skip to section 5.

Expected paths:
- Snort binary: `/opt/snort3/bin/snort`
- Snort config: `/opt/snort3/etc/snort/snort.lua`

### 4.1 Install system packages
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

### 4.2 Build LibDAQ
```bash
cd ~
git clone https://github.com/snort3/libdaq.git
cd libdaq
./bootstrap
./configure --prefix=/usr/local/lib/daq_s3
make -j"$(nproc)"
sudo make install
```

Register library path:
```bash
echo '/usr/local/lib/daq_s3/lib/' | sudo tee /etc/ld.so.conf.d/libdaq3.conf
sudo ldconfig
```

### 4.3 Build Snort 3
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

### 4.4 Check Snort
```bash
/opt/snort3/bin/snort -V
```

---

## 5) Prepare datasets

### 5.1 CSV files
Put CIC-IDS2017 CSV files in:

```text
csv_CIC_IDS2017/
```

### 5.2 PCAP files
Put PCAP files in:

```text
pcap_CIC_IDS2017/
```

### 5.3 Recommended for WSL users
For faster replay, copy PCAPs into the Linux filesystem:

```bash
mkdir -p ~/pcap_CIC_IDS2017
ls /mnt/c/Users/user/Downloads/NIDSaaS_Experiment/pcap_CIC_IDS2017
cp /mnt/c/Users/user/Downloads/NIDSaaS_Experiment/pcap_CIC_IDS2017/*.pcap ~/pcap_CIC_IDS2017/
ls -lh ~/pcap_CIC_IDS2017
```

Expected destination:

```text
/home/<your_user>/pcap_CIC_IDS2017
```

Do **not** rely on relative copy commands unless you are sure you are already in the project root.

---

## 6) Run RF pipeline

From the project root:

```bash
python3 pipeline/main_detection.py \
  --data-dir csv_CIC_IDS2017 \
  --output-dir pipeline/outputs_detection_v2 \
  --mode rf
```

Expected output:

```text
pipeline/outputs_detection_v2/rf_predictions.csv
```

Check:

```bash
head -5 pipeline/outputs_detection_v2/rf_predictions.csv
```

---

## 7) Run Snort replay

Go to the Snort folder:

```bash
cd snort
```

Replay PCAP files:

```bash
python3 snort_runner.py \
  --snort-exe "/opt/snort3/bin/snort" \
  --pcap-dir "/home/$USER/pcap_CIC_IDS2017" \
  --rules "/opt/snort3/etc/snort/snort.lua" \
  --extra-rules "rules/community/snort3-community.rules" \
  --out-dir "/home/$USER/snort_outputs"
```

---

## 8) Parse Snort alerts

```bash
python3 parse_fast_alerts.py \
  --input-dir "/home/$USER/snort_outputs" \
  --output-csv "/home/$USER/snort_alerts.csv"
```

Expected output:

```text
/home/<your_user>/snort_alerts.csv
```

---

## 9) Filter with final policy `v4a`

```bash
python3 filter_policy_snort.py \
  --input-csv /home/$USER/snort_alerts.csv \
  --policy-file rules/policy/v4a_keep_sids.txt \
  --output-csv /home/$USER/snort_alerts_v4a_ftp_only.csv
```

Expected output:

```text
/home/<your_user>/snort_alerts_v4a_ftp_only.csv
```

---

## 10) Map Snort alerts to RF test rows

```bash
python3 snort_eval_fixed_v3.py \
  --data-dir ../csv_CIC_IDS2017 \
  --snort-alerts /home/$USER/snort_alerts_v4a_ftp_only.csv \
  --output-dir outputs_snort_eval_v4a \
  --ignore-time
```

Expected output:

```text
snort/outputs_snort_eval_v4a/snort_signature_predictions.csv
```

---

## 11) Run final assisted hybrid

Go back to the project root:

```bash
cd ..
```

Run the final selected model:

```bash
python3 pipeline/hybrid_assisted_from_snort_rf.py \
  --rf-predictions pipeline/outputs_detection_v2/rf_predictions.csv \
  --snort-predictions snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir pipeline/outputs_hybrid_assisted_v4a \
  --paper-model-name Hybrid-Assisted-Snort+RF-v4a-F1push \
  --base-threshold 0.875 \
  --assist-threshold 0.55
```

Expected outputs:

```text
pipeline/outputs_hybrid_assisted_v4a/assisted_hybrid_metrics_comparison.csv
pipeline/outputs_hybrid_assisted_v4a/assisted_hybrid_predictions.csv
```

Check final metrics:

```bash
cat pipeline/outputs_hybrid_assisted_v4a/assisted_hybrid_metrics_comparison.csv
```

---

## 12) Minimal quick rerun path

If these already exist:
- `pipeline/outputs_detection_v2/rf_predictions.csv`
- `snort/outputs_snort_eval_v4a/snort_signature_predictions.csv`

then you can skip RF and Snort rerun and directly run:

```bash
python3 pipeline/hybrid_assisted_from_snort_rf.py \
  --rf-predictions pipeline/outputs_detection_v2/rf_predictions.csv \
  --snort-predictions snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
  --output-dir pipeline/outputs_hybrid_assisted_v4a \
  --paper-model-name Hybrid-Assisted-Snort+RF-v4a-F1push \
  --base-threshold 0.875 \
  --assist-threshold 0.55
```

---

## 13) Common failure points

### Problem: `externally-managed-environment`
Fix:
- use `python3 -m pip install --break-system-packages ...`

### Problem: `matplotlib` / `pandas` / `sklearn` missing
Fix:
- rerun:
```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

### Problem: `cd pipeline` fails
Fix:
- you are not in the project root
- use full path to the project root first

### Problem: `cp pcap_CIC_IDS2017/*.pcap ...` fails
Fix:
- use full project path
- do not assume your current directory is the project root

### Problem: `rf_predictions.csv` missing
Fix:
- rerun section 6
- make sure `main_detection.py` exports RF predictions

### Problem: `snort_signature_predictions.csv` missing
Fix:
- rerun sections 7 to 10

---

## 14) Final note

For the cleanest environment, `.venv` is still better.

But if your goal is simply to:
- reset your machine
- reinstall quickly
- rerun the project the same way as before

then this no-venv workflow is acceptable and closer to your previous setup.
