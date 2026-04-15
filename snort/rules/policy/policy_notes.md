# Snort Alert Policy Notes

This directory stores **post-parse alert filtering policies** used in the experiment.

These are **not** Snort runtime `.rules` files.
Instead, they are lists of Snort SIDs used to filter parsed alerts **after**
running Snort and converting `alert_fast.txt` into CSV.

## Policy types

### v4a
FTP-focused conservative policy.

Keep SIDs:
- 491
- 553

Interpretation:
- `491` = FTP bad login indicator
- `553` = FTP anonymous login attempt

This policy was selected as the final signature policy for the deployment-oriented
assisted hybrid because it provided the best operational trade-off.

### v4b
FTP-focused policy plus targeted indicators.

Keep SIDs:
- 491
- 553
- 30788
- 1546

Interpretation:
- `30788` = TLS / heartbeat-related indicator
- `1546` = targeted web application indicator

This policy was kept as an ablation / comparison policy.

## Recommended usage

1. Run Snort using community rules
2. Parse `alert_fast.txt` to `snort_alerts.csv`
3. Load one of these SID lists
4. Filter alerts by SID
5. Evaluate and merge with RF outputs

## Example Python usage

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

df = pd.read_csv("/home/erthcyp/snort_alerts.csv")
keep_sids = load_sid_list("snort/rules/policy/v4a_keep_sids.txt")
filtered = df[df["sid"].isin(keep_sids)].copy()
```
