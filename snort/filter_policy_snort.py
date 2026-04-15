from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[filter_policy_snort] {msg}", flush=True)


def load_sid_list(path: str) -> set[int]:
    sids: set[int] = set()

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            sids.add(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid SID in policy file {path!r}: {line!r}") from exc

    if not sids:
        raise ValueError(f"No valid SIDs found in policy file: {path}")

    return sids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter parsed Snort alerts CSV using a policy SID list."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to parsed Snort alerts CSV, e.g. /home/user/snort_alerts.csv",
    )
    parser.add_argument(
        "--policy-file",
        required=True,
        help="Path to SID policy file, e.g. snort/rules/policy/v4a_keep_sids.txt",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save filtered alerts CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    policy_path = Path(args.policy_file)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    log(f"loading alerts: {input_path}")
    df = pd.read_csv(input_path)

    if "sid" not in df.columns:
        raise ValueError("Input CSV must contain a 'sid' column.")

    keep_sids = load_sid_list(str(policy_path))
    log(f"loaded {len(keep_sids)} SIDs from policy: {policy_path}")

    filtered = df[df["sid"].isin(keep_sids)].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    log(f"input rows   = {len(df):,}")
    log(f"output rows  = {len(filtered):,}")
    log(f"saved to     = {output_path}")

    if len(filtered) > 0:
        try:
            print("\nTop SID/message counts:")
            print(filtered[["sid", "message"]].value_counts().head(20).to_string())
        except Exception:
            pass


if __name__ == "__main__":
    main()
