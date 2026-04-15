from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_protocol(x):
    if pd.isna(x):
        return None
    return str(x).strip().upper()


def normalize_ip(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def maybe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def build_join_keys(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()

    if prefix == "snort":
        proto_col = "proto"
    else:
        proto_col = "protocol"

    out["src_ip_norm"] = out["src_ip"].map(normalize_ip)
    out["dst_ip_norm"] = out["dst_ip"].map(normalize_ip)
    out["src_port_norm"] = out["src_port"].map(maybe_int)
    out["dst_port_norm"] = out["dst_port"].map(maybe_int)
    out["proto_norm"] = out[proto_col].map(normalize_protocol)

    out["join_key"] = (
        out["src_ip_norm"].astype(str) + "|" +
        out["dst_ip_norm"].astype(str) + "|" +
        out["src_port_norm"].astype(str) + "|" +
        out["dst_port_norm"].astype(str) + "|" +
        out["proto_norm"].astype(str)
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snort-alerts", required=True)
    parser.add_argument("--rf-preds", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    snort_df = pd.read_csv(args.snort_alerts)
    rf_df = pd.read_csv(args.rf_preds)

    required_snort = {"src_ip", "dst_ip", "src_port", "dst_port", "proto"}
    required_rf = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol", "rf_pred"}

    missing_snort = required_snort - set(snort_df.columns)
    missing_rf = required_rf - set(rf_df.columns)

    if missing_snort:
        raise ValueError(f"Missing Snort columns: {sorted(missing_snort)}")
    if missing_rf:
        raise ValueError(f"Missing RF columns: {sorted(missing_rf)}")

    snort_df = build_join_keys(snort_df, "snort")
    rf_df = build_join_keys(rf_df, "rf")

    snort_hits = snort_df[["join_key"]].drop_duplicates().copy()
    snort_hits["snort_hit"] = 1

    merged = rf_df.merge(snort_hits, on="join_key", how="left")
    merged["snort_hit"] = merged["snort_hit"].fillna(0).astype(int)
    merged["rf_pred"] = merged["rf_pred"].fillna(0).astype(int)
    merged["hybrid_pred"] = ((merged["snort_hit"] == 1) | (merged["rf_pred"] == 1)).astype(int)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"[hybrid_merge] saved: {out}", flush=True)
    print(f"[hybrid_merge] rows={len(merged):,}", flush=True)
    print(f"[hybrid_merge] snort_hit_rows={(merged['snort_hit'] == 1).sum():,}", flush=True)
    print(f"[hybrid_merge] hybrid_attack_rows={(merged['hybrid_pred'] == 1).sum():,}", flush=True)


if __name__ == "__main__":
    main()
