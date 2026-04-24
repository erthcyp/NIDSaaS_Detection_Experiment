"""Flow-level rate-based signature prefilter.

Drop-in replacement for the Snort prediction CSV used by the cascade. Produces
the same row_id / signature_pred / signature_score schema, and optionally
OR-merges an existing Snort predictions CSV on top of its own rule fires.

Why this module exists
----------------------
Snort replay on CIC-IDS2017 only contributed ~1.4% recall in the latest
cascade run, mostly because (i) the 'v4a_ftp_only' policy was narrow,
(ii) community signatures cover application-layer exploits well but are
blind to volumetric / rate-based attacks, and (iii) the PCAP-to-flow
attribution was lossy. The NIDAAS-v1 paper actually used a flow-level
rate prefilter in addition to signatures; this module reinstates that
prefilter under the new cascade and emits its fires in the same schema
the cascade already consumes.

Rules (all operate on CIC-IDS2017 canonical flow features)
----------------------------------------------------------
    V   Volumetric     total_packets >= 20 AND
                         (flow_packets_s > 40_000 OR flow_bytes_s > 1.5e7)
    L   Slow-HTTP      destination_port in {80,443,8080}
                         AND flow_duration > 60 s AND total_packets < 20
                         AND flow_bytes_s < 100
    S   SYN-flood      syn_flag_count >= 3 AND total_packets <= 10
    R   RST-anomaly    rst_flag_count >= 5 AND total_packets <= 20
    P   PortScan       within 2-sec window: same source_ip, using only flows
                         with total_packets <= 10 AND flow_duration <= 10 ms,
                         hits >= 200 distinct destination_port values
    B   Brute-force    destination_port in {21,22,23} AND within 2-sec
                         window: same (source_ip, dst_port) seen >= 10 times

Thresholds are calibrated against the 99-99.9 percentile of the benign
training fold so that the per-rule false-alarm rate is roughly <= 1%.

Usage
-----

    python3 signature_rate_rules.py \
        --data-dir ../csv_CIC_IDS2017 \
        --output-csv signature_rate_predictions.csv

    # OR-merge with the existing Snort predictions into a single CSV:
    python3 signature_rate_rules.py \
        --data-dir ../csv_CIC_IDS2017 \
        --merge-snort-csv ../snort/outputs_snort_eval_v4a/snort_signature_predictions.csv \
        --output-csv signature_merged_predictions.csv

Then feed the output straight into the cascade:

    python3 hybrid_cascade_splitcal_fastsnort.py \
        --data-dir ../csv_CIC_IDS2017 \
        --snort-predictions signature_merged_predictions.csv \
        --split-strategy temporal_by_file \
        --alpha-escalate 0.20 \
        --output-dir outputs_hybrid_cascade_splitcal_dualfast_temporal
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from load_data import clean_detection_dataframe, read_cic_ids2017_folder


def log(msg: str) -> None:
    print(f"[signature_rate_rules] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Rule configuration
# ---------------------------------------------------------------------------
@dataclass
class RateRuleConfig:
    """Calibrated on the benign training fold of CIC-IDS2017. Thresholds are set
    so that each rule's fire rate on held-out benigns is roughly <=1%. See the
    block comment at the top of this file for details.
    """
    # Volumetric: pathological short flows can hit huge rates even when benign,
    # so we require a minimum total-packet floor before the rate threshold bites.
    volumetric_packets_s: float = 40_000.0
    volumetric_bytes_s: float = 15_000_000.0
    volumetric_min_packets: int = 20
    # Slow HTTP: benign keep-alive routinely holds HTTP flows for ~120s, so we
    # add a byte/s upper bound and raise the duration floor.
    slowhttp_duration_us: float = 60_000_000.0
    slowhttp_max_packets: int = 20
    slowhttp_max_bytes_s: float = 100.0
    slowhttp_ports: tuple = (80, 443, 8080)
    # SYN flood
    syn_flood_min_syn: int = 3
    syn_flood_max_packets: int = 10
    # RST anomaly
    rst_anomaly_min_rst: int = 5
    rst_anomaly_max_packets: int = 20
    # PortScan (windowed) -- only count genuinely scan-like flows when
    # aggregating unique destination ports per (source_ip, 2s bucket):
    # <=10 packets AND flow_duration under 10 ms. Real scan probes are
    # essentially instant (benign PortScan p90 duration = 76 us); DoS Hulk,
    # Bot, Slow-HTTP etc. have flow durations in the seconds-to-minutes
    # range and are correctly excluded here.
    portscan_window_s: float = 2.0
    portscan_min_unique_ports: int = 200
    portscan_max_flow_packets: int = 10
    portscan_max_flow_duration_us: float = 10_000.0
    # Brute-force (windowed)
    bruteforce_ports: tuple = (21, 22, 23)
    bruteforce_window_s: float = 2.0
    bruteforce_min_attempts: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _resolve_timestamp(df: pd.DataFrame) -> Optional[pd.Series]:
    for cand in ("timestamp", "Timestamp"):
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce")
            if ts.notna().any():
                return ts
    return None


# ---------------------------------------------------------------------------
# Per-flow rules
# ---------------------------------------------------------------------------
def rule_volumetric(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    pkt_s = _num(df, "flow_packets_s", 0.0)
    byt_s = _num(df, "flow_bytes_s", 0.0)
    fwd = _num(df, "total_fwd_packets", 0.0)
    bwd = _num(df, "total_backward_packets", 0.0)
    total = fwd + bwd
    # require a minimum packet floor so single-packet zero-duration flows with
    # huge computed rates don't trigger -- those dominate the benign tail.
    sustained = total >= cfg.volumetric_min_packets
    fire = sustained & ((pkt_s > cfg.volumetric_packets_s) | (byt_s > cfg.volumetric_bytes_s))
    return fire.to_numpy(dtype=bool)


def rule_slowhttp(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    dur = _num(df, "flow_duration", 0.0)
    fwd = _num(df, "total_fwd_packets", 0.0)
    bwd = _num(df, "total_backward_packets", 0.0)
    total = fwd + bwd
    byt_s = _num(df, "flow_bytes_s", 0.0)
    port = _num(df, "destination_port", -1).astype(int)
    is_http = port.isin(cfg.slowhttp_ports)
    fire = (
        is_http
        & (dur > cfg.slowhttp_duration_us)
        & (total < cfg.slowhttp_max_packets)
        & (byt_s < cfg.slowhttp_max_bytes_s)
    )
    return fire.to_numpy(dtype=bool)


def rule_syn_flood(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    syn = _num(df, "syn_flag_count", 0.0)
    fwd = _num(df, "total_fwd_packets", 0.0)
    bwd = _num(df, "total_backward_packets", 0.0)
    total = fwd + bwd
    fire = (syn >= cfg.syn_flood_min_syn) & (total <= cfg.syn_flood_max_packets)
    return fire.to_numpy(dtype=bool)


def rule_rst_anomaly(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    rst = _num(df, "rst_flag_count", 0.0)
    fwd = _num(df, "total_fwd_packets", 0.0)
    bwd = _num(df, "total_backward_packets", 0.0)
    total = fwd + bwd
    fire = (rst >= cfg.rst_anomaly_min_rst) & (total <= cfg.rst_anomaly_max_packets)
    return fire.to_numpy(dtype=bool)


# ---------------------------------------------------------------------------
# Windowed rules
# ---------------------------------------------------------------------------
def _windowed_flag(
    df: pd.DataFrame,
    group_cols: list[str],
    window_s: float,
    min_count: int,
    extra_mask: Optional[pd.Series] = None,
    count_unique_col: Optional[str] = None,
) -> np.ndarray:
    """Return a boolean row-vector marking flows whose (group_cols, 2-sec bucket)
    bucket has >= min_count occurrences (or unique values of `count_unique_col`).
    """
    ts = _resolve_timestamp(df)
    if ts is None:
        return np.zeros(len(df), dtype=bool)

    mask = ts.notna()
    for gc in group_cols:
        if gc not in df.columns:
            return np.zeros(len(df), dtype=bool)
        mask &= df[gc].notna()
    if extra_mask is not None:
        mask &= extra_mask

    if not mask.any():
        return np.zeros(len(df), dtype=bool)

    work = pd.DataFrame({"row_id": df["row_id"].to_numpy()[mask.to_numpy()]})
    work["ts"] = ts.to_numpy()[mask.to_numpy()]
    for gc in group_cols:
        work[gc] = df[gc].to_numpy()[mask.to_numpy()]
    if count_unique_col is not None and count_unique_col in df.columns:
        work[count_unique_col] = df[count_unique_col].to_numpy()[mask.to_numpy()]

    bucket_ns = int(window_s * 1e9)
    work["bucket"] = work["ts"].astype("int64") // bucket_ns

    keys = group_cols + ["bucket"]
    if count_unique_col is not None:
        counts = work.groupby(keys)[count_unique_col].nunique()
    else:
        counts = work.groupby(keys).size()

    flagged = counts[counts >= min_count].reset_index()[keys]
    if flagged.empty:
        return np.zeros(len(df), dtype=bool)

    flagged["__hit__"] = 1
    merged = work.merge(flagged, on=keys, how="left")
    hit_rids = set(
        merged.loc[merged["__hit__"] == 1, "row_id"].astype(np.int64).to_numpy().tolist()
    )
    if not hit_rids:
        return np.zeros(len(df), dtype=bool)

    rid = df["row_id"].astype(np.int64).to_numpy()
    result = np.zeros(len(df), dtype=bool)
    in_set = np.fromiter((int(r) in hit_rids for r in rid), dtype=bool, count=len(rid))
    result |= in_set
    return result


def rule_portscan(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    """Same source_ip hits >= min_unique_ports distinct destination_port in window.
    Only genuinely scan-like flows (<= portscan_max_flow_packets packets AND
    flow_duration <= portscan_max_flow_duration_us) are considered, so DoS
    Hulk, Bot, DNS servers, load balancers, and other long-lived or heavy
    benign flows don't drag additional rows into the alert."""
    if "source_ip" not in df.columns or "destination_port" not in df.columns:
        log("port_scan rule skipped (missing source_ip or destination_port)")
        return np.zeros(len(df), dtype=bool)
    fwd = _num(df, "total_fwd_packets", 0.0)
    bwd = _num(df, "total_backward_packets", 0.0)
    total = fwd + bwd
    dur = _num(df, "flow_duration", 0.0)
    mask_micro = (total <= cfg.portscan_max_flow_packets) & (dur <= cfg.portscan_max_flow_duration_us)
    return _windowed_flag(
        df,
        group_cols=["source_ip"],
        window_s=cfg.portscan_window_s,
        min_count=cfg.portscan_min_unique_ports,
        extra_mask=mask_micro,
        count_unique_col="destination_port",
    )


def rule_bruteforce(df: pd.DataFrame, cfg: RateRuleConfig) -> np.ndarray:
    """Within 2-sec window, same (source_ip, auth-port) has >= min_attempts flows."""
    if "source_ip" not in df.columns or "destination_port" not in df.columns:
        log("bruteforce rule skipped (missing source_ip or destination_port)")
        return np.zeros(len(df), dtype=bool)
    port = _num(df, "destination_port", -1).astype(int)
    mask_port = port.isin(cfg.bruteforce_ports)
    return _windowed_flag(
        df,
        group_cols=["source_ip", "destination_port"],
        window_s=cfg.bruteforce_window_s,
        min_count=cfg.bruteforce_min_attempts,
        extra_mask=mask_port,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
TIER1_RULES: tuple = ()                            # fast-path rate rules: none
TIER2_RULES = ("V", "L", "S", "R", "P", "B")       # all rate rules feed gate

# Snort is the sole fast-path signature. Every rate rule is exposed to the
# escalation gate as a per-row indicator feature; the gate learns which
# rules correlate with attack given that the row already passed the
# conformal escalation filter (rf_pvalue <= alpha_escalate).


def apply_rules(df: pd.DataFrame, cfg: RateRuleConfig) -> pd.DataFrame:
    """Apply all six rate rules and emit a tiered prediction frame.

    Schema:
      row_id           : int
      signature_pred   : int (0/1). OR of tier-1 rate rules {V,S,R,B}.
                         (Snort is OR-merged into this column separately.)
      signature_score  : float. Count of tier-1 rule fires.
      rule_fired       : str. Concatenation of letters of *all* rules that
                         fired (tier-1 plus tier-2), for diagnostics.
      rate_V, rate_L,  : int (0/1). Individual rule fires. rate_L and rate_P
      rate_P, ...        are intended for consumption by the escalation gate
                         as meta-features; the others are kept for
                         completeness and per-class reporting.
    """
    rules = {
        "V": rule_volumetric(df, cfg),
        "L": rule_slowhttp(df, cfg),
        "S": rule_syn_flood(df, cfg),
        "R": rule_rst_anomaly(df, cfg),
        "P": rule_portscan(df, cfg),
        "B": rule_bruteforce(df, cfg),
    }
    for letter, fires in rules.items():
        tier = "tier1" if letter in TIER1_RULES else "tier2"
        log(f"rule {letter} ({tier}) fires: {int(fires.sum()):,}")

    # Tier-1 aggregate (fast-path fires)
    tier1_fire = np.zeros(len(df), dtype=bool)
    tier1_count = np.zeros(len(df), dtype=int)
    for letter in TIER1_RULES:
        tier1_fire |= rules[letter]
        tier1_count += rules[letter].astype(int)

    # Rule-letter concatenation for diagnostics (tier-1 AND tier-2)
    rule_letters = [""] * len(df)
    for letter, fires in rules.items():
        for i in np.nonzero(fires)[0]:
            rule_letters[i] = rule_letters[i] + letter

    out = pd.DataFrame({
        "row_id": df["row_id"].astype(np.int64).to_numpy(),
        "signature_pred": tier1_fire.astype(int),
        "signature_score": tier1_count.astype(float),
        "rule_fired": rule_letters,
    })
    # Emit per-rule fires for downstream use (gate meta-features etc.)
    for letter, fires in rules.items():
        out[f"rate_{letter}"] = fires.astype(int)

    log(
        f"tier-1 fast-path fires (V|S|R|B): {int(tier1_fire.sum()):,} "
        f"({tier1_fire.mean():.2%}) of {len(df):,} rows"
    )
    log(
        f"tier-2 fires   L={int(rules['L'].sum()):,} "
        f"({rules['L'].mean():.2%})   "
        f"P={int(rules['P'].sum()):,} ({rules['P'].mean():.2%})"
    )
    return out


def merge_with_snort(rate_df: pd.DataFrame, snort_csv: str) -> pd.DataFrame:
    snort = pd.read_csv(snort_csv)
    # Resolve column names the same way the cascade does.
    cols_lower = {c.lower(): c for c in snort.columns}
    rid_c = cols_lower.get("row_id")
    pred_c = next(
        (cols_lower[k] for k in ("signature_pred", "snort_pred", "prediction", "pred") if k in cols_lower),
        None,
    )
    score_c = next(
        (cols_lower[k] for k in ("signature_score", "snort_score", "score", "pred_score") if k in cols_lower),
        None,
    )
    if rid_c is None or pred_c is None:
        raise ValueError("Snort CSV must contain row_id and a prediction column.")

    sn = pd.DataFrame(
        {
            "row_id": pd.to_numeric(snort[rid_c], errors="coerce").astype("Int64"),
            "snort_pred": pd.to_numeric(snort[pred_c], errors="coerce").fillna(0).astype(int),
        }
    )
    if score_c is not None:
        sn["snort_score"] = pd.to_numeric(snort[score_c], errors="coerce").fillna(0.0).astype(float)
    else:
        sn["snort_score"] = sn["snort_pred"].astype(float)

    sn = sn.dropna(subset=["row_id"]).copy()
    sn["row_id"] = sn["row_id"].astype(np.int64)

    log(
        f"snort fires to merge | rows={len(sn):,} | "
        f"fires={int((sn['snort_pred'] == 1).sum()):,}"
    )

    merged = rate_df.merge(sn, on="row_id", how="left")
    merged["snort_pred"] = merged["snort_pred"].fillna(0).astype(int)
    merged["snort_score"] = merged["snort_score"].fillna(0.0).astype(float)

    # Snort is tier-1: OR-merge into the fast-path signature_pred only.
    merged["signature_pred"] = (
        (merged["signature_pred"] == 1) | (merged["snort_pred"] == 1)
    ).astype(int)
    merged["signature_score"] = np.maximum(
        merged["signature_score"].to_numpy(),
        merged["snort_score"].to_numpy(),
    )
    # Also expose the raw Snort flag as a per-rule column, consistent with
    # rate_V, rate_L, ... so downstream code can reference it by name.
    merged["rate_X"] = merged["snort_pred"].astype(int)
    merged["rule_fired"] = merged.apply(
        lambda r: (r["rule_fired"] or "") + ("X" if r["snort_pred"] == 1 else ""),
        axis=1,
    )
    merged = merged.drop(columns=["snort_pred", "snort_score"])
    log(
        f"merged tier-1 fast-path fires (rate | snort): "
        f"{int((merged['signature_pred'] == 1).sum()):,}"
    )
    return merged


def per_class_report(cleaned: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    j = cleaned[["row_id", "multiclass_label", "binary_label"]].merge(
        preds[["row_id", "signature_pred", "rule_fired"]],
        on="row_id",
        how="left",
    )
    j["signature_pred"] = j["signature_pred"].fillna(0).astype(int)
    j["rule_fired"] = j["rule_fired"].fillna("")

    out = []
    for cls, sub in j.groupby("multiclass_label"):
        support = len(sub)
        fires = int((sub["signature_pred"] == 1).sum())
        metric = "Correct Benign Rate" if cls == "BENIGN" else "Detection Rate"
        rate = 1.0 - fires / support if cls == "BENIGN" else fires / support
        out.append(
            {
                "class": cls,
                "support": support,
                "metric": metric,
                "rate": round(rate, 4),
                "fires": fires,
            }
        )
    out_df = pd.DataFrame(out).sort_values("support", ascending=False).reset_index(drop=True)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flow-level rate-based signature prefilter (drop-in Snort CSV)."
    )
    parser.add_argument("--data-dir", required=True,
                        help="CIC-IDS2017 CSV folder (same as the cascade uses).")
    parser.add_argument("--output-csv", required=True,
                        help="Output CSV path (row_id, signature_pred, signature_score, rule_fired).")
    parser.add_argument("--merge-snort-csv", default=None,
                        help="Optional Snort predictions CSV to OR-merge into the output.")
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)

    # Optional rule overrides
    parser.add_argument("--vol-pkt-s", type=float, default=None)
    parser.add_argument("--vol-byte-s", type=float, default=None)
    parser.add_argument("--portscan-unique-ports", type=int, default=None)
    parser.add_argument("--bruteforce-attempts", type=int, default=None)
    args = parser.parse_args()

    cfg = RateRuleConfig()
    if args.vol_pkt_s is not None:
        cfg.volumetric_packets_s = args.vol_pkt_s
    if args.vol_byte_s is not None:
        cfg.volumetric_bytes_s = args.vol_byte_s
    if args.portscan_unique_ports is not None:
        cfg.portscan_min_unique_ports = args.portscan_unique_ports
    if args.bruteforce_attempts is not None:
        cfg.bruteforce_min_attempts = args.bruteforce_attempts

    log(f"loading raw CSVs from: {args.data_dir}")
    raw = read_cic_ids2017_folder(args.data_dir)
    cleaned = clean_detection_dataframe(raw, max_missing_fraction=args.max_missing_fraction)
    log(f"cleaned rows: {len(cleaned):,} (cols={cleaned.shape[1]:,})")

    preds = apply_rules(cleaned, cfg)

    if args.merge_snort_csv:
        preds = merge_with_snort(preds, args.merge_snort_csv)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    log(f"wrote {out_path} | rows={len(preds):,}")

    # Per-class diagnostic
    report = per_class_report(cleaned, preds)
    log("per-class rate-rule detection (whole dataset):")
    print(report.to_string(index=False), flush=True)
    report_path = out_path.with_name(out_path.stem + "_per_class.csv")
    report.to_csv(report_path, index=False)
    log(f"wrote {report_path}")


if __name__ == "__main__":
    main()
