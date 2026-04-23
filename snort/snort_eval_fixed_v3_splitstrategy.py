from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
    
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def log(msg: str) -> None:
    print(f"[snort_eval] {msg}", flush=True)


def normalize_protocol(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    numeric_map = {
        "6": "TCP", "6.0": "TCP",
        "17": "UDP", "17.0": "UDP",
        "1": "ICMP", "1.0": "ICMP",
        "0": None, "0.0": None,
    }
    if s in numeric_map:
        return numeric_map[s]
    text_map = {"TCP": "TCP", "UDP": "UDP", "ICMP": "ICMP"}
    return text_map.get(s, s)


def normalize_ip(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def normalize_port(x) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def get_service_port(src_port, dst_port) -> Optional[int]:
    ports = [p for p in (src_port, dst_port) if p is not None]
    if not ports:
        return None
    well_known = [p for p in ports if p <= 1024]
    return min(well_known) if well_known else min(ports)


def unordered_ip_pair(a, b):
    vals = [x for x in (a, b) if x is not None]
    if len(vals) != 2:
        return None
    return tuple(sorted(vals))


def resolve_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def infer_pcap_name_from_source_file(value) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value)
    if "Monday-WorkingHours" in s:
        return "Monday-WorkingHours"
    if "Tuesday-WorkingHours" in s:
        return "Tuesday-WorkingHours"
    if "Wednesday-workingHours" in s or "Wednesday-WorkingHours" in s:
        return "Wednesday-workingHours"
    if "Thursday-WorkingHours" in s:
        return "Thursday-WorkingHours"
    if "Friday-WorkingHours" in s:
        return "Friday-WorkingHours"
    return None


DAY_TO_DATE_2017 = {
    "Monday-WorkingHours": "2017-07-03",
    "Tuesday-WorkingHours": "2017-07-04",
    "Wednesday-workingHours": "2017-07-05",
    "Thursday-WorkingHours": "2017-07-06",
    "Friday-WorkingHours": "2017-07-07",
}


def parse_snort_timestamp(ts: str, pcap_name: Optional[str]) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    s = str(ts).strip()
    if not s:
        return pd.NaT
    try:
        month = int(s[0:2])
        day = int(s[3:5])
        clock = s[6:]
        return pd.to_datetime(f"2017-{month:02d}-{day:02d} {clock}", errors="coerce")
    except Exception:
        pass
    base = DAY_TO_DATE_2017.get(pcap_name or "")
    if base is None:
        return pd.NaT
    try:
        clock = s.split("-", 1)[1]
    except Exception:
        return pd.NaT
    return pd.to_datetime(f"{base} {clock}", errors="coerce")


def parse_flow_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    far = float(fp / max(1, (fp + tn)))

    out = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "far": far,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if scores is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def load_test_dataframe_from_project(
    data_dir: str,
    max_missing_fraction: float = 0.20,
    test_size: float = 0.20,
    val_size_from_train: float = 0.20,
    random_state: int = 42,
    drop_unknown_labels: bool = True,
    split_row_ids_path: Optional[str] = None,
    split_strategy: str = "random",
) -> pd.DataFrame:
    from load_data import load_and_prepare_detection_data  # type: ignore

    cleaned_df, splits = load_and_prepare_detection_data(
        data_dir,
        max_missing_fraction=max_missing_fraction,
        test_size=test_size,
        val_size_from_train=val_size_from_train,
        random_state=random_state,
        drop_unknown_labels=drop_unknown_labels,
        split_strategy=split_strategy,
    )

    if split_row_ids_path:
        payload = json.loads(Path(split_row_ids_path).read_text(encoding="utf-8"))
        test_ids = None
        for k in ("test_row_ids", "test_ids", "test"):
            if k in payload and isinstance(payload[k], list):
                test_ids = payload[k]
                break
        if test_ids is None:
            for k, v in payload.items():
                if "test" in str(k).lower() and isinstance(v, list):
                    test_ids = v
                    break
        if test_ids is None:
            raise ValueError(f"Could not find test-row list inside: {split_row_ids_path}")
        if "row_id" not in cleaned_df.columns:
            raise ValueError("cleaned_df does not contain 'row_id', cannot apply split_row_ids.json")
        test_df = cleaned_df[cleaned_df["row_id"].isin(test_ids)].copy().reset_index(drop=True)
        log(f"loaded test set from split_row_ids.json | rows={len(test_df):,}")
        return test_df

    test_df = splits.test_all.copy().reset_index(drop=True)
    log(f"loaded test set from load_data.py split [{split_strategy}] | rows={len(test_df):,}")
    return test_df


def build_test_index(
    test_df: pd.DataFrame,
    time_col: str,
    proto_col: str,
    src_ip_col: str,
    src_port_col: str,
    dst_ip_col: str,
    dst_port_col: str,
    pcap_col: str,
) -> Tuple[pd.DataFrame, Dict[Tuple, List[int]], Dict[Tuple, List[int]]]:
    df = test_df.copy()
    df["_pcap_name"] = df[pcap_col].map(infer_pcap_name_from_source_file)
    df["_proto"] = df[proto_col].map(normalize_protocol)
    df["_src_ip"] = df[src_ip_col].map(normalize_ip)
    df["_dst_ip"] = df[dst_ip_col].map(normalize_ip)
    df["_src_port"] = df[src_port_col].map(normalize_port)
    df["_dst_port"] = df[dst_port_col].map(normalize_port)
    df["_ts"] = parse_flow_timestamp(df[time_col])

    key_to_indices: Dict[Tuple, List[int]] = {}
    fallback_index: Dict[Tuple, List[int]] = {}

    def add_key(k: Tuple, idx: int) -> None:
        key_to_indices.setdefault(k, []).append(idx)

    def add_fallback_key(k: Tuple, idx: int) -> None:
        fallback_index.setdefault(k, []).append(idx)

    for idx, row in df.iterrows():
        pcap = row["_pcap_name"]
        proto = row["_proto"]
        src_ip = row["_src_ip"]
        dst_ip = row["_dst_ip"]
        src_port = row["_src_port"]
        dst_port = row["_dst_port"]
        if pcap is None or proto is None:
            continue

        add_key((pcap, proto, src_ip, src_port, dst_ip, dst_port), idx)
        add_key((pcap, proto, dst_ip, dst_port, src_ip, src_port), idx)

        service_port = get_service_port(src_port, dst_port)
        ip_pair = unordered_ip_pair(src_ip, dst_ip)
        if ip_pair is not None and service_port is not None:
            add_fallback_key((pcap, proto, ip_pair, service_port), idx)

    return df, key_to_indices, fallback_index


def match_alerts_to_test_rows(
    alerts_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_to_indices: Dict[Tuple, List[int]],
    fallback_index: Dict[Tuple, List[int]],
    time_window_seconds: float = 2.0,
    ignore_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.zeros(len(test_df), dtype=np.int8)
    scores = np.zeros(len(test_df), dtype=np.float32)
    exact_hits = 0
    fallback_hits = 0

    for _, a in alerts_df.iterrows():
        pcap = a["_pcap_name"]
        proto = a["_proto"]
        src_ip = a["_src_ip"]
        dst_ip = a["_dst_ip"]
        src_port = a["_src_port"]
        dst_port = a["_dst_port"]
        ts = a["_ts"]

        if pcap is None or proto is None:
            continue

        candidate_indices = key_to_indices.get((pcap, proto, src_ip, src_port, dst_ip, dst_port), [])
        used_fallback = False

        if not candidate_indices:
            service_port = get_service_port(src_port, dst_port)
            ip_pair = unordered_ip_pair(src_ip, dst_ip)
            if ip_pair is not None and service_port is not None:
                candidate_indices = fallback_index.get((pcap, proto, ip_pair, service_port), [])
                used_fallback = bool(candidate_indices)

        if not candidate_indices:
            continue

        matched_any = False
        if ignore_time or pd.isna(ts):
            for idx in candidate_indices:
                preds[idx] = 1
                scores[idx] = 1.0
                matched_any = True
        else:
            low = ts - pd.Timedelta(seconds=time_window_seconds)
            high = ts + pd.Timedelta(seconds=time_window_seconds)
            for idx in candidate_indices:
                t = test_df.iloc[idx]["_ts"]
                if pd.isna(t):
                    continue
                if low <= t <= high:
                    preds[idx] = 1
                    scores[idx] = 1.0
                    matched_any = True

        if matched_any:
            if used_fallback:
                fallback_hits += 1
            else:
                exact_hits += 1

    log(f"alert matching summary | exact_hits={exact_hits:,}, fallback_hits={fallback_hits:,}, ignore_time={ignore_time}")
    return preds, scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate filtered Snort alerts against the CIC-IDS2017 test split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--snort-alerts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-row-ids", default=None)
    parser.add_argument("--time-window-seconds", type=float, default=2.0)
    parser.add_argument("--ignore-time", action="store_true", default=False)
    parser.add_argument("--max-missing-fraction", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size-from-train", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        default="random",
        choices=["random", "temporal", "temporal_by_file"],
    )
    parser.add_argument("--drop-unknown-labels", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading project test split ...")
    test_df = load_test_dataframe_from_project(
        data_dir=args.data_dir,
        max_missing_fraction=args.max_missing_fraction,
        test_size=args.test_size,
        val_size_from_train=args.val_size_from_train,
        random_state=args.random_state,
        drop_unknown_labels=args.drop_unknown_labels,
        split_row_ids_path=args.split_row_ids,
        split_strategy=args.split_strategy,
    )

    time_col = resolve_first_existing(test_df, ["timestamp", "Timestamp"])
    proto_col = resolve_first_existing(test_df, ["protocol", "Protocol"])
    src_ip_col = resolve_first_existing(test_df, ["source_ip", "Source IP", "src_ip"])
    src_port_col = resolve_first_existing(test_df, ["source_port", "Source Port", "src_port"])
    dst_ip_col = resolve_first_existing(test_df, ["destination_ip", "Destination IP", "dst_ip"])
    dst_port_col = resolve_first_existing(test_df, ["destination_port", "Destination Port", "dst_port"])
    source_file_col = resolve_first_existing(test_df, ["source_file", "sourcefile", "source"])

    missing = {
        "time_col": time_col,
        "proto_col": proto_col,
        "src_ip_col": src_ip_col,
        "src_port_col": src_port_col,
        "dst_ip_col": dst_ip_col,
        "dst_port_col": dst_port_col,
        "source_file_col": source_file_col,
    }
    missing_names = [k for k, v in missing.items() if v is None]
    if missing_names:
        raise ValueError(f"Could not resolve required columns in test dataframe: {missing_names}")

    binary_col = resolve_first_existing(test_df, ["binary_label"])
    label_col = resolve_first_existing(test_df, ["label", "Label", "multiclass_label"])
    if binary_col is None and label_col is None:
        raise ValueError("Could not find binary_label or label column in test dataframe.")

    log("building test index ...")
    test_norm, key_to_indices, fallback_index = build_test_index(
        test_df=test_df,
        time_col=time_col,
        proto_col=proto_col,
        src_ip_col=src_ip_col,
        src_port_col=src_port_col,
        dst_ip_col=dst_ip_col,
        dst_port_col=dst_port_col,
        pcap_col=source_file_col,
    )
    log(f"index built | keyed_flows={len(key_to_indices):,}, fallback_keys={len(fallback_index):,}")

    log("loading filtered snort alerts ...")
    alerts_df = pd.read_csv(args.snort_alerts)
    if alerts_df.empty:
        raise ValueError("The supplied Snort alerts CSV is empty.")

    alerts_df["_pcap_name"] = alerts_df["pcap_name"].astype(str)
    alerts_df["_proto"] = alerts_df["proto"].map(normalize_protocol)
    alerts_df["_src_ip"] = alerts_df["src_ip"].map(normalize_ip)
    alerts_df["_dst_ip"] = alerts_df["dst_ip"].map(normalize_ip)
    alerts_df["_src_port"] = alerts_df["src_port"].map(normalize_port)
    alerts_df["_dst_port"] = alerts_df["dst_port"].map(normalize_port)
    alerts_df["_ts"] = [parse_snort_timestamp(ts, pcap) for ts, pcap in zip(alerts_df["timestamp"], alerts_df["pcap_name"])]

    before = len(alerts_df)
    alerts_df = alerts_df.drop_duplicates(
        subset=["pcap_name", "timestamp", "sid", "proto", "src_ip", "src_port", "dst_ip", "dst_port"]
    ).reset_index(drop=True)
    log(f"alerts loaded | raw={before:,}, deduped={len(alerts_df):,}")

    log("matching alerts to test flows ...")
    y_pred, scores = match_alerts_to_test_rows(
        alerts_df=alerts_df,
        test_df=test_norm,
        key_to_indices=key_to_indices,
        fallback_index=fallback_index,
        time_window_seconds=args.time_window_seconds,
        ignore_time=args.ignore_time,
    )

    if binary_col is not None:
        y_true = test_norm[binary_col].astype(int).to_numpy()
    else:
        y_true = (test_norm[label_col].astype(str).str.upper() != "BENIGN").astype(int).to_numpy()

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, scores=scores)

    metrics_df = pd.DataFrame([{
        "paper_model": "Signature-Snort",
        "model": "snort_signature",
        **metrics,
        "alerts_input_rows": int(before),
        "alerts_deduped_rows": int(len(alerts_df)),
        "matched_test_rows": int(y_pred.sum()),
        "time_window_seconds": float(args.time_window_seconds),
        "ignore_time": bool(args.ignore_time),
    }])

    pred_df = test_norm.copy()
    pred_df["signature_pred"] = y_pred
    pred_df["signature_score"] = scores

    metrics_path = out_dir / "snort_signature_metrics.csv"
    preds_path = out_dir / "snort_signature_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(preds_path, index=False)

    log(f"saved metrics: {metrics_path}")
    log(f"saved predictions: {preds_path}")
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
