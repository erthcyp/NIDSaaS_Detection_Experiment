from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


GID_SID_REV_RE = re.compile(r"\[(\d+):(\d+):(\d+)\]")
PRIORITY_RE = re.compile(r"\[Priority:\s*(\d+)\]")
PROTO_RE = re.compile(r"\{([^}]+)\}")
ENDPOINT_RE = re.compile(
    r"(?P<src_ip>\S+?)(?::(?P<src_port>\d+))?\s+->\s+(?P<dst_ip>\S+?)(?::(?P<dst_port>\d+))?$"
)


def parse_fast_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None

    # Example:
    # 07/03-18:55:58.598308 [**] [1:1000009:1] "TEST ANY IP" [**] [Priority: 0] {TCP} 8.8.8.8:80 -> 192.168.1.2:50000

    # Split around [**]
    parts = [p.strip() for p in line.split("[**]")]
    if len(parts) < 3:
        return None

    # Part 0: timestamp
    timestamp = parts[0].strip()
    if not timestamp:
        return None

    # Part 1: [gid:sid:rev] message
    middle = parts[1].strip()
    m_gsr = GID_SID_REV_RE.search(middle)
    if not m_gsr:
        return None

    gid, sid, rev = m_gsr.groups()

    # message = text after [gid:sid:rev]
    msg = middle[m_gsr.end():].strip().strip('"')

    # Part 2+: may contain [Priority: x], {PROTO}, endpoints, maybe extra fields
    tail = " ".join(parts[2:]).strip()

    m_pri = PRIORITY_RE.search(tail)
    priority = m_pri.group(1) if m_pri else None

    m_proto = PROTO_RE.search(tail)
    proto = m_proto.group(1) if m_proto else None

    # Endpoint text is everything after the last }
    endpoint_text = tail
    if m_proto:
        endpoint_text = tail[m_proto.end():].strip()

    m_ep = ENDPOINT_RE.search(endpoint_text)
    if not m_ep:
        return None

    row = {
        "timestamp": timestamp,
        "gid": gid,
        "sid": sid,
        "rev": rev,
        "message": msg,
        "priority": priority,
        "proto": proto,
        "src_ip": m_ep.group("src_ip"),
        "src_port": m_ep.group("src_port"),
        "dst_ip": m_ep.group("dst_ip"),
        "dst_port": m_ep.group("dst_port"),
    }
    return row


def parse_fast_file(path: Path) -> list[dict]:
    rows = []
    total = 0
    parsed = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            row = parse_fast_line(line)
            if row is None:
                continue
            row["source_file"] = str(path)
            row["pcap_name"] = path.parent.name
            rows.append(row)
            parsed += 1

    print(f"[parse_fast_alerts] parsed {parsed}/{total} lines from {path}", flush=True)
    return rows


def iter_alert_fast_files(input_dir: Path):
    seen = set()
    for p in sorted(input_dir.rglob("alert_fast.txt")):
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with raw snort outputs")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    files = list(iter_alert_fast_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No alert_fast files found under: {input_dir}")

    print(f"[parse_fast_alerts] found {len(files)} alert files", flush=True)

    rows = []
    for f in files:
        print(f"[parse_fast_alerts] parsing: {f}", flush=True)
        rows.extend(parse_fast_file(f))

    df = pd.DataFrame(rows)

    if not df.empty:
        for c in ["gid", "sid", "rev", "priority", "src_port", "dst_port"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[parse_fast_alerts] saved: {output_csv}", flush=True)
    print(f"[parse_fast_alerts] rows={len(df):,}", flush=True)


if __name__ == "__main__":
    main()