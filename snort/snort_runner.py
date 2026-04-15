from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def run_snort_on_pcap(
    snort_exe: Path,
    snort_conf: Path,
    pcap_path: Path,
    out_dir: Path,
    extra_rules: Path | None = None,
    packet_limit: int | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    alert_file = out_dir / "alert_fast.txt"
    console_log = out_dir / f"{pcap_path.stem}_snort_stdout.txt"

    cmd = [
        str(snort_exe),
        "--daq-dir", "/usr/local/lib/daq_s3/lib/daq",
        "-r", str(pcap_path),
        "-c", str(snort_conf),
        "-A", "alert_fast",
        "-l", str(out_dir),
        "--lua", "alert_fast = {file = true}",
        "-q",
    ]

    if extra_rules is not None:
        cmd.extend(["-R", str(extra_rules)])

    if packet_limit is not None:
        cmd.extend(["-n", str(packet_limit)])

    print(f"[snort_runner] running: {pcap_path.name}", flush=True)
    print(f"[snort_runner] cmd: {' '.join(cmd)}", flush=True)

    start = time.time()
    with open(console_log, "w", encoding="utf-8", errors="ignore") as f:
        completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    print(f"[snort_runner] return_code={completed.returncode} | elapsed={elapsed:.2f}s", flush=True)

    if alert_file.exists():
        print(f"[snort_runner] alert_file={alert_file} | size={alert_file.stat().st_size} bytes", flush=True)
    else:
        print(f"[snort_runner] alert_file missing: {alert_file}", flush=True)

    return completed.returncode


def iter_pcaps(pcap_dir: Path):
    for ext in ("*.pcap", "*.pcapng", "*.cap"):
        for p in sorted(pcap_dir.glob(ext)):
            if p.is_file():
                yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snort-exe", required=True, help="Path to snort executable")
    parser.add_argument("--pcap-dir", required=True, help="Directory containing pcap files")
    parser.add_argument("--rules", required=True, help="Path to snort.lua")
    parser.add_argument("--out-dir", required=True, help="Directory for raw snort outputs")
    parser.add_argument("--extra-rules", default=None, help="Optional extra rules file, e.g. local.rules")
    parser.add_argument("--packet-limit", type=int, default=None, help="Optional packet limit per pcap")
    args = parser.parse_args()

    snort_exe = Path(args.snort_exe)
    pcap_dir = Path(args.pcap_dir)
    snort_conf = Path(args.rules)
    out_dir = Path(args.out_dir)
    extra_rules = Path(args.extra_rules) if args.extra_rules else None

    out_dir.mkdir(parents=True, exist_ok=True)

    pcaps = list(iter_pcaps(pcap_dir))
    if not pcaps:
        raise FileNotFoundError(f"No PCAP files found under: {pcap_dir}")

    print(f"[snort_runner] found {len(pcaps)} pcaps", flush=True)

    failures = 0
    for pcap in pcaps:
        per_pcap_out = out_dir / pcap.stem
        rc = run_snort_on_pcap(
            snort_exe=snort_exe,
            snort_conf=snort_conf,
            pcap_path=pcap,
            out_dir=per_pcap_out,
            extra_rules=extra_rules,
            packet_limit=args.packet_limit,
        )
        if rc != 0:
            failures += 1

    print(f"[snort_runner] done | failures={failures}", flush=True)


if __name__ == "__main__":
    main()