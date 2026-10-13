#!/usr/bin/env python3
"""
check_lidar_session.py — verify a record_lidar_raw.py session.

1. PACKET LEVEL (pure Python, reads lidar.pcap directly)
   Counts packets per port, re-checks MSOP counter continuity and fits the
   LiDAR clock against host CLOCK_MONOTONIC: offset, drift (ppm) and the
   receipt jitter around that fit.

2. DECODE LEVEL (rs_driver through python_wrapper_2)
   Replays the PCAP through the same decoder as live capture and checks
   frame count, frame spacing and points per frame.

USAGE
    python check_lidar_session.py dataset/rsfairy_session_<stamp>
    python check_lidar_session.py dataset/rsfairy_session_<stamp> --rate 4 --no-decode
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WRAPPER_DIR = HERE / "rs_driver" / "python_wrapper_2"

MSOP_ID = b"\x55\xaa\x05\x5a"
MSOP_CNT_OFFSET = 12
MSOP_CNT_MOD = 1 << 16   # the counter wraps 65535 -> 0
MSOP_TS_OFFSET = 20
UDP_PAYLOAD = 42


def read_pcap(path: Path):
    """Yield (t_ns, dst_port, payload) from a nanosecond Ethernet PCAP."""
    with path.open("rb") as fh:
        magic = struct.unpack("<I", fh.read(24)[:4])[0]
        if magic != 0xA1B23C4D:
            raise ValueError(f"{path} is not a nanosecond little-endian PCAP (magic 0x{magic:08X})")
        while True:
            rec = fh.read(16)
            if len(rec) < 16:
                return
            sec, nsec, incl, _ = struct.unpack("<IIII", rec)
            frame = fh.read(incl)
            if len(frame) < incl:
                return
            dport = struct.unpack_from("!H", frame, 36)[0]
            yield sec * 1_000_000_000 + nsec, dport, frame[UDP_PAYLOAD:]


def packet_check(session: Path, meta: dict) -> dict:
    ports = meta["ports"]
    counts = Counter()
    host, lidar, cnt = [], [], []
    for t_ns, dport, payload in read_pcap(session / "lidar.pcap"):
        counts[dport] += 1
        if dport == ports["msop"] and payload[:4] == MSOP_ID:
            host.append(t_ns)
            cnt.append(struct.unpack_from(">I", payload, MSOP_CNT_OFFSET)[0])
            lidar.append(int.from_bytes(payload[MSOP_TS_OFFSET:MSOP_TS_OFFSET + 6], "big")
                         + int.from_bytes(payload[MSOP_TS_OFFSET + 6:MSOP_TS_OFFSET + 10], "big") * 1e-6)

    host_s = np.array(host, dtype=np.int64)
    lidar_s = np.array(lidar)
    steps = np.diff(np.array(cnt, dtype=np.int64)) % MSOP_CNT_MOD
    lost = int((steps[steps >= 1] - 1).sum())

    # LiDAR clock vs host monotonic: host = a * lidar + b. Residuals are
    # receipt jitter (network + scheduling), always >= the true latency floor.
    t_host = (host_s - host_s[0]) / 1e9
    t_lidar = lidar_s - lidar_s[0]
    a, b = np.polyfit(t_lidar, t_host, 1)
    resid_ms = (t_host - (a * t_lidar + b)) * 1e3

    by_name = {name: counts.get(port, 0) for name, port in ports.items()}
    span = t_host[-1]
    return {
        "packets": by_name,
        "rates_hz": {k: round(v / span, 1) for k, v in by_name.items()},
        "msop_lost": lost,
        "msop_counter_steps": dict(Counter(steps.tolist()).most_common(3)),
        "span_host_s": round(span, 3),
        "span_lidar_s": round(t_lidar[-1], 3),
        "lidar_drift_ppm": round((a - 1) * 1e6, 2),
        "offset_monotonic_minus_lidar_s": round(host_s[0] / 1e9 - lidar_s[0] + b, 6),
        "receipt_jitter_ms": {"p50": round(float(np.percentile(resid_ms, 50)), 3),
                              "p99": round(float(np.percentile(resid_ms, 99)), 3),
                              "max": round(float(resid_ms.max()), 3),
                              "min": round(float(resid_ms.min()), 3)},
    }


def decode_check(session: Path, rate: float) -> dict:
    sys.path.insert(0, str(WRAPPER_DIR))
    from lidar import Lidar

    clouds, imus = [], []
    lock = threading.Lock()

    def on_cloud(points, timestamp, seq):
        with lock:
            clouds.append((timestamp, seq, len(points)))

    def on_imu(imu):
        with lock:
            imus.append(imu["timestamp"])

    lidar = Lidar(str(session / "config_used.yaml"))
    lidar.on_pointcloud(on_cloud)
    lidar.on_imu(on_imu)
    lidar.start(pcap_path=session / "lidar.pcap", pcap_rate=rate)
    while not lidar.input_finished():
        time.sleep(0.2)
    time.sleep(1.0)  # let the decoder flush the last frame
    lidar.stop()

    ts = np.array([c[0] for c in clouds])
    n = np.array([c[2] for c in clouds])
    # The first frame of a stream is a partial rotation stamped 0.0.
    valid = ts > 0
    dt_ms = np.diff(ts[valid]) * 1e3
    imu_ts = np.array(imus)
    return {
        "frames": len(clouds),
        "frames_unstamped_dropped": int((~valid).sum()),
        "frame_dt_ms": {"median": round(float(np.median(dt_ms)), 3), "min": round(float(dt_ms.min()), 3),
                        "max": round(float(dt_ms.max()), 3)} if dt_ms.size else None,
        "frame_gaps_over_150ms": int((dt_ms > 150).sum()),
        "points_per_frame": {"min": int(n[valid].min()), "median": int(np.median(n[valid])),
                             "max": int(n[valid].max())} if valid.any() else None,
        "frames_below_full": int((n[valid] < n[valid].max()).sum()) if valid.any() else None,
        "imu_samples": len(imus),
        "imu_rate_hz": round((len(imu_ts) - 1) / (imu_ts[-1] - imu_ts[0]), 1) if len(imu_ts) > 1 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify an RS-Fairy raw recording.")
    ap.add_argument("session", type=Path)
    ap.add_argument("--rate", type=float, default=4.0, help="decode speed as a multiple of real time")
    ap.add_argument("--no-decode", action="store_true", help="packet-level checks only")
    a = ap.parse_args()

    meta = json.loads((a.session / "metadata.json").read_text())
    print(f"Session: {a.session}")
    print(f"  recorder: complete={meta['complete']} error={meta.get('error')} duration={meta.get('duration_s')} s")
    print(f"  recorder: lost={meta.get('msop_lost')} queue_drops={meta.get('queue_drops')} "
          f"kernel_drops={meta.get('kernel_udp_drops')} rcvbuf={meta.get('rcvbuf_granted')}")

    pk = packet_check(a.session, meta)
    print("\nPacket level:")
    for k, v in pk.items():
        print(f"  {k}: {v}")

    ok = meta["complete"] and pk["msop_lost"] == 0
    if not a.no_decode:
        dc = decode_check(a.session, a.rate)
        print("\nDecode level (rs_driver):")
        for k, v in dc.items():
            print(f"  {k}: {v}")
        expected = pk["span_lidar_s"] * 10
        print(f"  expected ~{expected:.0f} frames at 10 Hz")
        ok = ok and dc["frame_gaps_over_150ms"] == 0

    print(f"\nRESULT: {'PASS' if ok else 'CHECK FINDINGS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
