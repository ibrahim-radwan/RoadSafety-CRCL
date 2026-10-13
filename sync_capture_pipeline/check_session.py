"""
check_session.py — validate a 3-sensor capture_session.py session.

Run it on return, BEFORE the next session. A clean exit is the condition
for keeping the data. (The radar + D455 version is in legacy/.)

WHAT THIS ANSWERS
    1. Did every recorder finish cleanly?
    2. Is each sensor's data complete and readable on disk?
         radar  - every .bin exactly one frame, losses and alignment
         camera - SVO2 finalized and verified, frame ledger, dropped frames
         LiDAR  - PCAP readable, packet counter continuous, counts agree
    3. Do camera and LiDAR cover the whole radar recording on
       CLOCK_MONOTONIC (so every radar frame CAN be paired)?

WHAT THIS DOES NOT DO
    Pairing. That is align_session.py, which fits the clocks and writes
    sync/pairs.csv. This tool only decides whether the session is worth
    aligning. It is stdlib-only so it runs anywhere, with any Python.

FAILURES vs WARNINGS
    A failure means data is missing or unusable at session level. A warning
    is worth knowing but already flagged per frame (e.g. a radar frame with
    lost bytes is marked in pairs.csv). Thresholds are options below.

USAGE
    python check_session.py --session ./sessions/run05
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import struct
from pathlib import Path

FRAME_BYTES = 256 * 64 * 48 * 4      # radar frame, r075m
ADC_BYTES = 1456                     # radar payload per packet
MSOP_ID = b"\x55\xaa\x05\x5a"
MSOP_CNT_OFFSET, MSOP_CNT_MOD = 12, 1 << 16


def load(path: Path) -> list:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


class Report:
    def __init__(self):
        self.failures, self.warnings = [], []

    def section(self, title):
        print(f"\n[{title}]")

    def line(self, text):
        print(f"    {text}")

    def fail(self, text):
        self.failures.append(text)
        print(f"    FAIL: {text}")

    def warn(self, text):
        self.warnings.append(text)
        print(f"    warn: {text}")


# ── session ─────────────────────────────────────────────────────────────────
def check_manifest(s: Path, r: Report) -> list:
    r.section("1 Session")
    meta = load_json(s / "session.json")
    if not meta:
        r.fail("session.json missing - launcher did not finish")
        return []
    sensors = meta.get("sensors", [])
    r.line(f"started     : {meta.get('started_local')}   note: {meta.get('note') or '-'}")
    r.line(f"sensors     : {', '.join(sensors)}")
    r.line(f"stop reason : {meta.get('stop_reason')}")
    r.line(f"exit codes  : {meta.get('exit_codes')}")
    th = meta.get("thermal") or {}
    if th.get("samples"):
        r.line(f"board temp  : tj max {th['tj_max_c']:.1f} C, fan max {th.get('fan_max_pct')}% "
               f"({th['samples']} samples)")
        if th["tj_max_c"] >= th.get("warn_c", 80.0):
            r.warn(f"board reached tj {th['tj_max_c']:.0f} C (warn {th.get('warn_c', 80.0):.0f} C, "
                   f"throttling near 99 C) - improve airflow/shade")
    for name in sensors:
        if meta.get("exit_codes", {}).get(name) != 0:
            r.fail(f"{name} recorder exit code {meta.get('exit_codes', {}).get(name)}")
    if str(meta.get("stop_reason", "")).startswith(("startup", "lidar", "camera", "free", "radar exited")):
        r.fail(f"session ended abnormally: {meta.get('stop_reason')}")
    return sensors


# ── radar ───────────────────────────────────────────────────────────────────
def check_radar(s: Path, r: Report, a) -> dict:
    r.section("2 Radar")
    rows = load(s / "radar" / "frames.csv")
    bins = sorted((s / "radar").glob("frame_*.bin"))
    if not rows:
        r.fail("radar/frames.csv missing or empty")
        return {}
    if "lost_bytes" not in rows[0]:
        r.fail("frames.csv is from the pre-3-sensor receiver: t_burst is one frame late")
    r.line(f"frames      : {len(rows)} in manifest, {len(bins)} .bin on disk")
    if len(bins) != len(rows):
        r.fail(f"radar manifest/disk mismatch: {len(rows)} vs {len(bins)}")
    wrong = [p.name for p in bins if p.stat().st_size != FRAME_BYTES]
    missing = [x["filename"] for x in rows if not (s / "radar" / x["filename"]).is_file()]
    r.line(f"wrong size  : {len(wrong)}   (expected {FRAME_BYTES:,} B each)   manifest rows without file: {len(missing)}")
    if wrong:
        r.fail(f"{len(wrong)} radar frames have the wrong size, e.g. {wrong[:3]}")
    if missing:
        r.fail(f"{len(missing)} manifest rows have no file")

    lost = [int(x.get("lost_bytes") or 0) for x in rows]
    n_lost = sum(1 for v in lost if v)
    r.line(f"loss        : {n_lost} frames with lost bytes ({sum(lost):,} B zero-filled)")
    if n_lost:
        msg = f"{n_lost} radar frames ({100 * n_lost / len(rows):.2f}%) have lost bytes (flagged in pairs.csv)"
        (r.fail if 100 * n_lost / len(rows) > a.max_radar_loss_pct else r.warn)(msg)
    offs = [int(x.get("burst_offset_bytes") or 0) for x in rows]
    misaligned = sum(1 for v in offs if abs(v) > ADC_BYTES)
    r.line(f"alignment   : {misaligned} frames whose burst is > 1 packet from a frame boundary")
    if misaligned:
        r.fail(f"{misaligned} radar frames misaligned with transmission bursts - frames may straddle two radar frames")

    t = [float(x["t_burst_monotonic"]) for x in rows]
    gaps = [b - a_ for a_, b in zip(t, t[1:])]
    n_bursts = len(set(t))
    if gaps and n_bursts < len(rows) / 2:
        # No idle gap between frames: the DCA1000 was sending continuously,
        # i.e. it could not keep up and buffered radar data. Arrival times no
        # longer say when a frame was measured (seen 2026-10-11: ~2,970 pkt/s
        # for 102 s, buffer full, then the DCA stopped capturing).
        tf = [float(x["t_first_monotonic"]) for x in rows]
        span = tf[-1] - tf[0]
        r.line(f"cadence     : {n_bursts} transmit bursts for {len(rows)} frames - DCA1000 streamed with NO idle gap; "
               f"{len(rows)} frames ({len(rows) * 0.125:.1f} s at 8 fps) arrived over {span:.1f} s")
        r.fail("DCA1000 output throttled/backlogged: frame arrival times are not acquisition times, and radar data "
               "after the DCA buffer filled is missing. Power-cycle the radar + DCA1000 and record again")
    elif gaps:
        med = st.median(gaps)
        worst = max(gaps) / med
        r.line(f"cadence     : {1 / med:.3f} fps, worst interval {worst:.2f}x median")
        if worst > 1.8:
            r.fail(f"radar interval {worst:.2f}x the median: frames missing")
    return {"t0": t[0], "t1": t[-1]} if t else {}


# ── camera ──────────────────────────────────────────────────────────────────
def check_camera(s: Path, r: Report, a) -> dict:
    r.section("3 Camera (ZED 2i)")
    c = s / "camera"
    status = load_json(c / "capture_status.json")
    if not status:
        r.fail("camera/capture_status.json missing - recorder did not finalize")
        return {}
    svo = c / "camera.svo2"
    r.line(f"status      : complete={status.get('complete')}  phase={status.get('phase')}  "
           f"error={status.get('error')}")
    r.line(f"SVO2        : {svo.stat().st_size / 1e6 if svo.is_file() else 0:.1f} MB, frame count verified="
           f"{status.get('finalized_svo_frame_count_verified')}, timestamps verified="
           f"{status.get('finalized_svo_timestamps_verified')}")
    if not status.get("complete"):
        r.fail(f"camera recording incomplete: {status.get('error')}")
    if not svo.is_file() or svo.stat().st_size == 0:
        r.fail("camera.svo2 missing or empty")
    if not (status.get("finalized_svo_frame_count_verified") and status.get("finalized_svo_timestamps_verified")):
        r.fail("SVO2 frame count / timestamps were not verified after finalization")
    if not list((c / "calibration").glob("SN*.conf")):
        r.fail("camera factory calibration (calibration/SN*.conf) missing")

    rows = load(c / "capture" / "frames.csv")
    n, fps = status.get("recorded_frames") or 0, status.get("observed_fps") or 0
    dropped = status.get("estimated_missing_camera_frames") or 0
    r.line(f"frames      : {n} recorded, {len(rows)} in frames.csv, {fps:.2f} fps observed")
    if len(rows) != n:
        r.fail(f"frames.csv rows ({len(rows)}) != recorded frames ({n})")
    pct = 100 * dropped / max(1, n + dropped)
    r.line(f"dropped     : {dropped} frames ({pct:.2f}%)")
    if dropped:
        (r.fail if pct > a.max_camera_drop_pct else r.warn)(
            f"{dropped} camera frames dropped ({pct:.2f}%) - see capture/timing/camera_gaps.csv")
    if not rows:
        return {}
    if "host_receipt_mono_ns" not in rows[0]:
        r.fail("frames.csv has no host_receipt_mono_ns - cannot be put on CLOCK_MONOTONIC")
        return {}
    # Capture time on CLOCK_MONOTONIC, as align_session.py computes it.
    off = [int(x["host_receipt_ns"]) - int(x["host_receipt_mono_ns"]) for x in rows]
    steps = sum(1 for a_, b in zip(off, off[1:]) if abs(b - a_) > 1_000_000)
    r.line(f"wall clock  : {steps} steps > 1 ms during the session")
    if steps:
        r.warn(f"wall clock stepped {steps} times (NTP?) - conversion stays exact per frame, but check")
    t0 = (int(rows[0]["timestamp_ns"]) - off[0]) / 1e9
    t1 = (int(rows[-1]["timestamp_ns"]) - off[-1]) / 1e9
    return {"t0": t0, "t1": t1}


# ── LiDAR ───────────────────────────────────────────────────────────────────
def read_pcap(path: Path):
    with path.open("rb") as fh:
        if struct.unpack("<I", fh.read(24)[:4])[0] != 0xA1B23C4D:
            raise ValueError("not a nanosecond PCAP")
        while True:
            rec = fh.read(16)
            if len(rec) < 16:
                return
            sec, nsec, incl, _ = struct.unpack("<IIII", rec)
            frame = fh.read(incl)
            if len(frame) < incl:
                raise ValueError("truncated final record")
            yield sec * 1_000_000_000 + nsec, struct.unpack_from("!H", frame, 36)[0], frame[42:]


def check_lidar(s: Path, r: Report, a) -> dict:
    r.section("4 LiDAR (RS-Fairy)")
    meta = load_json(s / "lidar" / "metadata.json")
    if not meta:
        r.fail("lidar/metadata.json missing")
        return {}
    r.line(f"recorder    : complete={meta.get('complete')}  error={meta.get('error')}  "
           f"duration {meta.get('duration_s')} s")
    if not meta.get("complete"):
        r.fail(f"LiDAR recording incomplete: {meta.get('error')}")
    pcap = s / "lidar" / "lidar.pcap"
    if not pcap.is_file():
        r.fail("lidar.pcap missing")
        return {}
    ports = meta.get("ports", {})
    counts = {name: 0 for name in ports}
    by_port = {v: k for k, v in ports.items()}
    lost, last, t0, t1 = 0, None, None, None
    try:
        for t_ns, dport, payload in read_pcap(pcap):
            name = by_port.get(dport)
            if name:
                counts[name] += 1
            if name == "msop" and payload[:4] == MSOP_ID:
                t0 = t0 if t0 is not None else t_ns
                t1 = t_ns
                cnt = struct.unpack_from(">I", payload, MSOP_CNT_OFFSET)[0]
                if last is not None:
                    step = (cnt - last) % MSOP_CNT_MOD
                    lost += max(0, step - 1)
                last = cnt
    except ValueError as exc:
        r.fail(f"lidar.pcap unreadable: {exc}")
    r.line(f"PCAP        : {pcap.stat().st_size / 1e6:.1f} MB, packets {counts}")
    if counts != meta.get("packets"):
        r.fail(f"PCAP packet counts {counts} differ from the recorder's {meta.get('packets')}")
    total = counts.get("msop", 0) + lost
    pct = 100 * lost / total if total else 0.0
    kd = meta.get("kernel_udp_drops") or {}
    r.line(f"loss        : {lost} MSOP packets ({pct:.4f}%), queue drops {meta.get('queue_drops')}, "
           f"kernel drops {kd}")
    if lost:
        (r.fail if pct > a.max_lidar_loss_pct else r.warn)(f"{lost} LiDAR packets lost ({pct:.4f}%)")
    if counts.get("msop", 0) == 0:
        r.fail("no LiDAR point packets recorded")
    return {"t0": t0 / 1e9, "t1": t1 / 1e9} if t0 else {}


# ── coverage ────────────────────────────────────────────────────────────────
def check_coverage(r: Report, radar: dict, others: dict) -> None:
    r.section("5 Coverage (CLOCK_MONOTONIC)")
    if not radar:
        r.line("no radar timing - skipped")
        return
    r.line(f"radar       : {radar['t0']:.3f} .. {radar['t1']:.3f}  ({radar['t1'] - radar['t0']:.1f} s)")
    for name, span in others.items():
        if not span:
            continue
        lead, trail = radar["t0"] - span["t0"], span["t1"] - radar["t1"]
        r.line(f"{name:<11} : leads radar by {lead:+.2f} s, trails by {trail:+.2f} s")
        if lead < 0 or trail < 0:
            r.fail(f"radar frames fall outside {name} coverage - those frames cannot be paired")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a 3-sensor capture session")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--max-radar-loss-pct", type=float, default=1.0,
                    help="fail above this %% of radar frames with lost bytes (below: warning)")
    ap.add_argument("--max-camera-drop-pct", type=float, default=1.0,
                    help="fail above this %% of dropped camera frames (below: warning)")
    ap.add_argument("--max-lidar-loss-pct", type=float, default=0.1,
                    help="fail above this %% of lost LiDAR packets (below: warning)")
    a = ap.parse_args()
    s = a.session

    print("=" * 76)
    print(f"  SESSION: {s}")
    print("=" * 76)
    r = Report()
    sensors = check_manifest(s, r)
    radar = check_radar(s, r, a) if "radar" in sensors else {}
    spans = {}
    if "camera" in sensors:
        spans["camera"] = check_camera(s, r, a)
    if "lidar" in sensors:
        spans["lidar"] = check_lidar(s, r, a)
    check_coverage(r, radar, spans)

    sizes = sum(p.stat().st_size for p in s.rglob("*") if p.is_file())
    print(f"\n    session size: {sizes / 1e9:.2f} GB")
    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    if r.failures:
        print("  FAIL - do not use this session until resolved:")
        for f in r.failures:
            print(f"    - {f}")
    else:
        print("  PASS - data is complete and the sensors cover the radar recording.")
    if r.warnings:
        print("  Warnings:")
        for w in r.warnings:
            print(f"    - {w}")
    if not r.failures:
        print("\n  Next: python align_session.py --session", s,
              "[--offsets <Stage 3 offsets.json>]")
    return 1 if r.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
