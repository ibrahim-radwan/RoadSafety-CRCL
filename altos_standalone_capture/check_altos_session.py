#!/usr/bin/env python3
"""
check_altos_session.py — verify a record_altos_raw.py session, and optionally
export its point clouds.

Reads altos.pcap directly (it does not trust frames.csv), rebuilds every
frame and reports:
  - frames: count, frame_id gaps (whole frames lost), incomplete frames
  - rate: frame spacing on the host clock, and on the radar clock
  - clocks: radar time against host CLOCK_MONOTONIC: offset, drift (ppm),
    and the receipt jitter around that fit
  - points: per frame, and the spread of range, speed, azimuth, elevation, SNR

USAGE
    python check_altos_session.py dataset/altos_session_<stamp>
    python check_altos_session.py dataset/altos_session_<stamp> --export     # -> points.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import altos_v4 as av


def pct(a, qs=(0, 50, 95, 100)):
    return " / ".join("%.3g" % v for v in np.percentile(a, qs)) if len(a) else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("session", type=Path)
    ap.add_argument("--export", action="store_true",
                    help="write points.npz (see altos_v4.frames_to_arrays for the layout)")
    a = ap.parse_args()

    pcap = a.session / "altos.pcap"
    meta_path = a.session / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    asm = av.FrameAssembler()
    frames = []
    packets = 0
    for t_ns, pkt in av.read_pcap(pcap):
        packets += 1
        frames.extend(asm.feed(pkt, t_ns))
    frames.extend(asm.flush())
    if not frames:
        print("FAIL: no frames in %s" % pcap)
        return 1

    d = av.frames_to_arrays(frames)
    n = len(frames)
    host = d["t_host_last_ns"] * 1e-9            # a frame is usable once its last packet is in
    radar = d["radar_time"]
    pts = d["points"]

    print("Session  %s" % a.session)
    if meta:
        inst = (meta.get("radar_settings", {}).get("install") or [{}])[0]
        print("Radar    %s %s fw %s | profile %s | fps limit %s | sync %s | threshold %s" % (
            meta.get("sensor"), meta.get("serial"), meta.get("firmware"), inst.get("profile"),
            inst.get("fps_limit"), inst.get("sync_mode"),
            (inst.get("rpvb_control_overrides") or {}).get("threshold_coeff")))
        print("Recorder complete=%s, error=%s" % (meta.get("complete"), meta.get("error")))

    print("\nFrames   %d from %d packets | id %d..%d | lost (id gaps) %d | incomplete %d | "
          "late packets %d | bad packets %d | last frame cut by stop %d | radar restarts %d" % (
              n, packets, d["frame_id"][0], d["frame_id"][-1], asm.missing_ids, asm.incomplete,
              asm.out_of_order, asm.bad_packets, asm.truncated_end, asm.restarts))
    dur = host[-1] - host[0]
    if n > 1:
        dh = np.diff(host) * 1e3
        dr = np.diff(radar) * 1e3
        print("Rate     %.3f frames/s over %.1f s" % ((n - 1) / dur, dur))
        print("Spacing  host ms   min/median/p95/max %s" % pct(dh))
        print("         radar ms  min/median/p95/max %s" % pct(dr))
        # radar clock against host clock: host = a + b * radar
        b, a0 = np.polyfit(radar - radar[0], host - host[0], 1)
        resid = (host - host[0]) - (a0 + b * (radar - radar[0]))
        line = "Clocks   radar date %s UTC at the first frame" % (
            np.datetime64(int(radar[0] * 1e9), "ns").astype("datetime64[ms]"))
        if "start_realtime_ns" in meta:
            host_utc = (meta["start_realtime_ns"] + (d["t_host_last_ns"][0] - meta["start_monotonic_ns"])) * 1e-9
            gap = host_utc - radar[0]
            line += "; host UTC - radar = %.3f s (%.1f days)%s" % (
                gap, gap / 86400, "" if abs(gap) < 1 else ": radar clock not synced, use host time")
        print(line)
        print("         drift %+.1f ppm | receipt jitter around the fit: std %.2f ms, max %.2f ms" % (
            (b - 1) * 1e6, resid.std() * 1e3, np.abs(resid).max() * 1e3))
        print("         latency check: last packet - first packet per frame, ms %s" % pct(
            (d["t_host_last_ns"] - d["t_host_first_ns"]) * 1e-6))

    valid = d["valid"]
    print("\nPoints   per frame min/median/p95/max %s | total %d | invalid (NaN fields) %d = %.2f%%"
          % (pct(d["n_points"]), len(pts), (~valid).sum(), 100.0 * (~valid).mean() if len(pts) else 0))
    good = pts[valid]
    if len(good):
        print("  over valid points, min / median / p95 / max:")
        print("  range m         %s" % pct(good[:, 0]))
        print("  doppler m/s     %s" % pct(good[:, 1]))
        print("  azimuth deg     %s   (+ = left)" % pct(np.degrees(good[:, 2])))
        print("  elevation deg   %s   (raw sign)" % pct(np.degrees(good[:, 3])))
        print("  snr             %s" % pct(good[:, 4]))

    if a.export:
        out = a.session / "points.npz"
        np.savez_compressed(out, **d)
        print("\nWrote %s" % out)

    ok = asm.incomplete == 0 and asm.missing_ids == 0 and asm.bad_packets == 0 and asm.restarts == 0
    print("\n%s" % ("PASS" if ok else "CHECK: frames were lost or incomplete, or the radar restarted"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
