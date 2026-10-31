#!/usr/bin/env python3
"""
live/lidar_live.py - the LiDAR's live view: newest sweep -> the replay's top view (forward up,
vehicle-left on the left, coloured by height above our local ground), plus the brightest
reflectors, a few times a second. Nothing is written to disk.

INPUT
    live            rs_driver through our wrapper, on the live sockets, with the recorder's driver
                    config (lidar_standalone_capture/rs_driver/config/config.yaml: split_angle 180).
    --from-session  a recorded session's lidar.pcap decoded in real time (looping); its driver
                    config goes to a temporary directory, removed at exit.

BRIGHTEST REFLECTORS
    Corner reflectors return far higher intensity than anything else in the scene. Returns at or
    above the sweep's 99.9th intensity percentile (at least BRIGHT_MIN) within the view are binned
    in 0.5 m cells; the strongest cells (summed intensity, 1.5 m apart) are sent as candidates, and
    the page picks the strongest inside its range gate.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from common import (ROOT, Link, Pacer, Rate, ASPECT, add_args, bev_extent, jpeg, log, rounded)

LIVE_CONFIG = ROOT / "lidar_standalone_capture" / "rs_driver" / "config" / "config.yaml"
PX = 600                                       # image width (cells across)
BRIGHT_MIN = 100.0                             # intensity floor of a "bright" return (0..255)
N_BRIGHT = 8


def bright_spots(pts: np.ndarray, extent, near_m: float = 0.5) -> list:
    """[x, y, range, azimuth (deg, + left), peak intensity, n returns] of the brightest spots."""
    x0, x1, y0, y1 = extent
    x, y, r, inten = pts[:, 0], pts[:, 1], pts[:, 3], pts[:, 4]
    ok = (r > near_m) & (x > x0) & (x < x1) & (y > y0) & (y < y1)
    if ok.sum() < 100:
        return []
    thr = max(BRIGHT_MIN, float(np.percentile(inten[ok], 99.9)))
    b = ok & (inten >= thr)
    if not b.any():
        return []
    bx, by, bi = x[b], y[b], inten[b]
    cell = np.floor(bx / 0.5).astype(np.int64) * 100000 + np.floor((by + 500) / 0.5).astype(np.int64)
    keys, inv = np.unique(cell, return_inverse=True)
    score = np.bincount(inv, weights=bi)
    out = []
    for k in np.argsort(-score):
        m = inv == k
        w = bi[m] / bi[m].sum()
        cx, cy = float(w @ bx[m]), float(w @ by[m])
        if any(np.hypot(cx - o[0], cy - o[1]) < 1.5 for o in out):
            continue
        out.append([cx, cy, float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cy, cx))), float(bi[m].max()),
                    int(m.sum())])
        if len(out) >= N_BRIGHT:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LiDAR live view (no saving)")
    add_args(ap)
    ap.add_argument("--no-detect", action="store_true", help="no ground / object detection (colour by raw height)")
    a = ap.parse_args()
    link = Link("lidar", a.max_secs, {"range_m": a.range_m})
    from perception.core.lidar_io import decode_config, load_wrapper
    from sources import render_lidar           # the replay's LiDAR view
    newest = {"n": 0, "pts": None, "t": 0.0}
    nlock = threading.Lock()

    def on_cloud(points, ts, seq):
        p = np.asarray(points)
        if len(p):
            with nlock:
                newest.update(n=newest["n"] + 1, pts=p.astype(np.float32, copy=True), t=time.monotonic())

    tmp = None
    lidar = None
    try:
        Lidar = load_wrapper()
        if a.from_session:
            tmp = Path(tempfile.mkdtemp(prefix="live_lidar_"))
            cfg = decode_config(a.from_session, tmp / "decode.yaml")
            pcap = Path(a.from_session) / "lidar" / "lidar.pcap"
            log(f"lidar: playing {pcap}")
        else:
            cfg, pcap = LIVE_CONFIG, None
        detector = None
        if not a.no_detect:
            from perception.lidar import LidarDetector
            detector = LidarDetector()

        def open_driver():
            d = Lidar(str(cfg))
            d.on_pointcloud(on_cloud)
            d.start(pcap_path=str(pcap) if pcap else None, pcap_rate=1.0)
            return d

        lidar = open_driver()
        rate, pacer = Rate(), Pacer(a.hz, link.stop)
        seen, t_start = 0, time.monotonic()
        while link.running():
            if pcap is not None and lidar.input_finished():
                lidar.stop()                       # end of the recording: start over
                lidar = open_driver()
            with nlock:
                n, pts, t = newest["n"], newest["pts"], newest["t"]
            if n == seen:
                if n == 0 and time.monotonic() - t_start > 5:
                    link.send({"waiting": True, "sweeps": 0})
                    t_start = time.monotonic()
                link.stop.wait(0.02)
                continue
            seen = n
            tic = time.perf_counter()
            R = float(link.control.get("range_m", 20.0))
            ground = None
            n_obj = 0
            if detector is not None:
                d = detector.detect(pts, labels=False)
                ground, n_obj = d["ground_z"], len(d["objects"])
            view = dict(mode="top", range_m=R, color="height", near_m=0.5, aspect=ASPECT, px=PX)
            rgb, _, shown = render_lidar(pts, view, ground)
            img = rgb[::-1, ::-1]                 # rows: forward up; columns: vehicle-left on the left
            ms = (time.perf_counter() - tic) * 1e3
            rate.tick()
            link.send({"t": t, "extent": bev_extent(R), "ms": round(ms, 1), "hz": round(rate.hz, 1),
                       "sweeps": n, "points": shown, "objects": n_obj,
                       "ground_z": None if ground is None or not np.isfinite(ground) else round(float(ground), 2),
                       # [x, y, range, azimuth (deg, + left), peak intensity, n returns], strongest first
                       "bright": rounded(bright_spots(pts, bev_extent(R)))},
                      jpeg(img, 80))
            pacer.wait()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        link.error(str(exc))
        return 1
    finally:
        if lidar is not None:
            lidar.stop()
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        log("lidar: live view stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
