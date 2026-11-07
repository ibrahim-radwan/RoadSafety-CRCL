#!/usr/bin/env python3
"""
live/lidar_live.py - the LiDAR's live view: newest sweep -> the replay's top view (forward up,
vehicle-left on the left, coloured by height above our local ground), plus the brightest
reflectors, a few times a second. Nothing is written to disk, except a calibration capture on the
page's request (below).

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

BACKGROUND AND CALIBRATION CAPTURE (2026-11-07; perception/core/reflector.py)
    The background is the voxels (0.15 m, dilated by one) occupied in its sweeps (vehicle frame).
    {"reflector_only": true} draws only the returns in voxels the background did not have.
    "Save background" / "Save position" (common.Capture) write the next CAPTURE_SWEEPS sweeps to
    <dir>/lidar/: sweeps.npz (points in the LiDAR's OWN frame, before the mounting transform, so a
    new extrinsic can be applied later; counts and t = CLOCK_MONOTONIC at receipt), meta.json (the
    transform used); a background also background_voxels.npy, and is used at once; a position also
    reflector.json and reflector_points.npz (lidar_reflector: the persistent change, vehicle frame,
    with its LiDAR-frame position too), against the set's saved background if there is one.
    Height window for the reflector: REFLECTOR_Z (LiDAR frame), which leaves out the floor of the
    2026-11 rig (~1.1 m below the LiDAR).
"""
from __future__ import annotations

import argparse
import collections
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from common import (ROOT, Background, Capture, Link, Pacer, Rate, ASPECT, add_args, bev_extent, jpeg, json_dump, log,
                    rounded)

LIVE_CONFIG = ROOT / "lidar_standalone_capture" / "rs_driver" / "config" / "config.yaml"
PX = 600                                       # image width (cells across)
BRIGHT_MIN = 100.0                             # intensity floor of a "bright" return (0..255)
N_BRIGHT = 8
BG_SWEEPS = 20                                 # sweeps in the background (voxel occupancy)
VOX = 0.15                                     # background voxel (m); dilated by one voxel
CHANGE_CELL = 0.3                              # new returns are grouped in cells of this size (m)
CHANGE_MIN = 3                                 # ... with at least this many returns
CHANGE_SWEEPS = 5                              # new returns pooled over the last sweeps (a far reflector
                                               # gives ~1 return per sweep at 8 m)


CAPTURE_SWEEPS = 20                            # sweeps per calibration capture (2 s)
REFLECTOR_Z = (-0.9, 2.5)                      # reflector height window (m, LiDAR z)


def background_voxels(sweeps: list) -> np.ndarray:
    """Sorted voxel keys occupied in any background sweep, dilated by one voxel."""
    from perception.core.reflector import background_voxels as bv
    return bv(sweeps, VOX)


def new_returns(pts: np.ndarray, ref: np.ndarray, extent, near_m: float = 0.5) -> np.ndarray:
    """The returns of a sweep in voxels the background did not have, within the view."""
    from perception.core.reflector import new_points
    x0, x1, y0, y1 = extent
    ok = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    q = pts[ok]
    return q[new_points(q, ref, near_m, VOX)]


def changed_spots(q: np.ndarray) -> list:
    """Groups of new returns (new_returns, pooled over a few sweeps): [x, y, range, azimuth
    (deg, + left), peak intensity, n returns], most returns first. Anything new counts (people too)."""
    if len(q) < CHANGE_MIN:
        return []
    g = np.floor(q[:, :2] / CHANGE_CELL).astype(np.int64)
    cells, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    out = []
    for c in np.argsort(-cnt):
        if cnt[c] < CHANGE_MIN or len(out) >= N_BRIGHT:
            break
        m = q[inv == c]
        cx, cy = float(m[:, 0].mean()), float(m[:, 1].mean())
        if any(np.hypot(cx - o[0], cy - o[1]) < 1.0 for o in out):
            continue
        out.append([cx, cy, float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cy, cx))),
                    float(m[:, 4].max()), int(len(m))])
    return out


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
    ap.add_argument("--config", type=Path, default=LIVE_CONFIG, help="rs_driver config for the live sensor")
    a = ap.parse_args()
    link = Link("lidar", a.max_secs, {"range_m": a.range_m})
    from perception.core.extrinsic import (current_extrinsic, describe, extrinsic_for_session,
                                           lidar_points_to_vehicle)
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
            cfg, pcap = a.config, None
        # into the vehicle (radar) frame: the session's mounting when playing one, else the newest
        ext = extrinsic_for_session(a.from_session) if a.from_session else current_extrinsic()
        log(f"lidar: extrinsic {describe(ext)}")
        bg = Background(BG_SWEEPS, background_voxels)            # "Take background": in memory only
        recent = collections.deque(maxlen=CHANGE_SWEEPS)          # new returns of the last sweeps
        cap = Capture("lidar", CAPTURE_SWEEPS)                    # "Save background / position": written
        saved_bg = {}
        loaded_bg = [None]
        from perception.core import reflector as rf

        def to_own_xy(x, y):
            """Vehicle frame -> the LiDAR's own frame (inverse of the mounting transform)."""
            c, s_ = np.cos(np.radians(ext.get("yaw_deg", 0.0))), np.sin(np.radians(ext.get("yaw_deg", 0.0)))
            x, y = x - ext.get("tx_m", 0.0), y - ext.get("ty_m", 0.0)
            return c * x + s_ * y, -s_ * x + c * y

        def background_for(spec):
            d = Path(spec["dir"]).parent / "background" / "lidar" / "background_voxels.npy"
            if d.is_file():
                if str(d) not in saved_bg:
                    saved_bg[str(d)] = np.load(d)
                return saved_bg[str(d)], "saved"
            return (bg.ref, "memory") if bg.ref is not None else (None, None)

        def finish_capture(items, spec):
            own = [q for q, _, _ in items]
            veh = [v for _, _, v in items]
            meta = {"extrinsic": ext, "sweeps": len(items), "frame": "points in the LiDAR's own frame (before the extrinsic)",
                    "source": str(a.from_session) if a.from_session else "live", "kind": spec.get("kind"),
                    "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            arrays, result = {}, {}
            if spec.get("kind") == "background":
                ref = background_voxels(veh)
                bg.set(ref)
                recent.clear()
                saved_bg.clear()
                arrays["background_voxels.npy"] = ref
            else:
                ref, where = background_for(spec)
                result = {"reflector": None, "background": where}
                if ref is not None:
                    res = rf.lidar_reflector(veh, ref, float(spec.get("r_lo", 1.0)), float(spec.get("r_hi", 15.0)),
                                             z_lo=REFLECTOR_Z[0], z_hi=REFLECTOR_Z[1])
                    summary = {k: v for k, v in res.items() if k != "points"}
                    summary["background"] = where
                    if res["found"]:
                        ox, oy = to_own_xy(res["x"], res["y"])
                        summary.update(own_x=float(ox), own_y=float(oy), own_az_deg=float(np.degrees(np.arctan2(oy, ox))))
                    result["reflector"] = summary
                    meta["reflector"] = summary
                    pts = res["points"]
                    arrays["reflector_points.npz"] = dict(
                        points=np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 6), np.float32),
                        sweep=np.concatenate([np.full(len(q), i) for i, q in enumerate(pts)]).astype(np.int32)
                        if pts else np.zeros(0, np.int32),
                        columns=np.array("x y z range intensity t (vehicle frame)"))

            def write():
                d = cap.out_dir()
                np.savez_compressed(d / "sweeps.npz", points=np.concatenate(own).astype(np.float32),
                                    counts=np.array([len(q) for q in own]), t=np.array([t for _, t, _ in items]))
                for name, arr in arrays.items():
                    if name.endswith(".npz"):
                        np.savez_compressed(d / name, **arr)
                    else:
                        np.save(d / name, arr)
                if "reflector" in meta:
                    json_dump(d / "reflector.json", meta["reflector"])
                json_dump(d / "meta.json", meta)      # last: the capture is complete
                cap.saved(**result)
            cap.in_thread(write)
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
            own = pts
            pts = lidar_points_to_vehicle(pts, ext)
            cap.follow(link.control.get("capture"))
            if cap.collecting:
                done = cap.add((own[own[:, 3] > 0].copy(), t, pts[pts[:, 3] > 0]))
                if done is not None:
                    finish_capture(done, cap.spec)
            R = float(link.control.get("range_m", 20.0))
            if bg.follow(link.control.get("background")) or bg.ref is None:
                recent.clear()
            use = link.control.get("use_background")
            if use and use != loaded_bg[0] and (Path(use) / "lidar" / "meta.json").is_file():
                bg.set(np.load(Path(use) / "lidar" / "background_voxels.npy"))   # the page's set's saved one
                recent.clear()
                loaded_bg[0] = use
                log(f"lidar: background from {use}/lidar")
            bg.add(pts)
            if bg.ref is not None:
                recent.append(new_returns(pts, bg.ref, bev_extent(R)))
            ground = None
            n_obj = 0
            if detector is not None:
                d = detector.detect(pts, labels=False)
                ground, n_obj = d["ground_z"], len(d["objects"])
            view = dict(mode="top", range_m=R, color="height", near_m=0.5, aspect=ASPECT, px=PX)
            only = bool(link.control.get("reflector_only")) and bg.ref is not None
            # reflector only: just the returns the background did not have (people count too)
            rgb, _, shown = render_lidar(pts[rf.new_points(pts, bg.ref, 0.5, VOX)] if only else pts, view, ground)
            img = rgb[::-1, ::-1]                 # rows: forward up; columns: vehicle-left on the left
            ms = (time.perf_counter() - tic) * 1e3
            rate.tick()
            link.send({"t": t, "extent": bev_extent(R), "ms": round(ms, 1), "hz": round(rate.hz, 1),
                       "sweeps": n, "points": shown, "objects": n_obj,
                       "ground_z": None if ground is None or not np.isfinite(ground) else round(float(ground), 2),
                       # [x, y, range, azimuth (deg, + left), peak intensity, n returns], strongest first
                       "bright": rounded(bright_spots(pts, bev_extent(R))),
                       "extrinsic": describe(ext), "background": bg.state(),
                       # new returns since the background, most returns first (same columns)
                       "changed": rounded(changed_spots(np.concatenate(recent)) if recent else []),
                       "reflector_only": only, "capture": cap.state()},
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
