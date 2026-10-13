#!/usr/bin/env python3
"""
radar_cloud.py - pre-compute the FILTERED radar point cloud of a session with
the team's own processing chain, unchanged:
cascaded_radar_capture_pipeline/processing/radar_processing.process_frame
(MTI, CFAR, sidelobe/consistency filters, azimuth + elevation, xyz).

process_frame takes ~1 s per frame on the CPU (its CFAR runs as the CPU
fallback; no CuPy), so it cannot run live in the replay at 8 frames/s. It is
run once per session over several CPU cores and saved; the replay reads it.

    python radar_cloud.py --session /mnt/sdcard/sessions/<name> [--workers 10]

Writes <session>/sync/radar_cloud_filtered.npz:
    frame_t      [F]     radar frame times t_ref (s, CLOCK_MONOTONIC, Stage 3 offset applied)
    frame_file   [F]     radar frame file names
    start        [F+1]   points of frame k are rows start[k]:start[k+1]
    team_xyz     [N, 7]  process_frame's xyz_cloud, AS RETURNED:
                         x (lateral, = r cos(el) sin(az)), y (forward), z, power, doppler, range, az
    xyz          [N, 3]  the same points in the pipeline frame used everywhere else:
                         x forward, y LEFT, z up. y = -team_x corrects the processing's
                         mirrored azimuth sign (verified against camera and LiDAR on
                         freewalk_ref_02). z is the team's elevation estimate (unverified).
    doppler      [N]     m/s (processing convention)
    power        [N]     dB
Needs align_session.py to have been run (sync/radar_timeline.csv).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")         # one core per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                         # noqa: E402

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent / "cascaded_radar_capture_pipeline"
_W = {}


def _init(cfg_path: str, radar_dir: str):
    sys.path.insert(0, str(TEAM))
    from processing.radar_config import parse_calibration, parse_cfg_file
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = parse_cfg_file(cfg_path)
        corr = parse_calibration(cfg_path, cfg.num_tx, cfg.num_rx)
    cfg.quiet_mode = True
    _W.update(cfg=cfg, corr=corr, dir=radar_dir)


def _one(fn: str):
    from processing.radar_processing import process_frame
    from processing.radar_reader import read_frame
    with contextlib.redirect_stdout(io.StringIO()):
        cube = read_frame(os.path.join(_W["dir"], fn), _W["cfg"], 0)
        res = process_frame(cube, _W["cfg"], _W["corr"], verbose=False)
    xyz = res.xyz_cloud
    return np.zeros((0, 7), np.float32) if xyz is None else np.asarray(xyz, np.float32).reshape(-1, 7)


def main() -> int:
    ap = argparse.ArgumentParser(description="Filtered radar point cloud (team pipeline) for a session")
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import radar_firmware
    for S in a.session:
        rows = list(csv.DictReader((S / "sync" / "radar_timeline.csv").open()))
        if not rows:
            raise SystemExit(f"{S}: run align_session.py first")
        files = [r["filename"] for r in rows]
        cfg_path = str(radar_firmware.cfg_for_session(S))
        print(f"=== {S}: {len(files)} frames, {a.workers} workers, cfg {Path(cfg_path).name}", flush=True)
        t0 = time.monotonic()
        clouds = []
        with Pool(a.workers, initializer=_init, initargs=(cfg_path, str(S / "radar"))) as pool:
            for k, c in enumerate(pool.imap(_one, files, chunksize=4)):
                clouds.append(c)
                if k % 500 == 0:
                    el = time.monotonic() - t0
                    print(f"  {k}/{len(files)}  {el:.0f} s", flush=True)
        n = np.array([len(c) for c in clouds])
        team = np.concatenate(clouds) if n.sum() else np.zeros((0, 7), np.float32)
        xyz = np.c_[team[:, 1], -team[:, 0], team[:, 2]].astype(np.float32)
        out = S / "sync" / "radar_cloud_filtered.npz"
        np.savez_compressed(out, frame_t=np.array([float(r["t_ref"]) for r in rows]), frame_file=np.array(files),
                            start=np.r_[0, np.cumsum(n)], team_xyz=team, xyz=xyz, doppler=team[:, 4],
                            power=team[:, 3],
                            note=np.array("xyz: x forward, y left (team x mirrored), z up; team_xyz as returned "
                                          "by process_frame"))
        el = time.monotonic() - t0
        print(f"  wrote {out}: {int(n.sum()):,} points, {n.mean():.1f}/frame (max {n.max()}), "
              f"frames with points {int((n > 0).sum())}/{len(n)}, {el:.0f} s ({1e3 * el / len(n):.0f} ms/frame)",
              flush=True)
        meta = {"frames": len(n), "points": int(n.sum()), "seconds": round(el, 1), "workers": a.workers,
                "cfg": Path(cfg_path).name}
        (S / "sync" / "radar_cloud_filtered.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
