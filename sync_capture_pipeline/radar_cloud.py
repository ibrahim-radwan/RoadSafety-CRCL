#!/usr/bin/env python3
"""
radar_cloud.py - pre-compute the FILTERED radar point cloud of a session with
the team's own processing chain, unchanged:
cascaded_radar_capture_pipeline/processing/radar_processing.process_frame
(MTI, CFAR, sidelobe/consistency filters, azimuth + elevation, xyz).

With CuPy installed (cupy-cuda11x 12.3 in rel_1_py_3_8, 2026-10-24) the team's
own GPU path runs: ~0.1 s per frame instead of ~1 s on the CPU. The two paths
DIFFER, inside the team's code: process_frame asks for CASO-CFAR on the
range-Doppler map (cfar_2d_gpu(..., force_caso=True)), but without CuPy the
fallback _cfar_2d_cpu(heat, cfg) drops force_caso and runs cfg.cfar_method
(OS). The team's own environment has CuPy, so the GPU path is what they get;
clouds computed before 2026-10-24 used the CPU path. The path used is recorded
in radar_cloud_filtered.json. --cpu forces the CPU fallback (the old output).

    python radar_cloud.py --session /mnt/sdcard/sessions/<name> [--workers N] [--cpu]

Writes <session>/sync/radar_cloud_filtered.npz:
    frame_t      [F]     radar frame times t_ref (s, CLOCK_MONOTONIC, Stage 3 offset applied)
    frame_file   [F]     radar frame file names
    start        [F+1]   points of frame k are rows start[k]:start[k+1]
    team_xyz     [N, 7]  process_frame's xyz_cloud, AS RETURNED:
                         x (lateral, = r cos(el) sin(az)), y (forward), z, power, doppler, range, az
    xyz          [N, 3]  the same points in the pipeline frame used everywhere else:
                         x forward, y LEFT, z up. y = -team_x: the team's lateral axis is
                         positive to the RIGHT (their convention, not an error; verified
                         against camera and LiDAR on freewalk_ref_02). z is the team's
                         elevation estimate (unverified).
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


def _init(cfg_path: str, radar_dir: str, cpu: bool = False):
    if cpu:
        sys.modules["cupy"] = None                         # the team's gpu_accel falls back to the CPU
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
    ap.add_argument("--workers", type=int, default=None,
                    help="processes (default: 3 on the GPU path, CPU cores - 2 on the CPU path)")
    ap.add_argument("--cpu", action="store_true", help="the team's CPU fallback (OS-CFAR on range-Doppler)")
    a = ap.parse_args()
    gpu = False
    if not a.cpu:
        try:
            import cupy  # noqa: F401
            gpu = True
        except Exception:
            print("CuPy not available: the team's CPU fallback runs (~1 s/frame, OS-CFAR on range-Doppler)")
    if a.workers is None:
        a.workers = 3 if gpu else max(1, (os.cpu_count() or 4) - 2)
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
        with Pool(a.workers, initializer=_init, initargs=(cfg_path, str(S / "radar"), not gpu)) as pool:
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
                            note=np.array("xyz: x forward, y left (= -team x: team x is right-positive), z up; team_xyz as returned "
                                          "by process_frame"))
        el = time.monotonic() - t0
        print(f"  wrote {out}: {int(n.sum()):,} points, {n.mean():.1f}/frame (max {n.max()}), "
              f"frames with points {int((n > 0).sum())}/{len(n)}, {el:.0f} s ({1e3 * el / len(n):.0f} ms/frame)",
              flush=True)
        meta = {"frames": len(n), "points": int(n.sum()), "seconds": round(el, 1), "workers": a.workers,
                "cfg": Path(cfg_path).name, "team_path": "GPU (CuPy)" if gpu else "CPU fallback",
                "rd_cfar": "CASO (as process_frame requests)" if gpu else
                "OS (the CPU fallback ignores force_caso)", "computed": time.strftime("%Y-%m-%d")}
        (S / "sync" / "radar_cloud_filtered.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
