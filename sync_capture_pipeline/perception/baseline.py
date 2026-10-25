#!/usr/bin/env python3
"""
baseline.py - the frozen reference for merging the old radar chain into this library.

Two kinds of reference:

golden   per-frame outputs on a fixed set of frames, for equivalence tests while porting:
           the old chain (cascaded_radar_capture_pipeline/processing.process_frame) on BOTH of
           its paths - CPU (no CuPy: OS-CFAR on range-Doppler) and GPU (CuPy: CASO) - and our
           detector (perception/radar.py). Saved in <session>/sync/baseline/ (not in git: tens
           of MB per session):
             golden_team_cpu.npz, golden_team_gpu.npz   RD_raw, RD_mti, RA, RE, rd_det, ra_det,
                                                        rd_thresh, ra_thresh [frames, ...];
                                                        xyz [N, 7] indexed by xyz_start [frames + 1]
             golden_ours.npz                            rd, ra [frames, ...]; det [N, DET_COLS]
                                                        indexed by det_start
             golden.json                                frames, git commit, cfg, calibration hash
scores   the LiDAR-referenced scores of every session (run_perception.py's metrics) for our
         pipeline and for the old chain's filtered clouds, in perception/baseline/scores.json
         (in git, stamped with the git commit).
compare  scores the sessions again from their current outputs and prints the change against
         scores.json: the regression check before any algorithm replaces another.

    python perception/baseline.py golden  --session <s> [<s> ...] [--frames 20]
    python perception/baseline.py scores  --session <s> [<s> ...] [--write]
    python perception/baseline.py compare --session <s> [<s> ...]

Inputs per session: sync/perception_{radar,lidar}.npz (run_perception.py),
sync/radar_cloud_filtered.npz (radar_cloud.py; its .json says which path made it) and, when
present, sync/radar_cloud_filtered_cpu.npz (the old CPU-path cloud kept for comparison),
sync/timing_lidar_track.npz (make_report.py).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
REPO = PIPELINE.parent
TEAM = REPO / "cascaded_radar_capture_pipeline"
SCORES = HERE / "baseline" / "scores.json"
sys.path.insert(0, str(PIPELINE))
from perception.core.calibration import file_md5 as file_hash                 # noqa: E402
from perception.core.session import Session                                    # noqa: E402

TEAM_FIELDS = ("RD_raw", "RD_mti", "RA", "RE", "rd_det", "ra_det", "rd_thresh", "ra_thresh")
# sessions used to measure calibration/radar_rx_phase.json: their radar scores are in-sample
CALIBRATION_SESSIONS = {"stage3_02", "stage3_03"}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


# ── golden ──────────────────────────────────────────────────────────────────
def pick_frames(S: Path, n: int) -> list:
    """n frames: 80% evenly over frames with a LiDAR target at 2-20 m, 20% over frames without."""
    sess = Session(S)
    tr, t = sess.target, sess.radar.t
    j = np.clip(np.searchsorted(tr["t"], t), 1, len(tr["t"]) - 1)
    j = np.where(np.abs(tr["t"][j] - t) < np.abs(tr["t"][j - 1] - t), j, j - 1)
    has = (np.abs(tr["t"][j] - t) < 0.08) & np.isfinite(tr["r"][j]) & (tr["r"][j] > 2) & (tr["r"][j] < 20)
    tgt, empty = np.flatnonzero(has), np.flatnonzero(~has)
    n_t = n - n // 5 if len(empty) else n
    pick = list(tgt[np.linspace(0, len(tgt) - 1, n_t).astype(int)]) if len(tgt) else []
    if len(empty):
        pick += list(empty[np.linspace(0, len(empty) - 1, n - n_t).astype(int)])
    return sorted(set(int(k) for k in pick))


def _team_golden(S: Path, frames: list, path: str, out: Path):
    """Runs in its own process: the CPU path needs CuPy hidden before the team code is imported."""
    if path == "cpu":
        sys.modules["cupy"] = None
    sys.path.insert(0, str(TEAM))
    from processing import gpu_accel
    from processing.radar_processing import process_frame
    from perception.core.calibration import load_radar_config
    from perception.core.radar_io import read_frame
    if (path == "gpu") != bool(gpu_accel.GPU_AVAILABLE):
        raise SystemExit(f"team chain: asked for the {path} path but GPU_AVAILABLE={gpu_accel.GPU_AVAILABLE}")
    sess = Session(S)
    cfg, corr = load_radar_config(sess.cfg_path)
    cfg.quiet_mode = True
    acc = {k: [] for k in TEAM_FIELDS}
    xyz, ms = [], []
    for k in frames:
        with contextlib.redirect_stdout(io.StringIO()):
            cube = read_frame(str(sess.radar.path(k)), cfg, 0)
            t0 = time.perf_counter()
            res = process_frame(cube, cfg, corr, verbose=False)
            ms.append(1e3 * (time.perf_counter() - t0))
        for f in TEAM_FIELDS:
            v = getattr(res, f)
            acc[f].append(np.zeros(0, np.float32) if v is None else np.asarray(v))
        x = res.xyz_cloud
        xyz.append(np.zeros((0, 7), np.float32) if x is None else np.asarray(x, np.float32).reshape(-1, 7))
    n = np.array([len(x) for x in xyz])
    np.savez_compressed(out, frames=np.array(frames), xyz=np.concatenate(xyz), xyz_start=np.r_[0, np.cumsum(n)],
                        ms=np.array(ms), **{f: np.stack(v) for f, v in acc.items()})
    print(f"  team {path}: {len(frames)} frames, median {np.median(ms[1:] or ms):.0f} ms/frame -> {out.name}")


def golden(S: Path, n: int):
    from perception.radar import RX_PHASE_FILE, RadarDetector, rx_phase_for
    d = S / "sync" / "baseline"
    d.mkdir(exist_ok=True)
    frames = pick_frames(S, n)
    print(f"=== {S.name}: {len(frames)} golden frames")
    for path in ("cpu", "gpu"):
        subprocess.run([sys.executable, __file__, "_team", "--session", str(S), "--path", path,
                        "--frames", ",".join(map(str, frames)), "--out", str(d / f"golden_team_{path}.npz")],
                       check=True)
    sess = Session(S)
    variant = sess.variant
    det = RadarDetector(sess.cfg_path, rx_phase_deg=rx_phase_for(variant))
    rd, ra, dets = [], [], []
    for k in frames:
        o = det.process(sess.radar.raw(k))
        rd.append(o["rd"].cpu().numpy()); ra.append(o["ra"].cpu().numpy()); dets.append(o["detections"])
    n_d = np.array([len(x) for x in dets])
    np.savez_compressed(d / "golden_ours.npz", frames=np.array(frames), rd=np.stack(rd), ra=np.stack(ra),
                        det=np.concatenate(dets), det_start=np.r_[0, np.cumsum(n_d)])
    print(f"  ours: {int(n_d.sum())} detections -> golden_ours.npz")
    meta = {"session": S.name, "frames": frames, "git_commit": git_commit(), "created": time.strftime("%Y-%m-%d"),
            "cfg": sess.cfg_path.name, "firmware": variant,
            "rx_phase_file_md5": file_hash(RX_PHASE_FILE),
            "note": "team_cpu: process_frame without CuPy (OS-CFAR on range-Doppler); team_gpu: with CuPy (CASO)"}
    (d / "golden.json").write_text(json.dumps(meta, indent=2) + "\n")


# ── scores ──────────────────────────────────────────────────────────────────
def session_scores(S: Path) -> dict:
    import run_perception as rp
    from perception.radar import C
    out = {}
    track = Session(S).target
    pr, pl = S / "sync" / "perception_radar.npz", S / "sync" / "perception_lidar.npz"
    radar = lidar = None
    if pr.is_file():
        radar = dict(np.load(pr))
        d = radar["det"]
        out["ours_radar"] = rp.score_radar_points(radar["frame_t"], radar["det_start"], d[:, C["range"]],
                                                  d[:, C["az_deg"]], track, radar["trk"])
        out["ours_radar"]["detections_per_frame"] = round(float(np.diff(radar["det_start"]).mean()), 2)
    if pl.is_file():
        lidar = dict(np.load(pl))
        out["ours_lidar"] = rp.score_lidar(lidar, track)
        out["ours_lidar"]["objects_per_sweep"] = round(float(np.diff(lidar["obj_start"]).mean()), 2)
    if radar is not None and lidar is not None:
        out["ours_radar_vs_lidar"] = rp.score_radar_vs_lidar(radar, lidar)
    frame_t = Session(S).radar.t
    for name, f in (("team", "radar_cloud_filtered"), ("team_cpu_kept", "radar_cloud_filtered_cpu")):
        cp = S / "sync" / f"{f}.npz"
        if not cp.is_file():
            continue
        c = np.load(cp)
        if len(c["start"]) != len(frame_t) + 1:
            continue
        meta = json.loads((S / "sync" / f"{f}.json").read_text()) if (S / "sync" / f"{f}.json").is_file() else {}
        key = "team_gpu" if meta.get("team_path", "").startswith("GPU") else "team_cpu"
        if name == "team_cpu_kept":
            key = "team_cpu"
        xyz = c["xyz"]
        out[key] = rp.score_radar_points(frame_t, c["start"], np.hypot(xyz[:, 0], xyz[:, 1]),
                                         np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), track)
        out[key]["detections_per_frame"] = round(float(np.diff(c["start"]).mean()), 2)
    if S.name in CALIBRATION_SESSIONS:
        out["note"] = "radar phase correction measured on this session: radar azimuth scores are in-sample"
    return out


def all_scores(sessions) -> dict:
    from perception.radar import RX_PHASE_FILE
    return {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d"),
            "rx_phase_file_md5": file_hash(RX_PHASE_FILE),
            "metrics": "see run_perception.py: detected = a detection within 0.6 m range and 10 deg of the "
                       "LiDAR walker; az_spread = median |err - bias|; team_* = the old chain's filtered "
                       "clouds (cpu: OS-CFAR, gpu: CASO on range-Doppler)",
            "sessions": {S.name: session_scores(S) for S in sessions}}


KEYS = (("ours_radar", "detected"), ("ours_radar", "az_spread_deg"), ("ours_radar", "az_bias_deg"),
        ("ours_radar", "confirmed_track_on_target"), ("ours_radar_vs_lidar", "unsupported_per_frame"),
        ("ours_lidar", "detected"), ("ours_lidar", "confirmed_track_on_target"),
        ("team_gpu", "detected"), ("team_gpu", "az_spread_deg"), ("team_cpu", "detected"),
        ("team_cpu", "az_spread_deg"))


def print_table(cur: dict, base: dict = None):
    print(f"{'session':20s} " + " ".join(f"{a.replace('ours_', '').replace('_vs_lidar', '~L')[:9]}.{b[:12]:12s}"
                                         for a, b in KEYS))
    for s, sc in cur["sessions"].items():
        cells = []
        for a, b in KEYS:
            v = sc.get(a, {}).get(b)
            if v is None:
                cells.append(f"{'-':>22s}")
                continue
            bv = None if base is None else base.get("sessions", {}).get(s, {}).get(a, {}).get(b)
            delta = "" if bv is None or bv == v else f" ({v - bv:+.3g})"
            cells.append(f"{v:>10.3g}{delta:12s}")
        print(f"{s:20s} " + " ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline for the radar-chain merge: golden frames and scores")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("golden", "scores", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--session", type=Path, nargs="+", required=True)
        if name == "golden":
            p.add_argument("--frames", type=int, default=20)
        if name == "scores":
            p.add_argument("--write", action="store_true", help=f"store in {SCORES.relative_to(PIPELINE)}")
    p = sub.add_parser("_team")                                    # internal: one team-chain path per process
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--path", choices=["cpu", "gpu"], required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "_team":
        _team_golden(a.session, [int(k) for k in a.frames.split(",")], a.path, a.out)
    elif a.cmd == "golden":
        for S in a.session:
            golden(S, a.frames)
    else:
        cur = all_scores(a.session)
        base = json.loads(SCORES.read_text()) if SCORES.is_file() else None
        print(f"current: commit {cur['git_commit']}" + ("" if base is None else
                                                         f"   baseline: commit {base['git_commit']} ({base['created']})"))
        print_table(cur, base if a.cmd == "compare" else None)
        if a.cmd == "scores" and a.write:
            SCORES.parent.mkdir(exist_ok=True)
            SCORES.write_text(json.dumps(cur, indent=2, default=float) + "\n")
            print(f"wrote {SCORES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
