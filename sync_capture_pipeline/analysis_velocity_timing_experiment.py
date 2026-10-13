"""Experiment: radar-vs-LiDAR timing from VELOCITY (radar Doppler vs LiDAR range rate).

Range comparisons are biased by where each sensor 'sees' the walker (plate,
torso, limbs, nearest surface) and that bias can flip with walking direction.
Velocity is the same for every part of the body moving rigidly with the walker,
so a velocity match should be free of that geometry bias.

usage: velocity_timing.py <session> [<session> ...]
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

PIPE = Path("/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers/sync_capture_pipeline")
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE / "replay"))
import measure_sync_offset as m          # noqa: E402
import radar_firmware as rf              # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import sources                       # noqa: E402

WINDOW = 30.0


def radar_velocity(S, tr, files, lt, lr):
    """Per radar frame: power-weighted Doppler velocity in the range bins around the
    LiDAR target range (moving-only map, |v| > 0.15 m/s)."""
    with contextlib.redirect_stdout(io.StringIO()):
        rp = sources.RadarProcessor(np.linspace(1, 20, 8), np.linspace(-5, 5, 8), rf.cfg_for_session(S))
    v_axis = np.asarray(rp.dop_axis)
    rng_axis = rp.range_axis
    ok = np.isfinite(lr)
    out = np.full(len(files), np.nan)
    for k, fn in enumerate(files):
        if not (lt[ok][0] <= tr[k] <= lt[ok][-1]):
            continue
        r0 = np.interp(tr[k], lt[ok], lr[ok])
        rd, _ = rp.process(np.fromfile(S / "radar" / fn, dtype=np.int16))
        p = 10 ** (rd / 10)                               # [doppler, range]
        rb = (rng_axis > r0 - 0.4) & (rng_axis < r0 + 0.8)
        if not rb.any():
            continue
        prof = p[:, rb].sum(axis=1)
        prof = prof - np.median(prof)                     # floor
        prof[np.abs(v_axis) < 0.15] = 0
        prof = np.clip(prof, 0, None)
        if prof.sum() <= 0:
            continue
        out[k] = float((prof * v_axis).sum() / prof.sum())
    return out


def lidar_rate(lt, lr):
    ok = np.isfinite(lr)
    t, r = lt[ok], lr[ok]
    # only contiguous stretches (gaps <= 0.3 s) are differentiated
    rate = np.full(len(t), np.nan)
    breaks = np.r_[0, np.nonzero(np.diff(t) > 0.3)[0] + 1, len(t)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a >= 7:
            rs = savgol_filter(r[a:b], 7, 2)
            rate[a:b] = np.gradient(rs, t[a:b])
    return t, rate


for s in sys.argv[1:]:
    S = Path(s)
    rows = list(csv.DictReader((S / "sync" / "radar_timeline.csv").open()))
    tr = np.array([float(r["t_ref"]) for r in rows]); files = [r["filename"] for r in rows]
    d = np.load(S / "sync" / "timing_lidar_track.npz"); lt, lr = d["t"], d["r"]
    rv = radar_velocity(S, tr, files, lt, lr)
    tvl, lv = lidar_rate(lt, lr)
    print(f"\n=== {S.name}  radar frames with velocity {np.isfinite(rv).sum()}/{len(rv)}")
    t0 = tr[0]; res_all = []
    for w0 in np.arange(0, tr[-1] - t0, WINDOW):
        rs = (tr - t0 >= w0) & (tr - t0 < w0 + WINDOW) & np.isfinite(rv)
        ls = (tvl - t0 >= w0 - 1) & (tvl - t0 < w0 + WINDOW + 1) & np.isfinite(lv)
        if rs.sum() < 60 or ls.sum() < 60 or np.sqrt(np.nanmean(lv[ls] ** 2)) < 0.4:
            continue
        fit = m.best_shift(tr[rs], rv[rs], tvl[ls], lv[ls], scale=True, min_samples=40)
        if fit is None or fit["at_search_edge"]:
            print(f"  {w0:5.0f}-{w0 + WINDOW:<5.0f} no fit"); continue
        sig = m.bootstrap_sigma(tr[rs], rv[rs], tvl[ls], lv[ls], True, 40, fit["tau_s"])
        print(f"  {w0:5.0f}-{w0 + WINDOW:<5.0f} residual {fit['tau_s'] * 1e3:+7.1f} ms +/- {sig * 1e3:5.1f}"
              f"   scale {fit['alpha']:+.2f}  rms {fit['rms']:.2f} m/s  n {fit['samples']}")
        if np.isfinite(sig) and sig < 15 and 0.5 < abs(fit["alpha"]) < 2:
            res_all.append((fit["tau_s"] * 1e3, sig * 1e3))
    if res_all:
        r = np.array(res_all)
        w = 1 / np.maximum(r[:, 1], 1) ** 2
        mean = (w * r[:, 0]).sum() / w.sum()
        chi2 = np.sum(((r[:, 0] - mean) / np.maximum(r[:, 1], 1)) ** 2) / max(1, len(r) - 1)
        print(f"  SUMMARY: {len(r)} windows, weighted mean {mean:+.1f} ms +/- {1 / np.sqrt(w.sum()):.1f}, "
              f"median {np.median(r[:, 0]):+.1f}, spread {np.ptp(r[:, 0]):.1f}, reduced chi2 {chi2:.2f}")
